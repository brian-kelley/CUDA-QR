#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/time.h>

//Scalar type and panel width
#define Scalar float

//PR is how big the square trailing update block matrix should be (per CUDA block)
//(PR^2 + 2 * PR * PC) * sizeof(Scalar) should fit in 48 KiB
#define PR 8
//PC is how many columns of A get grouped into one compressed block Householder transform
#define PC 2

//integer division a/b, rounded up
#define ceildiv(a, b) ((a) / (b) + ((a) % (b) != 0))

void printMat(Scalar* mat, int m, int n);
void dgemm(Scalar* A, Scalar* B, Scalar* C, int k, int m, int n);
void explicitQR(Scalar* A, Scalar* tau, Scalar* Q, Scalar* R, int m, int n);
void identity(Scalar* A, int m);

//print a column-major matrix row-by-row (for debugging)
void printMat(Scalar* mat, int m, int n)
{
  printf("Matrix %d x %d, row by row:\n", m, n);
  for(int i = 0; i < m; i++)
  {
    for(int j = 0; j < n; j++)
    {
      printf("%9f ", mat[j * m + i]);
    }
    putchar('\n');
  }
  putchar('\n');
}

void getPanelDims(int m, int n, int* rowPanels, int* colPanels)
{
  *colPanels = ceildiv(n, PC);
  *rowPanels = 1;
  if(m > PR)
    *rowPanels += ceildiv(m - PR, PR - PC);
}

//Do block Householder factorization of the first PC columns of mat, starting at pc
//this kernel is only meant to be run on a single block with as many threads as possible
//this is because it contains many synchronization points
__global__ void panelHouseholderKernel(Scalar* mat, Scalar* tau, Scalar* W, int m, int n, int pr, int pc)
{
  //useful to think of shared memory buffer as a stack
  //for simple dynamic allocations
  extern __shared__ Scalar sharedBuf[];
  //Preallocate all shared arrays here
  Scalar* toReduce = sharedBuf;
  Scalar* finalReduce = sharedBuf + blockDim.x;
  int finalReduceNum = 16;
  Scalar* panel = finalReduce + finalReduceNum;
  Scalar* Wshared = panel + (PR * PC);
  Scalar* Acol = Wshared + (PR * PC);
  //zero out Wshared
  for(int i = 0; i < PR * PC; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PC)
      Wshared[index] = 0;
  }
  //load panel into shared
  for(int i = 0; i < PR * PC; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PC)
    {
      int col = index / PR;
      int row = index % PR;
      panel[row + col * PR] = mat[(pc + col) * m + pr + row];
    }
  }
  __syncthreads();
  for(int col = 0; col < PC && pc + col < n; col++)
  {
    //is the panel at the bottom of A?
    bool bottomPanel = pr == m - PR;
    //does col 0 of panel cross A's diagonal?
    bool topPanel = pr <= pc;
    int vstart;
    int vend;
    if(topPanel && bottomPanel)
    {
      vstart = pc - pr + col;
      vend = PR;
    }
    else if(!topPanel && bottomPanel)
    {
      vstart = col;
      vend = PR;
    }
    else if(topPanel && !bottomPanel)
    {
      //vstart needs to be at or below A's diagonal, even if
      //panel boundaries extends above it
      vstart = pc - pr + col;
      vend = PR - PC + col + 1;
    }
    else
    {
      //neither top nor bottom panel
      vstart = col;
      vend = PR - PC + col + 1;
    }
    if(threadIdx.x == 0)
    {
      printf("Vstart: %d Vend: %d\n", vstart, vend);
    }
    Scalar leading = panel[col * PR + vstart];
    //note: when computing tau/reflectors,
    //work directly with global mat (only 2 flops per element anyway)
    //compute the inner product and norm of column
    Scalar innerProd = 0;
    {
      Scalar localInnerProd = 0;
      //use a cyclic row distribution for perfect coalesced accesses
      for(int i = vstart; i < vend; i += blockDim.x)
      {
        int index = i + threadIdx.x;
        if(index < vend)
        {
          localInnerProd += panel[index + col * PR] * panel[index + col * PR];
        }
      }
      //now, sum up the localInnerProds across the whole block
      //write the partial sums to shared, then do a simple linear reduction
      toReduce[threadIdx.x] = localInnerProd;
      __syncthreads();
      if(threadIdx.x < finalReduceNum)
      {
        localInnerProd = 0;
        for(int i = 0; i < blockDim.x; i += finalReduceNum)
        {
          int index = i + threadIdx.x;
          if(index < blockDim.x)
            localInnerProd += toReduce[index];
        }
        finalReduce[threadIdx.x] = localInnerProd;
      }
      __syncthreads();
      //now, every thread sums up finalReduce to get innerProd
      for(int i = 0; i < finalReduceNum; i++)
        innerProd += finalReduce[i];
    }
    Scalar norm = sqrt(innerProd);
    Scalar sign = (leading < 0) ? -1.0 : 1.0;
    Scalar u = leading + sign * norm;
    Scalar thisTau = sign * u / norm;
    //compute entire w vector in-place, storing it back to A subdiag
    for(int i = vstart; i < vend; i += blockDim.x)
    {
      int index = i + threadIdx.x;
      if(index == vstart)
      {
        //thread 0 uniquely responsible for setting R diagonal entry and tau
        tau[col] = thisTau;
        panel[col * PR + vstart] = -sign * norm;
      }
      else if(index < vend)
      {
        panel[col * PR + index] /= u;
      }
    }
    __threadfence_block();
    //v is now fully computed and stored back to panel
    //compute z vector using W,Y
    //each thread will compute one entry in z
    for(int i = 0; i < PR; i += blockDim.x)
    {
      int index = i + threadIdx.x;
      if(index < PR)
      {
        Scalar zval = 0;
        //set zval to v[index]
        if(index == vstart)
          zval = -thisTau;
        else if(index > vstart && index < vend)
          zval = -thisTau * panel[col * PR + index];
        //finish computing entry i of z
        //compute zval as (W * Y^T * v)(i)
        Scalar wytvi = 0;
        for(int j = vstart; j < vend; j++)
        {
          //need inner product of row i of W and row j of Y
          //this is (WY^T)(i, j)
          //use the fact that only the first col+1 columns of W and Y are nonzero
          Scalar wyt = 0;
          for(int k = 0; k < col; k++)
          {
            Scalar yval = 0;
            //find the set of rows for column k of panel
            int vstartK, vendK;
            if(topPanel && bottomPanel)
            {
              vstartK = pc - pr + k;
              vendK = PR;
            }
            else if(!topPanel && bottomPanel)
            {
              vstartK = col;
              vendK = PR;
            }
            else if(topPanel && !bottomPanel)
            {
              //vstart needs to be at or below A's diagonal, even if
              //panel boundaries extends above it
              vstartK = pc - pr + k;
              vendK = PR - PC + k+ 1;
            }
            else
            {
              //neither top nor bottom panel
              vstartK = k;
              vendK = PR - PC + k + 1;
            }
            if(j > vstartK && j < vendK)
              yval = panel[k * PR + j];
            else if(j == vstartK)
              yval = 1;
            wyt += Wshared[k * PR + index] * yval;
          }
          Scalar vval = 0;
          if(j > vstart && j < vend)
            vval = panel[col * PR + j];
          else if(j == vstart)
            vval = 1;
          wytvi += wyt * vval;
        }
        zval -= thisTau * wytvi;
        Wshared[col * PR + index] = zval;
      }
    }
    __threadfence_block();
    //apply reflector in col to remaining columns in panel
    for(int applyCol = col + 1; applyCol < PC && pc + applyCol < n; applyCol++)
    {
      //Create a copy of the updating column of A which will
      //persist while each entry is computed
      //Only the height range [vstart, m) is read, used and written back
      for(int i = vstart; i < vend; i += blockDim.x)
      {
        int index = i + threadIdx.x;
        if(index < vend)
          Acol[index] = panel[applyCol * PR + index];
      }
      __syncthreads();
      for(int applyRow = vstart; applyRow < PR; applyRow += blockDim.x)
      {
        int index = applyRow + threadIdx.x;
        if(index < PR)
        {
          Scalar val = Acol[index];
          Scalar vIndex = 0;
          if(index == col)
            vIndex = 1;
          else
            vIndex = panel[col * PR + index];
          for(int i = vstart; i < vend; i++)
          {
            Scalar vi = 0;
            if(i == col)
              vi = 1;
            else
              vi = panel[col * PR + i];
            val -= thisTau * vIndex * vi * Acol[i];
          }
          panel[applyCol * PR + index] = val;
        }
      }
    }
  }
  __syncthreads();
  //write out W and panel back to global
  for(int i = 0; i < PR * PC; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PC)
    {
      W[index] = Wshared[index];
    }
  }
  for(int i = 0; i < PR * PC; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PC)
    {
      int row = index % PR;
      int col = index / PR;
      mat[pr + row + (pc + col) * m] = panel[row + col * PR];
    }
  }
}

//all arrays passed in are preallocated global arrays be be used by all blocks
//mat is m*n, W is m*PC, z is m, and AcolGlobal is numBlocks * m
//arrays declared volatile mean entries are not cached,
//so that a threadfence_*() is sufficient to keep all values coherent
//(meaning all writes before the fence are reflected in all reads after)
__global__ void trailingUpdateKernel(Scalar* mat, Scalar* matScratch, Scalar* W, int m, int n, int pr, int pc)
{
  //All dynamic shared memory goes here
  //Is a flat 48 KiB buffer, free for use by each block
  extern __shared__ Scalar sharedBuf[];
  //determine range of rows "owned" by this thread
  //Allocate some shared arrays that all blocks will use for computations
  //Note: W is not transposed in memory (coalesce memory accesses)
  //The YW^T entries are computed as inner products of rows of Yblock and Wblock
  Scalar* Wblock = &sharedBuf[0];
  Scalar* Yblock = &Wblock[PR * PC];
  Scalar* Ablock = &Yblock[PR * PC];
  //update trailing columns of A: A = (I + YW^T)A
  //Each block reads into Wblock/Yblock/Ablock, does multiplication and writes results out to Ascratch
  int blockRow = pr;
  int blockCol = pc + PC + blockIdx.x * PR;
  //first load in Y block
  //it will stay constant for whole kernel
  for(int i = 0; i < PR * PC; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PC)
    {
      int row = index % PR;
      int col = index / PR;
      int matRow = blockRow + row;
      int matCol = blockCol + col;
      //Y's columns are simply the reflectors stored in mat's subdiagonal.
      //this reads back the implicit 0/1 entries
      Scalar yval = 0;
      if(matRow > matCol)
        yval = mat[matRow + m * matCol];
      else if(matRow == matCol)
        yval = 1;
      Yblock[row + col * PR] = yval;
    }
  }
  //copy mat values to matScratch where this block's results will be placed
  for(int i = 0; i < PR * PR; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PR)
    {
      int row = index % PR;
      int col = index / PR;
      int matRow = blockRow + row;
      int matCol = blockCol + col;
      if(matRow < m && matCol < n)
        matScratch[matRow + matCol * m] = mat[matRow + matCol * m];
    }
  }
  for(int factorBlock = 0; factorBlock < m; factorBlock += PR)
  {
    //load Wblock into shared (from the global W)
    for(int i = 0; i < PR * PC; i += blockDim.x)
    {
      int index = i + threadIdx.x;
      if(index < PR * PC)
      {
        int row = index % PR;
        int col = index / PR;
        //W is stored explicitly in global (also column major), so just read out entries
        if(factorBlock + row < m)
          Wblock[row + col * PR] = W[(factorBlock + row) + col * m];
        else
          Wblock[row + col * PR] = 0;
      }
    }
    //load block from A
    for(int i = 0; i < PR * PR; i += blockDim.x)
    {
      int index = i + threadIdx.x;
      if(index < PR * PR)
      {
        int row = index % PR;
        int col = index / PR;
        int matCol = blockCol + col;
        if(factorBlock + row < m && matCol < n)
          Ablock[row + col * PR] = mat[factorBlock + row + matCol * m];
        else
          Ablock[row + col * PR] = 0;
      }
    }
    __syncthreads();
    //compute all the updates in matScratch (one per thread until done)
    for(int i = 0; i < PR * PR; i += blockDim.x)
    {
      int index = i + threadIdx.x;
      if(index < PR * PR)
      {
        int row = index % PR;
        int col = index / PR;
        int matRow = blockRow + row;
        int matCol = blockCol + col;
        if(matRow < m && matCol < n)
        {
          //need to compute (Wblock * Yblock) * Ablock, then add entries of that product to matScratch
          //compute partial = dot product of row'th row of Yblock * Wblock^T, and col'th col of Ablock
          Scalar partial = 0;
          for(int j = 0; j < PR; j++)
          {
            Scalar ywt = 0;
            for(int k = 0; k < PC; k++)
            {
              ywt += Yblock[row + k * PR] * Wblock[j + k * PR];
            }
            partial += ywt * Ablock[j + col * PR];
          }
          matScratch[matRow + matCol * m] += partial;
        }
      }
    }
  }
}

__global__ void trailingCopyKernel(Scalar* Adev, Scalar* matScratch, int m, int n, int pr, int pc)
{
  int matRow = pr;
  int matCol = blockIdx.x * PR + (PC + pc);
  for(int i = 0; i < PR * PR; i += blockDim.x)
  {
    int index = i + threadIdx.x;
    if(index < PR * PR)
    {
      int row = i % PR;
      int col = i / PR;
      if(matRow + row < m && matCol + col < n)
      {
        Adev[matRow + row + (matCol + col) * m] = matScratch[matRow + row + (matCol + col) * m];
      }
    }
  }
}

#define HANDLE_ERROR(x) \
{\
  cudaError_t err = x; \
  if(x) {printf("CUDA error on line %i: %d\n", __LINE__, x); exit(1);} \
}

//Host wrapper for the main CUDA kernel
//No extra overhead since copies to/from device would be necessary anyway
void mmqr(Scalar* mat, Scalar* tau, int m, int n)
{
  int rowPanels, colPanels;
  getPanelDims(m, n, &rowPanels, &colPanels);
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int shmem = prop.sharedMemPerBlock;
  int maxThreads = prop.maxThreadsPerBlock;
  int factorThreads = 256;
  //int sm = prop.multiProcessorCount;
  //MUST have at least 48 KiB of shared memory for this to work in its current state
  //TODO: adapt to any amount of shared
  if(prop.sharedMemPerBlock < 48 * 1024)
  {
    //this should never actually happen
    puts("CUDA device has < 48 KiB shared memory per block!");
    exit(1);
  }
  Scalar* Adev;
  HANDLE_ERROR(cudaMalloc((void**) &Adev, m * n * sizeof(Scalar)));
  Scalar* W;
  cudaMalloc((void**) &W, PR * PC * sizeof(Scalar));
  Scalar* matScratch;
  cudaMalloc((void**) &matScratch, m * n * sizeof(Scalar));
  Scalar* tauDev;
  HANDLE_ERROR(cudaMalloc((void**) &tauDev, rowPanels * n * sizeof(Scalar)));
  HANDLE_ERROR(cudaMemcpy(Adev, mat, m * n * sizeof(Scalar), cudaMemcpyHostToDevice));
  //launch the kernel
  //
  //only use one block and fixed threads for main kernel,
  //but main kernel will itself launch several blocks to saturate FLOPs during trailing updates)
  //want one block per SM and as many threads as possible (up to 1 per row of A)
  int pcCount = 0;
  for(int pc = 0; pc < n; pc += PC)
  {
    int prCount = 0;
    for(int pr = m - PR; (pr + PR > pc) && pr >= 0; pr -= (PR-PC))
    {
      //know exactly how much shared memory each kernel needs (at runtime)
      int kernel1shared = (factorThreads + 16 + 2 * PR * PC + PR) * sizeof(Scalar);
      assert(kernel1shared <= shmem);
      printf("Kernel 1 (panel factor) needs %d bytes shared\n", kernel1shared);
      printf("Launching kernel 1...");
      Scalar* panelTau = &tauDev[(rowPanels * pcCount + prCount) * PC];
      panelHouseholderKernel<<<1, factorThreads, kernel1shared>>>(Adev, panelTau, W, m, n, pr, pc);
      puts("done");
      int changedColumns = PC;
      if(changedColumns + pc > n)
        changedColumns = n - pc;
      //now, only need to update matScratch with the entries that have changed
      cudaMemcpy(matScratch, Adev, m * n * sizeof(Scalar), cudaMemcpyDeviceToDevice);
      if(pc + PC < n)
      {
        int blocks = ceildiv(n - pc - PC, PR);
        printf("Launching block update kernel with %d blocks, updating %dx%d region\n", blocks, PR, n - pc - PC);
        puts("Note: W matrix:\n");
        Scalar* Whost = (Scalar*) malloc(PR * PC * sizeof(Scalar));
        HANDLE_ERROR(cudaMemcpy(Whost, W, PR * PC * sizeof(Scalar), cudaMemcpyDeviceToHost));
        printMat(Whost, PR, PC);
        free(Whost);
        int kernel2shared = (2 * PR * PC + PR * PR) * sizeof(Scalar);
        assert(kernel2shared <= shmem);
        printf("Kernel 2 (trailing update) needs %d bytes shared\n", kernel2shared);
        printf("Launching kernel 2...");
        trailingUpdateKernel<<<blocks, maxThreads, kernel2shared>>>(Adev, matScratch, W, m, n, pr, pc);
        puts("done");
        printf("Copying updated trail back to Adev...");
        trailingCopyKernel<<<blocks, maxThreads>>>(Adev, matScratch, m, n, pr, pc);
        puts("done");
      }
      prCount++;
    }
    pcCount++;
  }
  //retrieve A and tau
  HANDLE_ERROR(cudaMemcpy(mat, Adev, m * n * sizeof(Scalar), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(tau, tauDev, rowPanels * n * sizeof(Scalar), cudaMemcpyDeviceToHost));
  cudaFree(matScratch);
  cudaFree(W);
  cudaFree(tauDev);
  cudaFree(Adev);
}

//A = mxm identity matrix
void identity(Scalar* A, int m)
{
  for(int i = 0; i < m * m; i++)
    A[i] = 0;
  for(int i = 0; i < m; i++)
  {
    A[i + m * i] = 1;
  }
}

//From A and tau array produced by mmqr,
//explicitly find Q and R matrices
//Q is mxm, A and R are mxn
//All matrices are column-major
void explicitQR(Scalar* A, Scalar* tau, Scalar* Q, Scalar* R, int m, int n)
{
  puts("******\n\nCOMPUTING EXPLICIT QR REPRESENTATION*****\n\n");
  //first, R is simply the upper triangular part of A (including diagonal)
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if(i >= j)
        R[i * m + j] = A[i * m + j];
      else
        R[i * m + j] = 0;
    }
  }
  //next, Q is the result of applying each Householder reflector
  //(stored in subdiagonals) to I(m)
  //note: this is very expensive to do naively on host
  //first, get I(m) into Q
  identity(Q, m);
  int rowPanels, colPanels;
  getPanelDims(m, n, &rowPanels, &colPanels);
  int pcCount = 0;
  printf("Matrix is %d by %d panels.\n", rowPanels, colPanels);
  for(int pc = 0; pc < n; pc += PC)
  {
    //then bottom to top, sliding panel up by R-C each iteration
    //prCount gives row index of panel (bottom is 0)
    int prCount = 0;
    for(int pr = m - PR; (pr + PR > pc) && pr >= 0; pr -= (PR-PC))
    {
      //is the panel at the bottom of A?
      bool bottomPanel = pr == m - PR;
      //does col 0 of panel cross A's diagonal?
      bool topPanel = pr <= pc;
      for(int col = 0; col < PC && col + pc < n; col++)
      {
        printf("Applying reflector: col %d of panel %d, %d\n", col, pr, pc);
        Scalar tauVal = tau[(rowPanels * pcCount + prCount) * PC + col];
        printf("Tau for this column: %f\n", tauVal);
        //update each trailing column (pr:pr+R, pc+C:N):
        //for each column, compute HH reflectors
        //(middle panels are both top and bottom)
        int vstart;
        int vend;
        if(topPanel && bottomPanel)
        {
          vstart = pc - pr + col;
          vend = PR;
        }
        else if(!topPanel && bottomPanel)
        {
          vstart = col;
          vend = PR;
        }
        else if(topPanel && !bottomPanel)
        {
          //vstart needs to be at or below A's diagonal, even if
          //panel boundaries extends above it
          vstart = pc - pr + col;
          vend = PR - PC + col + 1;
        }
        else
        {
          //neither top nor bottom panel
          vstart = col;
          vend = PR - PC + col + 1;
        }
        Scalar* v = (Scalar*) malloc(m * sizeof(Scalar));
        //read v from subdiagonal of A
        for(int i = 0; i < m; i++)
        {
          if(i < pr + vstart || i >= pr + vend)
            v[i] = 0;
          else if(i == pr + vstart)
            v[i] = 1;
          else
            v[i] = A[(pc + col) * m + i];
        }
        //create H matrix for this reflector
        Scalar* H = (Scalar*) malloc(m * m * sizeof(Scalar));
        identity(H, m);
        for(int j = 0; j < m; j++)
        {
          for(int k = 0; k < m; k++)
          {
            H[k + j * m] -= tauVal * v[k] * v[j];
          }
        }
        //dgemm can't multiply Q by H in-place,
        //so make a persistent copy of Q
        Scalar* prevQ = (Scalar*) malloc(m * m * sizeof(Scalar));
        for(int j = 0; j < m * m; j++)
          prevQ[j] = Q[j];
        dgemm(prevQ, H, Q, m, m, m);
        free(prevQ);
        free(v);
        free(H);
      }
      prCount++;
    }
    pcCount++;
  }
}

//General dense matrix-matrix product
//A is kxm, B is mxn and C is kxn
//All matrices are column-major
void dgemm(Scalar* A, Scalar* B, Scalar* C, int k, int m, int n)
{
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < k; j++)
    {
      //compute C(j, i) as
      //row j of A dotted with column i of B
      Scalar cval = 0;
      for(int l = 0; l < m; l++)
      {
        cval += A[j + l * k] * B[l + i * m];
      }
      C[j + i * k] = cval;
    }
  }
}

int main()
{
  int m = PR;
  int n = PC * 2;
  assert(m >= n);
  //only use one device (at least, for now)
  HANDLE_ERROR(cudaSetDevice(0));
  //First, make sure device is using proper 48 KB of shared, 16 KB L1
  //during all calls to L1 kernel
  //Note that this is not the default
  HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int sm = prop.multiProcessorCount;
  printf("Testing mmqr on \"%s\"\n", prop.name);
  printf("Device has %d SMs, %zu bytes of shared, and up to %d threads per block\n", sm, prop.sharedMemPerBlock, prop.maxThreadsPerBlock);
  if(sizeof(Scalar) == 4)
  {
    HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
  }
  else if(sizeof(Scalar) == 8)
  {
    HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  }
  else
  {
    puts("Only float (32-bit) and double (64-bit) reals are supported scalar types");
    exit(1);
  }
  Scalar* A = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar* RV = (Scalar*) malloc(m * n * sizeof(Scalar));
  int rowPanels, colPanels;
  getPanelDims(m, n, &rowPanels, &colPanels);
  Scalar* tau = (Scalar*) malloc(rowPanels * n * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = (Scalar) rand() / RAND_MAX;
    RV[i] = A[i];
  }
  //puts("A matrix:\n");
  //printMat(A, m, n);
  int trials = 1;
  double elapsed = 0;
  struct timeval currentTime;
  gettimeofday(&currentTime, NULL);
  for(int i = 0; i < trials; i++)
  {
    mmqr(RV, tau, m, n);
    struct timeval nextTime;
    gettimeofday(&nextTime, NULL);
    //add to elapsed time
    elapsed += (nextTime.tv_sec + 1e-6 * nextTime.tv_usec) - (currentTime.tv_sec + 1e-6 * currentTime.tv_usec);
    currentTime = nextTime;
    //refresh RV for next trial (this isn't part of the algorithm and so isn't timed)
    if(i != trials - 1)
      memcpy(RV, A, m * n * sizeof(Scalar));
  }
  printf("Ran QR on %dx%d matrix in %f s (avg over %d)\n", m, n, elapsed / trials, trials);
  printf("tau values after QR (grid corresponding to columns within panels):\n");
  for(int j = 0; j < rowPanels; j++)
  {
    for(int i = 0; i < colPanels * PC; i++)
    {
      printf("%9f ", tau[i * rowPanels + j]);
    }
    putchar('\n');
  }
  putchar('\n');
  //printf("A raw storage after QR:\n");
  //printMat(RV, m, n);
  Scalar* Q = (Scalar*) malloc(m * m * sizeof(Scalar));
  Scalar* R = (Scalar*) malloc(m * n * sizeof(Scalar));
  explicitQR(RV, tau, Q, R, m, n);
  printf("Q:\n");
  printMat(Q, m, m);
  printf("R:\n");
  printMat(R, m, n);
  //now compute Q*R explicitly and compare to A
  Scalar* QR = (Scalar*) malloc(m * n * sizeof(Scalar));
  dgemm(Q, R, QR, m, m, n);
  //printf("QR:\n");
  //printMat(QR, m, n);
  Scalar* QRmA = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar errNorm = 0;
  for(int i = 0; i < m * n; i++)
  {
    QRmA[i] = QR[i] - A[i];
    errNorm += QRmA[i] * QRmA[i];
  }
  printf("QR-A (should be 0):\n");
  printMat(QRmA, m, n);
  free(QRmA);
  errNorm = sqrt(errNorm);
  printf("L2 norm of residual QR-A: %.9g\n", errNorm);
  free(R);
  free(Q);
  free(QR);
  free(RV);
  free(A);
  return 0;
}

