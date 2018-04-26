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
#define PR 64
//PC is how many columns of A get grouped into one compressed block Householder transform
#define PC 8

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

//all arrays passed in are preallocated global arrays be be used by all blocks
//mat is m*n, W is m*PC, z is m, and AcolGlobal is numBlocks * m
//arrays declared volatile mean entries are not cached,
//so that a threadfence_*() is sufficient to keep all values coherent
//(meaning all writes before the fence are reflected in all reads after)
__global__ void mmqrKernel(volatile Scalar* mat, volatile Scalar* matScratch, Scalar* tau, volatile Scalar* W, volatile Scalar* z, volatile Scalar* AcolGlobal, int m, int n)
{
  //All dynamic shared memory goes here
  //Is a flat 48 KiB buffer, free for use by each block
  extern __shared__ char sharedBuf[];
  //all blocks need an mx1 vector for scratch space (call it Acol)
  //the proper amount of global memory (for all blocks) is already allocated in AcolGlobal
  Scalar* Acol = AcolGlobal + gridDim.x * m * sizeof(Scalar);
  //determine range of rows "owned" by this thread
  for(int pc = 0; pc < n; pc += PC)
  {
    //useful to think of shared memory buffer as a stack
    //for simple dynamic allocations
    char* sharedStack = sharedBuf;
    Scalar* panel = &mat[pc * m];
    //Kernel uses 2D grid (to match the way trailing matrix is partitioned among blocks)
    if(blockIdx.x == 0 && blockIdx.y == 0)
    {
      //"allocate" some shared space for panelTau
      Scalar* panelTau = (Scalar*) sharedStack;
      sharedStack += PC * sizeof(Scalar);
      //zero out W
      for(int i = 0; i < m * n; i += blockDim.x)
      {
        int index = i + threadIdx.x;
        if(index < m * n)
          W[i] = 0;
      }
      for(int col = 0; col < PC; col++)
      {
        //(middle panels are both top and bottom)
        int vstart = col + pc;
        int vend = m;
        int vlen = vend - vstart;
        //note: when computing tau/reflectors,
        //work directly with global mat (only 2 flops per element anyway)
        //compute the inner product and norm of column
        Scalar innerProd = 0;
        {
          Scalar localInnerProd = 0;
          //use a cyclic row distribution for perfect coalesced accesses
          for(int i = vstart; i < vend; i += blockDim.x)
          {
            if(i + threadIdx.x < vend)
            {
              localInnerProd += panel[i + col * m] * panel[i + col * m];
            }
          }
          //now, sum up the localInnerProds across the whole block
          //write the partial sums to shared, then do a simple linear reduction
          Scalar* toReduce = (Scalar*) sharedBuf;
          toReduce[threadIdx.x] = localInnerProd;
          __syncthreads();
          Scalar* finalReduce = ((Scalar*) sharedBuf) + blockDim.x;
          int numFinal = blockDim.x / 16;
          if(blockDim.x & 0xF)
            numFinal++;
          if(threadIdx.x < numFinal)
          {
            localInnerProd = 0;
            for(int i = 0; i < 16; i++)
            {
              int index = i + threadIdx.x * 16;
              if(index < blockDim.x)
                localInnerProd += toReduce[index];
            }
            finalReduce[threadIdx.x] = localInnerProd;
          }
          __syncthreads();
          //now, every thread sums up finalReduce to get innerProd
          for(int i = 0; i < numFinal; i++)
            innerProd += finalReduce[i];
        }
        Scalar norm = sqrt(innerProd);
        Scalar leading = panel[col * m + vstart];
        Scalar sign = (leading < 0) ? -1.0 : 1.0;
        Scalar u = leading + sign * norm;
        Scalar thisTau = sign * u / norm;
        //compute entire w vector in-place
        for(int i = vstart; i < vend; i += blockDim.x)
        {
          int index = i + threadIdx.x;
          if(index == vstart)
          {
            panelTau[col] = thisTau;
            panel[col * m + vstart] = -sign * norm;
          }
          else if(index < vend)
          {
            panel[col * m + index] /= u;
          }
        }
        //v is now fully computed and stored back to panel
        //compute z vector using W,Y
        //each thread will compute one entry in z
        for(int i = 0; i < m; i += blockDim.x)
        {
          int index = i + threadIdx.x;
          if(index < m)
          {
            Scalar zval = 0;
            if(index >= vstart)
              zval = -panelTau[col] * panel[col * m + index];
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
                if(j > k)
                  yval = panel[k * m + j];
                else if(j == k)
                  yval = 1;
                wyt += W[k * m + i] * yval;
              }
              wytvi += wyt * panel[col * m + j];
            }
            zval -= panelTau[col] * wytvi;
            z[index] = zval;
          }
        }
        __syncthreads();  //make z coherent across threads
        //z is the next column of W
        for(int i = 0; i < m; i += blockDim.x)
        {
          int index = i + threadIdx.x;
          if(index < m)
            W[col * m + index] = z[index];
        }
        __syncthreads();  //make W coherent
        //apply reflector in col to remaining columns in panel
        for(int applyCol = col + 1; applyCol < PC; applyCol++)
        {
          //Create a copy of the updating column of A which will
          //persist while each entry is computed
          //Only the height range [vstart, vend) is read, used and written back
          for(int i = vstart; i < m; i += blockDim.x)
          {
            int index = i + threadIdx.x;
            if(index < m)
              Acol[index] = panel[applyCol * m + index];
          }
          __threadfence();
          for(int applyRow = vstart; applyRow < m; applyRow += blockDim.x)
          {
            int index = applyRow + threadIdx.x;
            if(index < m)
            {
              Scalar val = Acol[index];
              Scalar vIndex;
              if(index == col)
                vIndex = 1;
              else
                vIndex = panel[col * m + index];
              for(int i = vstart; i < m; i++)
              {
                Scalar vi;
                if(i == col)
                  vi = 1;
                else
                  vi = panel[col * m + i];
                val -= panelTau[col] * vIndex * vi * Acol[i];
              }
              panel[applyCol * m + index] = val;
            }
          }
        }
      }
      if(threadIdx.x < PC)
      {
        //finalize global tau values for this panel
        tau[pc + threadIdx.x] = panelTau[threadIdx.x];
      }
    }
    //this is necessary to guarantee that Block 0's updates to mat/W are visible to all other blocks
    __threadfence_system();
    sharedStack = sharedBuf;
    //Allocate some shared arrays that all blocks will use for computations
    Scalar* Wblock = (Scalar*) sharedStack;
    sharedStack += PR * PC * sizeof(Scalar);
    Scalar* Yblock = (Scalar*) sharedStack;
    sharedStack += PR * PC * sizeof(Scalar);
    Scalar* Ablock = (Scalar*) sharedStack;
    sharedStack += PR * PR * sizeof(Scalar);
    //update trailing columns of A: A = (I + YW^T)A
    //Each block in the 2D grid is responsible for updating one square region of A (starting at upper-left corner of trailing part)
    //Each block will read into Wblock/Yblock/Ablock once and write results out to Ascratch once
    int blockRow = blockIdx.x * (m / PR);
    int blockCol = pc + PC + blockIdx.y * (n / PR);
    for(int applyCol = pc + PC; applyCol < n; applyCol++)
    {
      Scalar* newAcol = (Scalar*) malloc(m * sizeof(Scalar));
      //gives perfect minimal memory bandwidth:
      //each entry read/written once in optimally coalesced accesses
      //the IA term above is implicit (other term added to this one)
      for(int i = 0; i < m; i++)
      {
        Acol[i] = mat[i + applyCol * m];
        newAcol[i] = Acol[i];
      }
      //now compute YW^T * A[<panel rows>, applyCol] and update newAcol
      for(int i = 0; i < m; i++)
      {
        Scalar newAval = 0;
        for(int j = 0; j < m; j++)
        {
          //need inner product of row i of Y and row j of W
          //generate entry (Y*W^T)(i, j)
          Scalar ywt = 0;
          for(int k = 0; k < PC; k++)
          {
            //yval = Y(i, k) is an element of v reflectors
            Scalar yval = 0;
            if(i > k)
              yval = panel[k * m + j];
            else if(i == k)
              yval = 1;
            ywt += yval * W[k * m + j];
          }
          //multiply that by entry j of A
          newAval += ywt * Acol[j];
        }
        newAcol[i] += newAval;
      }
      //write back newAcol
      for(int i = 0; i < m; i++)
      {
        mat[i + applyCol * m] = newAcol[i];
      }
      free(Acol);
      free(newAcol);
    }
    //wait for all writes of all blocks (to trailing entries of A) to complete
    //this kind of fence is expensive but is only way to synchronize across blocks
    __threadfence_system();
  }
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
  //first, R is simpy the diagonal and upper triangular parts of A
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
  //next, Q is the result of applying each Householder reflector to I(m)
  //note: this is very expensive to do naively on host
  //first, get I(m) into Q
  identity(Q, m);
  for(int i = 0; i < n; i++)
  {
    Scalar* v = (Scalar*) malloc(m * sizeof(Scalar));
    for(int j = 0; j < i; j++)
    {
      v[j] = 0;
    }
    v[i] = 1;
    for(int j = i + 1; j < m; j++)
    {
      v[j] = A[i * m + j];
    }
    Scalar* H = (Scalar*) malloc(m * m * sizeof(Scalar));
    identity(H, m);
    //j is column of H being updated
    for(int j = 0; j < m; j++)
    {
      //k is row
      for(int k = 0; k < m; k++)
      {
        H[k + j * m] -= tau[i] * v[k] * v[j];
      }
    }
    //dgemm can't multiply Q by H in-place,
    //so make a persistent copy of Q
    Scalar* prevQ = (Scalar*) malloc(m * m * sizeof(Scalar));
    for(int j = 0; j < m * m; j++)
      prevQ[j] = Q[j];
    dgemm(prevQ, H, Q, m, m, m);
    free(prevQ);
    free(H);
    free(v);
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

#define HANDLE_ERROR(x) \
{\
  cudaError_t err = x; \
  if(x) {printf("CUDA error on line %i: %d\n", __LINE__, x); exit(1);} \
}

//Host wrapper for the main CUDA kernel
//No extra overhead since copies to/from device would be necessary anyway
void mmqr(Scalar* mat, Scalar* tau, int m, int n)
{
  Scalar* Adev;
  HANDLE_ERROR(cudaMalloc((void**) &Adev, m * n * sizeof(Scalar)));
  Scalar* tauDev;
  HANDLE_ERROR(cudaMalloc((void**) &tauDev, n * sizeof(Scalar)));
  HANDLE_ERROR(cudaMemcpy(Adev, mat, m * n * sizeof(Scalar), cudaMemcpyHostToDevice));
  //launch the kernel
  //
  //only use one block and fixed threads for main kernel,
  //but main kernel will itself launch several blocks to saturate FLOPs during trailing updates)
  //
  //want to use every SM in order to use all shared memory in device
  //figure out how many SMs there are
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int sm = prop.multiProcessorCount;
  int shmem = prop.sharedMemPerBlock;
  int maxThreads = prop.maxThreadsPerBlock;
  //MUST have at least 48 KiB of shared memory for this to work in its current state
  //TODO: adapt to any amount of shared
  if(prop.sharedMemPerBlock < 48 * 1024)
  {
    puts("CUDA device has < 48 KiB shared memory per block!");
    exit(1);
  }
  //want one block per SM and as many threads as possible (up to 1 per row of A)
  int threadsPerBlock = m < maxThreads ? m : maxThreads;
  //call kernel with one block and many threads
  //this kernel will launch many blocks to do trailing update (asynchronously)
  mmqrKernel<<<1, threadsPerBlock, shmem, 0>>>(Adev, tauDev, m, n, sm);
  //retrieve A and tau
  HANDLE_ERROR(cudaMemcpy(mat, Adev, m * n * sizeof(Scalar), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(tau, tauDev, n * sizeof(Scalar), cudaMemcpyDeviceToHost));
  cudaFree(tauDev);
  cudaFree(Adev);
}

int main()
{
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
  int m = PC * 2;
  int n = PC;
  assert(m >= n);
  Scalar* A = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar* RV = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar* tau = (Scalar*) malloc(n * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = (Scalar) rand() / RAND_MAX;
    RV[i] = A[i];
  }
  //printMat(A, m, n);
  int trials = 20;
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
  //printf("A raw storage after QR:\n");
  //printMat(RV, m, n);
  Scalar* Q = (Scalar*) malloc(m * m * sizeof(Scalar));
  Scalar* R = (Scalar*) malloc(m * n * sizeof(Scalar));
  explicitQR(RV, tau, Q, R, m, n);
  //printf("Q:\n");
  //printMat(Q, m, m);
  //printf("R:\n");
  //printMat(R, m, n);
  //now compute Q*R explicitly and compare to A
  Scalar* QR = (Scalar*) malloc(m * n * sizeof(Scalar));
  dgemm(Q, R, QR, m, m, n);
  //printf("QR:\n");
  //printMat(QR, m, n);
  //printf("QR-A (should be 0):\n");
  Scalar* QRmA = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar errNorm = 0;
  for(int i = 0; i < m * n; i++)
  {
    QRmA[i] = QR[i] - A[i];
    errNorm += QRmA[i] * QRmA[i];
  }
  //printMat(QRmA, m, n);
  free(QRmA);
  errNorm = sqrt(errNorm);
  printf("L2 norm of residual QR-A: %.9g\n", errNorm);
  free(QR);
  free(RV);
  free(R);
  free(Q);
  free(A);
  return 0;
}

