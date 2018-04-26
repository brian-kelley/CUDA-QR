#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/time.h>

//Scalar type and panel width
#define Scalar float
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

//mat should be column-major
void mmqrKernel(Scalar* mat, Scalar* tau, int m, int n)
{
  //iterate over all subdiagonal panels
  //first left to right
  for(int pc = 0; pc < n; pc += PC)
  {
    //load panel into shared memory, one column at a time
    //Note that panel is column major
    Scalar* panel = (Scalar*) malloc(PC * m * sizeof(Scalar));
    Scalar panelTau[PC];
    for(int col = 0; col < PC; col++)
    {
      for(int row = 0; row < m; row++)
      {
        panel[col * m + row] = mat[row + (col + pc) * m];
      }
    }
    //see Kerr/Campbell/Richards paper for blocked Householder description
    //
    //The W matrix (for applying whole panel of HH reflectors at once)
    //should be in shared and is updated as each reflector is determined
    //
    //TODO: columns of Y matrix are just the reflectors, so an explicit
    //copy of it is unnecessary (in final version, read it from panel subdiagonal)
    Scalar* W = (Scalar*) malloc(PC * m * sizeof(Scalar));
    Scalar* Y = (Scalar*) malloc(PC * m * sizeof(Scalar));
    for(int i = 0; i < PC; i++)
    {
      for(int j = 0; j < m; j++)
      {
        W[i * m + j] = 0;
        Y[i * m + j] = 0;
      }
    }
    //for each column, compute HH reflectors
    for(int col = 0; col < PC; col++)
    {
      //(middle panels are both top and bottom)
      int vstart = col + pc;
      int vend = m;
      int vlen = vend - vstart;
      Scalar innerProd = 0;
      for(int row = vstart; row < vend; row++)
      {
        innerProd += panel[col * m + row] * panel[col * m + row];
      }
      Scalar norm = sqrt(innerProd);
      Scalar sign = (panel[col * m + vstart] < 0) ? -1.0 : 1.0;
      Scalar u = panel[col * m + vstart] + sign * norm;
      Scalar thisTau = sign * u / norm;
      panelTau[col] = thisTau;
      panel[col * m + vstart] = -sign * norm;
      Scalar* v = (Scalar*) malloc(vlen * sizeof(Scalar));
      //compute entire w explicitly,
      //and write back nontrivial entries to the panel
      v[0] = 1;
      for(int i = vstart + 1; i < vend; i++)
      {
        panel[col * m + i] /= u;
        v[i - vstart] = panel[col * m + i];
      }
      //v is now fully computed (explicitly)
      //update W matrix
      Scalar* z = (Scalar*) malloc(m * sizeof(Scalar));
      for(int i = 0; i < m; i++)
      {
        if(i >= vstart && i < vend)
          z[i] = -panelTau[col] * v[i - vstart];
        else
          z[i] = 0;
      }
      if(col > 0)
      {
        for(int i = 0; i < m; i++)
        {
          //finish computing entry i of z
          //compute zval as (W * Y^T * v)(i)
          Scalar wytvi = 0;
          for(int j = 0; j < m; j++)
          {
            //need inner product of row i of W and row j of Y
            //this is (WY^T)(i, j)
            //use the fact that only the first col+1 columns of W and Y are nonzero
            if(j >= vstart && j < vend)
            {
              Scalar wyt = 0;
              for(int k = 0; k < col; k++)
              {
                wyt += W[k * m + i] * Y[k * m + j];
              }
              wytvi += wyt * v[j - vstart];
            }
          }
          z[i] -= panelTau[col] * wytvi;
        }
      }
      //z is the next column of W
      for(int i = 0; i < m; i++)
      {
        W[col * m + i] = z[i];
      }
      free(z);
      //v is the next column of Y
      //note that Y is zeroed out initially, so only need to copy nonzeros
      for(int i = 0; i < vlen; i++)
      {
        Y[col * m + i + vstart] = v[i];
      }
      //apply reflector in col to remaining columns in panel
      //TODO: do in parallel
      for(int applyCol = col + 1; applyCol < PC; applyCol++)
      {
        //Create a copy of the updating column of A which can
        //persist while each entry is computed
        Scalar* Acol = (Scalar*) malloc(vlen * sizeof(Scalar));
        for(int i = 0; i < vlen; i++)
        {
          Acol[i] = panel[applyCol * m + vstart + i];
        }
        for(int applyRow = vstart; applyRow < vend; applyRow++)
        {
          int vindex = applyRow - vstart;
          Scalar val = Acol[vindex];
          for(int i = 0; i < vlen; i++)
          {
            val -= panelTau[col] * v[vindex] * v[i] * Acol[i];
          }
          panel[applyCol * m + applyRow] = val;
        }
        free(Acol);
      }
      free(v);
    }
    puts("W MATRIX:\n");
    printMat(W, m, PC);
    putchar('\n');
    //panel, panelTau, W and Y are all fully computed
    //write back panel to A
    for(int col = 0; col < PC; col++)
    {
      for(int row = 0; row < m; row++)
      {
        mat[row + (col + pc) * m] = panel[col * m + row];
      }
    }
    //update trailing columns of A: A = (I + YW^T)A
    //all columns of A can be updated in parallel
    //so this loop can be a kernel launch with a few A columns in each block
    for(int applyCol = pc + PC; applyCol < n; applyCol++)
    {
      //The new column, to be copied back into A
      Scalar* Acol = (Scalar*) malloc(m * sizeof(Scalar));     //these vectors both go in shared
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
            ywt += Y[k * m + i] * W[k * m + j];
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
    for(int i = 0; i < PC; i++)
    {
      tau[i + pc] = panelTau[i];
    }
    free(Y);
    free(W);
    free(panel);
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

//Host wrapper for the main CUDA kernel
//No extra overhead since copies to/from device would be necessary anyway
void mmqr(Scalar* mat, Scalar* tau, int m, int n)
{
  Scalar* Adev = malloc(m * n * sizeof(Scalar));
  Scalar* tauDev = malloc(n * sizeof(Scalar));
  //launch the kernel
  //
  //only use one block and fixed threads for main kernel,
  //but main kernel will itself launch several blocks to saturate FLOPs during trailing updates)
  //
  //want to use every SM in order to use all shared memory in device
  //figure out how many SMs there are
  memcpy(Adev, mat, m * n * sizeof(Scalar));
  mmqrKernel(Adev, tauDev, m, n);
  memcpy(mat, Adev, m * n * sizeof(Scalar));
  memcpy(tau, tauDev, n * sizeof(Scalar));
  //retrieve A and tau
  free(Adev);
  free(tauDev);
}

int main()
{
  //only use one device (at least, for now)
  //First, make sure device is using proper 48 KB of shared, 16 KB L1
  //during all calls to L1 kernel
  //Note that this is not the default
  int m = PC;
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
  puts("A matrix:\n");
  printMat(A, m, n);
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
    //refresh RV (this isn't part of the algorithm and so isn't timed)
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

