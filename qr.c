#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

//Scalar type and panel size (RxC)
//Scalar may be either float or double
//(2RC + C) * sizeof(Scalar) must fit in 48 KiB
#define Scalar float
#define PR 16
#define PC 4

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
      printf("%f ", mat[j * m + i]);
    }
    putchar('\n');
  }
  putchar('\n');
}

//mat should be column-major
void mmqr(Scalar* mat, Scalar* tau, int m, int n)
{
  //iterate over all subdiagonal panels
  //first left to right
  for(int pc = 0; pc < n; pc += PC)
  {
    //then bottom to top, sliding panel up by R-C each iteration
    for(int pr = m - PR; pr >= pc; pr -= (PR-PC))
    {
      printf("Processing panel at col %d, row %d\n", pc, pr);
      //load panel into shared memory, one column at a time
      //Note that panel is column major
      Scalar panel[PC][PR];
      Scalar panelTau[PC];
      for(int i = 0; i < PC; i++)
      {
        panelTau[i] = 0;
      }
      for(int col = 0; col < PC; col++)
      {
        for(int row = 0; row < PR; row++)
        {
          panel[col][row] = mat[row + col * m];
        }
      }
      printf("Initial panel values:\n");
      printMat((Scalar*) &panel[0][0], PR, PC);
      //see Kerr/Campbell/Richards paper for blocked Householder description
      //
      //The W matrix (for applying whole panel of HH reflectors at once)
      //should be in shared and is updated as each reflector is determined
      //
      //TODO: columns of Y matrix are just the reflectors, so an explicit
      //copy of it is unnecessary
      Scalar W[PC][PR];
      Scalar Y[PC][PR];
      for(int i = 0; i < PC; i++)
      {
        for(int j = 0; j < PR; j++)
        {
          W[i][j] = 0;
          Y[i][j] = 0;
        }
      }
      //update each trailing column (pr:pr+R, pc+C:N):
      //for each column, compute HH reflectors
      for(int col = 0; col < PC; col++)
      {
        Scalar innerProd = 0;
        printf("Computing norm of subdiagonal A entries (col %d)\n", col);
        for(int row = col; row < PR; row++)
        {
          innerProd += panel[col][row] * panel[col][row];
        }
        printf("inner prod is %f\n", innerProd);
        Scalar norm = sqrt(innerProd);
        Scalar sign = panel[col][col] < 0 ? -1 : 1;
        Scalar u = panel[col][col] + sign * norm;
        panel[col][col] = -sign * norm;
        //is the panel at the bottom of A?
        bool bottomPanel = pr == m - PR;
        //does col 0 of panel cross A's diagonal?
        bool topPanel = pr <= pc;
        if(bottomPanel)
          puts("Panel hits bottom of matrix");
        else
          puts("Panel does NOT hit bottom of matrix");
        if(topPanel)
          puts("Panel hits top of matrix");
        else
          puts("Panel does NOT hit top of matrix");
        //(middle panels are both top and bottom)
        int vstart;
        int vend;
        if(bottomPanel)
        {
          vstart = col;
          vend = PR;
        }
        else if(topPanel && !bottomPanel)
        {
          //vstart needs to be at or below A's diagonal, even if
          //panel boundaries extends above it
          vstart = pc - pr + col;
          vend = PR - col - 1;
        }
        else
        {
          //neither top nor bottom panel
          vstart = col;
          vend = PR - col - 1;
        }
        printf("in col %d of panel, v in rows [%d, %d)\n", col, vstart, vend);
        int vlen = vend - vstart;
        Scalar* v = malloc(vlen * sizeof(Scalar));
        for(int i = 0; i < vlen; i++)
        {
          v[i] = 0;
        }
        //compute entire w explicitly,
        //then write back nontrivial entries to the panel
        v[0] = 1;
        for(int i = vstart + 1; i < vend; i++)
        {
          panel[col][i] /= u;
          v[i - vstart] = panel[col][i];
        }
        //v is now fully computed (explicitly)
        printf("v (%d elems):\n", vlen);
        for(int i = 0; i < vlen; i++)
        {
          printf("%f\n", v[i]);
        }
        putchar('\n');
        //update W matrix
        //beta is from Algorithm 3 of Kerr
        /*
        Scalar beta = 2 / innerProd;
        //compute z = -Bv - WY^T * v
        Scalar z[R];
        for(int i = 0; i < R; i++)
        {
          z[i] = 0;
          if(i >= vstart && i < vend)
            z[i] = -beta * v[i - vstart];
        }
        for(int i = 0; i < R; i++)
        {
          //finish computing entry i of z
          //compute zval as entry i of W * Y^T * v
          Scalar zval = 0;
          for(int j = 0; j < R; j++)
          {
            //need inner product of row i of W and column j of Y
            //use the fact that only the first col+1 columns of W and Y are nonzero
            if(j >= vstart && j < vend)
            {
              Scalar wyt = 0;
              for(int k = 0; k < col + 1; k++)
              {
                wyt += W[k][i] * Y[j][k];
              }
              zval += wyt * v[j - vstart];
            }
          }
          z[i] -= beta * zval;
        }
        //z is the next column of W
        for(int i = 0; i < R; i++)
        {
          W[col][i] = z[i];
        }
        //v is the next column of Y
        //note that Y is zeroed out initially, so only need to copy nonzeros
        for(int i = vstart; i < vend; i++)
        {
          Y[col][i] = v[i - vstart];
        }
        */
        panelTau[col] = sign * u / norm;
        //apply reflector in col to remaining columns in panel
        for(int applyCol = col + 1; applyCol < PC; applyCol++)
        {
          //Create a copy of the updating column of A which can
          //persist while each entry is computed
          Scalar* Acol = malloc(vlen * sizeof(Scalar));
          for(int i = 0; i < vlen; i++)
          {
            Acol[i] = panel[applyCol][i + vstart];
          }
          for(int applyRow = vstart; applyRow < vend; applyRow++)
          {
            int vindex = applyRow - vstart;
            Scalar val = Acol[applyRow];
            for(int i = 0; i < vlen; i++)
            {
              val -= panelTau[applyCol] * v[vindex] * v[i] * Acol[applyRow];
            }
            panel[applyCol][applyRow] = val;
          }
          free(Acol);
        }
        free(v);
        printf("Panel after processing column %d:\n", col);
        printMat((Scalar*) &panel[0][0], PR, PC);
      }
      //panel, panelTau, W and Y are all fully computed
      //write back panel to A
      for(int col = 0; col < PC; col++)
      {
        for(int row = 0; row < PR; row++)
        {
          mat[row + col * m] = panel[col][row];
        }
      }
      //update trailing columns of A: A = (I + YW^T)A
      //the new columns of A can be updated independently (and in parallel)
      //so this loop could be a kernel launch with some A columns given to each SM
      /*
      for(int applyCol = pc + PC; applyCol < n; applyCol++)
      {
        //The new column, to be copied back into A
        Scalar Acol[PR];     //these vectors both go in shared
        Scalar newAcol[PR];
        //gives perfect minimal memory bandwidth:
        //each entry read/written once in optimally coalesced accesses
        //the IA term above is implicit (other term added to this one)
        for(int i = 0; i < PR; i++)
        {
          Acol[i] = mat[pr + i + applyCol * m];
          newAcol[i] = Acol[i];
        }
        //now compute YW^T * A[<panel rows>, applyCol] and update newAcol
        for(int i = 0; i < PR; i++)
        {
          Scalar newAval = 0;
          for(int j = 0; j < PR; j++)
          {
            //need inner product of row i of Y and row j of W
            Scalar ywt = 0;
            for(int k = 0; k < PC; k++)
            {
              ywt += Y[i][k] * W[j][k];
            }
            //multiply that by entry j of A
            newAval += ywt * Acol[j];
          }
          newAcol[i] += newAval;
        }
        //write back newAcol
        for(int i = 0; i < PR; i++)
        {
          mat[pr + i + applyCol * m] = newAcol[i];
        }
      }
      */
      for(int i = 0; i < PC; i++)
      {
        printf("updating tau[%d] value = %f\n", i + pc, panelTau[i]);
        tau[i + pc] = panelTau[i];
      }
    }
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
    Scalar* v = malloc(n * sizeof(Scalar));
    for(int j = 0; j < i; j++)
    {
      v[j] = 0;
    }
    v[i] = 1;
    for(int j = i + 1; j < n; j++)
    {
      v[j] = A[i * m + j];
    }
    Scalar* H = malloc(m * m * sizeof(Scalar));
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
    Scalar* prevQ = malloc(m * m * sizeof(Scalar));
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

int main()
{
  int m = PR;
  int n = PC;
  assert(m >= n);
  Scalar* A = malloc(m * n * sizeof(Scalar));
  Scalar* RV = malloc(m * n * sizeof(Scalar));
  Scalar* tau = malloc(n * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = (Scalar) rand() / RAND_MAX;
    RV[i] = A[i];
  }
  printMat(A, m, n);
  mmqr(RV, tau, m, n);
  printf("A raw storage after QR:\n");
  printMat(RV, m, n);
  printf("tau values after QR:\n");
  for(int i = 0; i < n; i++)
  {
    printf("%f ", tau[i]);
  }
  putchar('\n');
  Scalar* Q = malloc(m * m * sizeof(Scalar));
  Scalar* R = malloc(m * n * sizeof(Scalar));
  explicitQR(RV, tau, Q, R, m, n);
  printf("Q matrix:\n");
  printMat(Q, m, m);
  printf("R matrix:\n");
  printMat(R, m, n);
  //now compute Q*R explicitly and compare to A
  Scalar* QR = malloc(m * n * sizeof(Scalar));
  dgemm(Q, R, QR, m, m, n);
  printf("QR (should match A):\n");
  printMat(QR, m, n);
  printf("QR-A (should be 0):\n");
  Scalar* QRmA = malloc(m * n * sizeof(Scalar));
  for(int i = 0; i < m * n; i++)
    QRmA[i] = QR[i] - A[i];
  printMat(QRmA, m, n);
  free(QRmA);
  free(QR);
  free(RV);
  free(R);
  free(Q);
  free(A);
  free(tau);
  return 0;
}

