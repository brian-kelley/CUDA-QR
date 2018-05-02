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
#define PR 8
#define PC 2

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
//
//panelsX is how many panels are needed to tile mat horizontally
//panelsY is how many panels are needed vertically, taking into
//account overlap
//
//tau is allocated to be panelsX * n, col-major (tau values of one panel contiguous)
//technically tau array is lower triangular so this could be compressed, but don't for simplicity

//integer division a/b, rounded up
#define ceildiv(a, b) ((a) / (b) + ((a) % (b) != 0))

void getPanelDims(int m, int n, int* rowPanels, int* colPanels)
{
  *colPanels = ceildiv(n, PC);
  *rowPanels = 1;
  if(m > PR)
    *rowPanels += ceildiv(m - PR, PR - PC);
}

void mmqr(Scalar* mat, Scalar** tau, int m, int n)
{
  printf("Computing QR factorization of %d by %d matrix...\n\n", m, n);
  int rowPanels, colPanels;
  getPanelDims(m, n, &rowPanels, &colPanels);
  //allocate space for one tau value per col in each panel
  *tau = malloc(rowPanels * colPanels * PC * sizeof(Scalar));
  memset(*tau, 0, rowPanels * colPanels * PC * sizeof(Scalar));
  printf("Matrix is %d by %d panels (cols, rows).\n", colPanels, rowPanels);
  //iterate over all subdiagonal panels
  //first left to right
  //pcCount gives col index of panel (for tau)
  int pcCount = 0;
  for(int pc = 0; pc < n; pc += PC)
  {
    //then bottom to top, sliding panel up by R-C each iteration
    //prCount gives row index of panel
    int prCount = 0;
    for(int pr = m - PR; (pr + PR > pc) && pr >= 0; pr -= (PR-PC))
    {
      printf("Processing panel at (%d, %d)\n", pr, pc);
      //load panel into shared memory, one column at a time
      //Note that panel is column major
      Scalar (*panel)[PR] = malloc(PC * PR * sizeof(Scalar));
      Scalar* panelTau = malloc(PC * sizeof(Scalar));
      //TODO: save bandwidth by reusing part of panel which is already in shared
      //(just shift the top PCxPC entries down to the bottom)
      for(int col = 0; col < PC; col++)
      {
        for(int row = 0; row < PR; row++)
        {
          panel[col][row] = mat[(row + pr) + (col + pc) * m];
        }
      }
      //see Kerr/Campbell/Richards paper for blocked Householder description
      //
      //The W matrix (for applying whole panel of HH reflectors at once)
      //should be in shared and is updated as each reflector is determined
      //
      //TODO: columns of Y matrix are just the reflectors, so an explicit
      //copy of it is unnecessary (in final version, read it from panel subdiagonal)
      Scalar (*W)[PR] = malloc(PC * PR * sizeof(Scalar));
      Scalar (*Y)[PR] = malloc(PC * PR * sizeof(Scalar));
      for(int i = 0; i < PC; i++)
      {
        for(int j = 0; j < PR; j++)
        {
          W[i][j] = 0;
          Y[i][j] = 0;
        }
      }
      //is the panel at the bottom of A?
      bool bottomPanel = pr == m - PR;
      //does col 0 of panel cross A's diagonal?
      bool topPanel = pr <= pc;
      //update each trailing column (pr:pr+R, pc+C:N):
      //for each column, compute HH reflectors
      for(int col = 0; col < PC; col++)
      {
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
        printf("Col %d reflector begins at row %d and ends at %d\n", col, vstart, vend);
        int vlen = vend - vstart;
        Scalar innerProd = 0;
        for(int row = vstart; row < vend; row++)
        {
          innerProd += panel[col][row] * panel[col][row];
        }
        Scalar norm = sqrt(innerProd);
        Scalar sign = (panel[col][vstart] < 0) ? -1.0 : 1.0;
        Scalar u = panel[col][vstart] + sign * norm;
        Scalar thisTau = sign * u / norm;
        printf("\n\n\n\n\nNORM: %f\n", norm);
        printf("Leading entry (%d, %d): %f\n", vstart, col, panel[col][vstart]);
        printf("tau: %f\n", thisTau);
        panelTau[col] = thisTau;
        panel[col][vstart] = -sign * norm;
        Scalar* v = malloc(vlen * sizeof(Scalar));
        //compute entire w explicitly,
        //and write back nontrivial entries to the panel
        v[0] = 1;
        for(int i = vstart + 1; i < vend; i++)
        {
          panel[col][i] /= u;
          v[i - vstart] = panel[col][i];
        }
        //v is now fully computed (explicitly)
        //update W matrix
        Scalar z[PR];
        for(int i = 0; i < PR; i++)
        {
          if(i >= vstart && i < vend)
            z[i] = -panelTau[col] * v[i - vstart];
          else
            z[i] = 0;
        }
        if(col > 0)
        {
          for(int i = 0; i < PR; i++)
          {
            //finish computing entry i of z
            //compute zval as (W * Y^T * v)(i)
            Scalar wytvi = 0;
            for(int j = 0; j < PR; j++)
            {
              //need inner product of row i of W and row j of Y
              //this is (WY^T)(i, j)
              //use the fact that only the first col+1 columns of W and Y are nonzero
              if(j >= vstart && j < vend)
              {
                Scalar wyt = 0;
                for(int k = 0; k < col; k++)
                {
                  wyt += W[k][i] * Y[k][j];
                }
                wytvi += wyt * v[j - vstart];
              }
            }
            z[i] -= panelTau[col] * wytvi;
          }
        }
        //z is the next column of W
        for(int i = 0; i < PR; i++)
        {
          W[col][i] = z[i];
        }
        //v is the next column of Y
        //note that Y is zeroed out initially, so only need to copy nonzeros
        for(int i = 0; i < vlen; i++)
        {
          Y[col][i + vstart] = v[i];
        }
        //apply reflector in col to remaining columns in panel
        for(int applyCol = col + 1; applyCol < PC; applyCol++)
        {
          //Create a copy of the updating column of A which can
          //persist while each entry is computed
          Scalar* Acol = malloc(vlen * sizeof(Scalar));
          for(int i = 0; i < vlen; i++)
          {
            Acol[i] = panel[applyCol][vstart + i];
          }
          for(int applyRow = vstart; applyRow < vend; applyRow++)
          {
            int vindex = applyRow - vstart;
            Scalar val = Acol[vindex];
            for(int i = 0; i < vlen; i++)
            {
              val -= panelTau[col] * v[vindex] * v[i] * Acol[i];
            }
            panel[applyCol][applyRow] = val;
          }
          free(Acol);
        }
        free(v);
      }
      //panel, panelTau, W and Y are all fully computed
      //write back panel to A
      for(int col = 0; col < PC; col++)
      {
        for(int row = 0; row < PR; row++)
        {
          mat[(row + pr) + (col + pc) * m] = panel[col][row];
        }
      }
      puts("\nABOUT TO UPDATE TRAILING COLUMNS...\n");
      puts("W matrix:");
      printMat((Scalar*) W, PR, PC);
      //update trailing columns of A: A = (I + YW^T)A
      //all columns of A can be updated in parallel
      //so this loop can be a kernel launch with a few A columns in each block
      for(int applyCol = pc + PC; applyCol < n; applyCol++)
      {
        //The new column, to be copied back into A
        Scalar* Acol = malloc(PR * sizeof(Scalar));     //these vectors both go in shared
        Scalar* newAcol = malloc(PR * sizeof(Scalar));
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
            //generate entry (Y*W^T)(i, j)
            Scalar ywt = 0;
            for(int k = 0; k < PC; k++)
            {
              ywt += Y[k][i] * W[k][j];
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
        free(Acol);
        free(newAcol);
      }
      if(pc == 0)
      {
        puts("Trailing matrix after update:\n");
        printMat(&mat[2 * m], m, 2);
        putchar('\n');
      }
      for(int i = 0; i < PC; i++)
      {
        (*tau)[(rowPanels * pcCount + prCount) * PC + i] = panelTau[i];
        printf("tau(%d) in panel %d, %d is %f\n", i, pr, pc, panelTau[i]);
      }
      free(Y);
      free(W);
      free(panelTau);
      free(panel);
      prCount++;
    }
    pcCount++;
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
        Scalar* v = malloc(m * sizeof(Scalar));
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
        /*
        printf("v:\n");
        for(int i = 0; i < m; i++)
        {
          printf("%f ", v[i]);
        }
        putchar('\n');
        */
        //create H matrix for this reflector
        Scalar* H = malloc(m * m * sizeof(Scalar));
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
        Scalar* prevQ = malloc(m * m * sizeof(Scalar));
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
  Scalar* A = malloc(m * n * sizeof(Scalar));
  Scalar* RV = malloc(m * n * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = (Scalar) rand() / RAND_MAX;
    RV[i] = A[i];
  }
  printMat(A, m, n);
  Scalar* tau = NULL;
  mmqr(RV, &tau, m, n);
  printf("A raw storage after QR:\n");
  printMat(RV, m, n);
  int rowPanels, colPanels;
  getPanelDims(m, n, &rowPanels, &colPanels);
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
  Scalar* Q = malloc(m * m * sizeof(Scalar));
  Scalar* R = malloc(m * n * sizeof(Scalar));
  explicitQR(RV, tau, Q, R, m, n);
  printf("Q:\n");
  printMat(Q, m, m);
  printf("R:\n");
  printMat(R, m, n);
  //now compute Q*R explicitly and compare to A
  Scalar* QR = malloc(m * n * sizeof(Scalar));
  dgemm(Q, R, QR, m, m, n);
  printf("QR:\n");
  printMat(QR, m, n);
  printf("QR-A (should be 0):\n");
  Scalar* QRmA = malloc(m * n * sizeof(Scalar));
  Scalar errNorm = 0;
  for(int i = 0; i < m * n; i++)
  {
    QRmA[i] = QR[i] - A[i];
    errNorm += QRmA[i] * QRmA[i];
  }
  printMat(QRmA, m, n);
  free(QRmA);
  errNorm = sqrt(errNorm);
  printf("L2 norm of residual QR-A: %.9g\n", errNorm);
  free(QR);
  free(RV);
  free(R);
  free(Q);
  free(A);
  free(tau);
  return 0;
}

