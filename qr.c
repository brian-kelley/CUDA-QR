#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

//Scalar type and panel size (RxC)
//Scalar may be either float or double
//(2RC + C) * sizeof(Scalar) must fit in 48 KiB
#define Scalar float
#define R 16
#define C 4

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
  for(int pc = 0; pc < n; pc += C)
  {
    //then bottom to top, sliding panel up by R-C each iteration
    for(int pr = m - R; pr >= pc; pr -= (R-C))
    {
      //load panel into shared memory, one column at a time
      //Note that panel is column major
      Scalar panel[C][R];
      Scalar panelTau[C];
      for(int col = 0; col < C; col++)
      {
        memcpy(&panel[col][0], &mat[pr + pc * m], sizeof(Scalar) * R);
      }
      //see Kerr/Campbell/Richards paper for blocked Householder description
      //
      //The W matrix (for applying whole panel of HH reflectors at once)
      //should be in shared and is updated as each reflector is determined
      //
      //the Y matrix is read from the subdiagonal part of the panel that was
      //just computed (doesn't need separate explicit storage)
      Scalar W[C][R];
      for(int i = 0; i < C; i++)
        for(int j = 0; j < R; j++)
          W[i][j] = 0;
      //update each trailing column (pr:pr+R, pc+C:N):
      //for each column, compute HH reflectors
      for(int col = 0; col < C; col++)
      {
        Scalar innerProd = 0;
        for(int row = col; row < C; row++)
        {
          innerProd += panel[col][row] * panel[col][row];
        }
        norm = sqrt(innerProd);
        Scalar sign = panel[col][col] < 0 ? -1 : 1;
        Scalar u = panel[col][col] + sign * norm;
        panel[col][col] = -sign * normx;
        //is the panel at the bottom of A?
        bool bottomPanel = pr == m - R;
        //does col 0 of panel cross A's diagonal?
        bool topPanel = pr <= pc;
        //(middle panels are both top and bottom)
        int vstart;
        int vend;
        if(topPanel)
          vend = pr + (R-C) + col;
        else
          vend = R;
        if(bottomPanel)
          vstart = col;
        else
          vstart = -pr + pc + col;
        int vlen = vend - vstart;
        Scalar* v = malloc(vlen * sizeof(Scalar));
        //compute entire w explicitly,
        //then write back nontrivial entries to the panel
        v[0] = 1;
        for(int i = vstart + 1; i < vend; i++)
        {
          panel[col][i] /= u;
          v[i - vstart] = panel[col][i];
        }
        //v is now fully computed
        //update W matrix
        //column 0 of W is special: can save most of the computation
        if(col == 0)
        {
          for(int i = wstart; i < wend; i++)
          {
            W[0][i] = (-2 / innerProd) * v[i - wstart];
          }
        }
        else
        {
          for(int i = wstart; i < wend; i++)
          {
            W[col][i] = (-2 / innerProd) * v[i - wstart];
          }
        }
        panelTau[col] = sign * u / norm;
        //apply reflector in col to remaining columns in panel
        for(int applyCol = col; applyCol < C; applyCol++)
        {
          for(int applyRow = vstart; applyRow < vend; applyRow++)
          {
            int vindex = applyRow - vstart;
            Scalar val = panel[applyCol][applyRow];
            for(int i = 0; i < vlen; i++)
            {
              val -= panelTau[applyCol] * v[vindex] * v[i] * panel[applyCol][applyRow];
            }
            panel[applyCol][applyRow] = val;
          }
        }
        free(v);
      }
      //panel and panelTau are now both fully computed
      //write back panel to A
      for(int col = 0; col < C; col++)
      {
        memcpy(&mat[pr + pc * m], &panel[col][0], sizeof(Scalar) * R);
      }
      //multiply each column by (I - tau * ww')
      Scalar w[R];
      for(int applyCol = pc + C; applyCol < n; applyCol++)
      {
        for(int applyRow = pr; applyRow < pr + R; applyRow++)
        {
          Scalar val = mat[applyCol][applyRow];
          for(int i = 0; i < vlen; i++)
          {
            Scalar wwVal;
            val -= panelTau[applyCol] * v[windex] * v[i] * panel[applyCol][applyRow];
          }
          panel[applyCol][applyRow] = val;
        }
      }
    }
  }
}

int main()
{
  int m = R;
  int n = C;
  assert(m >= n);
  Scalar* A = malloc(m * n * sizeof(Scalar));
  Scalar* tau = malloc(n * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = (Scalar) rand() / RAND_MAX;
  }
  printMat(A, m, n);
  //mmqr(A, tau, m, n);
  return 0;
}

