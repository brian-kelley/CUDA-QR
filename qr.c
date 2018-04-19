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
      //for each column, compute HH reflectors
      for(int col = 0; col < C; col++)
      {
        Scalar norm = 0;
        for(int row = col; row < C; row++)
        {
          norm += panel[col][row] * panel[col][row];
        }
        norm = sqrt(norm);
        Scalar sign = panel[col][col] < 0 ? -1 : 1;
        Scalar u = panel[col][col] + sign * norm;
        panel[col][col] = -sign * normx;
        //is the panel at the bottom of A?
        bool bottomPanel = pr == m - R;
        //does col 0 of panel cross A's diagonal?
        bool topPanel = pr <= pc;
        //(middle panels are both top and bottom)
        int wstart;
        int wend;
        if(topPanel)
          wend = pr + (R-C) + col;
        else
          wend = R;
        if(bottomPanel)
          wstart = col;
        else
          wstart = -pr + pc + col;
        int wlen = wend - wstart;
        Scalar* w = malloc(wlen * sizeof(Scalar));
        //compute entire w explicitly,
        //then write back nontrivial entries to the panel
        w[0] = 1;
        for(int i = wstart + 1; i < wend; i++)
        {
          panel[col][i] /= u;
          w[i - wstart] = panel[col][i];
        }
        panelTau[col] = sign * u / norm;
        //apply reflector in col to remaining columns in panel
        for(int applyCol = col; applyCol < C; applyCol++)
        {
          for(int applyRow = wstart; applyRow < wend; applyRow++)
          {
            int windex = applyRow - wstart;
            Scalar val = panel[applyCol][applyRow];
            for(int i = 0; i < wlen; i++)
            {
              val -= panelTau[applyCol] * w[windex] * w[i] * panel[applyCol][applyRow];
            }
            panel[applyCol][applyRow] = val;
          }
        }
        free(w);
      }
      //panel and panelTau are now both fully computed
      //write back panel to A
      for(int col = 0; col < C; col++)
      {
        memcpy(&mat[pr + pc * m], &panel[col][0], sizeof(Scalar) * R);
      }
      /*
       * TODO: use real blocked algo
       * 
      //compute W explicitly, so that trailing updates can be a series of
      //mat-vecs in shared memory
      //
      //the Y matrix is read from the subdiagonal part of the panel that was
      //just computed
      //
      //see Kerr, Campbell, Richards paper for this (Algorithm 2)
      Scalar W[C][R];
      //first column of W is special: just w
      //compute each column of W (j) which corresponds to tcol in actual matrix
      for(int j = 1; j < C; j++)
      {
        int tcol = pc + j;
        for(int trow = pr; trow < pr + R; trow++)
        {
        }
      }
      */
      //update each trailing column (pr:pr+R, pc+C:N):
      //multiply each column by (I - tau * ww')
      Scalar w[R];
      for(int applyCol = pc + C; applyCol < n; applyCol++)
      {
        for(int applyRow = pr; applyRow < pr + R; applyRow++)
        {
          Scalar val = mat[applyCol][applyRow];
          for(int i = 0; i < wlen; i++)
          {
            Scalar wwVal;
            val -= panelTau[applyCol] * w[windex] * w[i] * panel[applyCol][applyRow];
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

