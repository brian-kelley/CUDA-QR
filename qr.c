#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

//Overall algorithm:
//For each panel column (SERIAL):
//  For each dense panel from the bottom up (SERIAL):
//    Load panel into shared
//    For each panel column (SERIAL):
//      compute and store hh reflector and beta
//      apply reflectors to trailing columns of panel (possibly parallel)
//      reflectors are written back to lower part of panel,
//    For each trailing panel (PARALLEL):
//      apply the reflections to trailing panel, replacing values

//Scalar type and panel size (RxC)
//Scalar may be either float or double
//(2RC + C) * sizeof(Scalar) must fit in 48 KiB
#define Scalar float
#define R 128
#define C 16

//This file is a working prototype of the algorithm in plain C (no CUDA)

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
          wstart = pr + (R-C) + col;
        else
          wstart = pr + R;
        if(bottomPanel)
          wend = pr + col;
        else
          wend = pc + col;
        int wlen = wend - wstart;
        Scalar* w = malloc(wlen * sizeof(Scalar));
        //compute entire w explicitly,
        //writing back entries to A
        for(int i = col; i < C; i++)
        {
          w[i] = panel[col][i] / u;
          if(i >= col + 1)
            panel[col][i] = w[i];
        }
        panelTau[col] = sign * u / norm;
        //apply reflector in col to remaining columns in panel
        for(int applyCol = col; applyCol < C; applyCol++)
        {
          //put updated column in panelCol
          Scalar* panelCol = malloc(wlen * sizeof(Scalar));
          for(int applyRow = 0; applyRow < wlen; applyRow++)
          {
            float val = 0;
            for(int i = 0; i < wlen; i++)
            {
              val += panelTau[applyCol] * 
            }
            panelCol[
          }
          free(panelCol);
        }
      }
      Scalar norm = 0;
    }
  }
}

int main()
{
  int m = 1024;
  int n = 256;
  assert(m < n);
  Scalar* A = malloc(m * n * sizeof(Scalar));
  Scalar* tau = malloc(n * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = float(rand()) / RAND_MAX;
  }
  mmqr(A, tau, m, n);
  return 0;
}

