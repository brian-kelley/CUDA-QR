#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

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

//mat should be column-major
__global__ void mmqr(Scalar* mat, int m, int n)
{
  //iterate over all subdiagonal panels
  //first left to right
  for(int pc = 0; pc < n; pc += C)
  {
    //then bottom to top, sliding panel up by R-C each iteration
    //TODO: in between iterations, keep the overlapping rows in shared mem
    //(shift those entries down (to higher indices) by R-C)
    for(int pr = m - R; pr >= pc; pr -= (R-C))
    {
      //load panel into shared memory, one column at a time
      __shared__ Scalar panel[C][R];
      for(int col = 0; col < C; col++)
      {
        memcpy(&panel[col][0], &mat[pr + pc * m], sizeof(Scalar) * R);
      }
      //for each column, compute HH reflectors (serial)
      for(int col = 0; col < C; col++)
      {
        Scalar norm = 0;
        for(int row = col; row < C; row++)
        {
          norm += panel[col][row] * panel[col][row];
        }
        norm = sqrt(norm);
        Scalar sign = panel[col][col] < 0 ? -1 : 1;
        Scalar u = panel[col][col] - sign * norm;
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
  Scalar* Ahost = new Scalar[m * n];
  srand(12);
  for(int i = 0; i < m * n; i++)
  {
    Ahost[i] = float(rand()) / RAND_MAX;
  }
  //initialize A randomly
  Scalar* Adevice;
  cudaMalloc((void**) &Adevice, m * n * sizeof(Scalar));
  cudaMemcpy(Adevice, Ahost, m * n * sizeof(Scalar), cudaMemcpyHostToDevice);
  mmqr<<<1, 1>>>(Adevice, m, n);
  //retrieve Q, R into a new host buffer
  Scalar* QRhost = new Scalar[m * n];
  cudaMemcpy(QRhost, Adevice, m * n * sizeof(Scalar), cudaMemcpyDeviceToHost);
  return 0;
}

