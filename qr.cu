#include <iostream>
#include <cmath>

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

//scalar type and panel size (RxC)
#define Scalar float
#define R 128
#define C 32

extern __shared__ float currentPanel[R][C];

//(2RC + C) * sizeof(Scalar) must fit in 48 KiB
//mat should be column-major
__global__ void mmqr(Scalar* mat, int m, int n)
{
  //iterate over all subdiagonal panels
  //first left to right
  for(int pc = 0; pc < n; pc += C)
  {
    //then bottom to top, sliding panel up by R-C each iteration
    //TODO: in between iterations, keep the overlapping rows in shared mem
    for(int pr = m - R; pr >= pc; pr -= (R-C))
    {
      //load panel into shared

    }
  }
}

//left and right are RxC arrays
//left is the current panel being eliminated
//right is a panel in the trailing matrix
__device__ void applyPanel()
{
}

__global__ void add(int* a, int* b, int* c)
{
  int i = blockIdx.x;
  c[i] = a[i] + b[i];
}

int main()
{
  int* a;
  int* b;
  int* c;
  cudaMalloc((void**) &a, R);
  cudaMalloc((void**) &b, R);
  cudaMalloc((void**) &c, R);
  add<<<R, 1>>>(a, b, c);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}

