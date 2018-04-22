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
      //TODO: columns of Y matrix are just the reflectors, so an explicit
      //copy of it is unnecessary
      Scalar W[C][R];
      Scalar Y[C][R];
      for(int i = 0; i < C; i++)
      {
        for(int j = 0; j < R; j++)
        {
          W[i][j] = 0;
          Y[i][j] = 0;
        }
      }
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
        //v is now fully computed (explicitly)
        //update W matrix
        //beta is from Algorithm 3 of Kerr
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
      //panel, panelTau, W and Y are all fully computed
      //write back panel to A
      for(int col = 0; col < C; col++)
      {
        memcpy(&mat[pr + pc * m], &panel[col][0], sizeof(Scalar) * R);
      }
      //update trailing columns of A: A = (I + YW^T)A
      //the new columns of A can be updated independently (and in parallel)
      //so this loop could be a kernel launch with some A columns given to each SM
      for(int applyCol = pc + C; applyCol < n; applyCol++)
      {
        //The new column, to be copied back into A
        Scalar Acol[R];     //these vectors both go in shared
        Scalar newAcol[R];
        //gives perfect minimal memory bandwidth:
        //each entry read/written once in optimally coalesced accesses
        //the IA term above is implicit (other term added to this one)
        for(int i = 0; i < R; i++)
        {
          Acol[i] = mat[pr + i + applyCol * m];
          newAcol[i] = Acol[i];
        }
        //now compute YW^T * A[<panel rows>, applyCol] and update newAcol
        for(int i = 0; i < R; i++)
        {
          Scalar newAval = 0;
          for(int j = 0; j < R; j++)
          {
            //need inner product of row i of Y and row j of W
            Scalar ywt = 0;
            for(int k = 0; k < C; k++)
            {
              ywt += Y[i][k] * W[j][k];
            }
            //multiply that by entry j of A
            newAval += ywt * Acol[j];
          }
          newAcol[i] = += newAval;
        }
        //write back newAcol
        for(int i = 0; i < R; i++)
        {
          mat[pr + i + applyCol * m] = newAcol[i];
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

