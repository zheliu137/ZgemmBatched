//Example 1. Application Using C and cuBLAS: 1-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2F(i,j,ld) ((((j))*(ld))+((i)))

// void run_eig_wrapper_(const int N, cuDoubleComplex *x);
//void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda);
int  matmul_strided_batched_comp(int n,int m,int k,
  cuDoubleComplex *A,cuDoubleComplex *B,cuDoubleComplex *C, int nmat);
void createRandoms(int size, double *randomArray);

int main (int argc, char* argv[]){
    cuDoubleComplex *A;
    cuDoubleComplex *B;
    cuDoubleComplex *C;
    int N=32;
    if (argc > 1 ) {
      N = strtol(argv[1],nullptr,0);
    }
    int nmat = 2000;
    if (argc > 2 ) {
      nmat = strtol(argv[2],nullptr,0);
    }
    A = (cuDoubleComplex *)malloc(pow(N,2)*sizeof(cuDoubleComplex)*nmat);
    B = (cuDoubleComplex *)malloc(pow(N,2)*sizeof(cuDoubleComplex)*nmat);
    C = (cuDoubleComplex *)malloc(pow(N,2)*sizeof(cuDoubleComplex)*nmat);
    double *rand1;
    double *rand2;
    
    rand1 = (double *)malloc(N*N*nmat*sizeof(double));
    rand2 = (double *)malloc(N*N*nmat*sizeof(double));
    printf("Generating %d by %d random matrix... \n",N,N);
    createRandoms( N*N*nmat, rand1 );
    createRandoms( N*N*nmat, rand2 );
    for (int l=0;l<nmat;l++){
      for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
          A[IDX2F(i,j,N)+l*N*N] = {rand1[i+j*N+l*N*N]+rand1[j+i*N+l*N*N],rand2[i+j*N+l*N*N]-rand2[j+i*N+l*N*N]};
          B[IDX2F(i,j,N)+l*N*N] = {rand1[i+j*N+l*N*N]+rand1[j+i*N+l*N*N],rand2[i+j*N+l*N*N]-rand2[j+i*N+l*N*N]};
        }
      } 
    }

    matmul_strided_batched_comp( N, N, N, A, B, C, nmat);

}