#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

template <typename DType>
void transpose(DType *A, DType *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}

template <typename DType>
void gemm(DType *A, DType *B, DType *C, int n)
{   
    int i, j, k;
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
			DType dot = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B[k*n+j];
            } 
            C[i*n+j ] = dot;
        }
    }
}

template <typename DType>
void gemm_omp(DType *A, DType *B, DType *C, int n)
{   
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
				DType dot = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B[k*n+j];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
}

template <typename DType>
void gemmT(DType *A, DType *B, DType *C, int n)
{   
    int i, j, k;
	DType *B2;
	B2 = (DType*)malloc(sizeof(DType)*n*n);
    transpose(B,B2, n);
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
			DType dot = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B2[j*n+k];
            } 
            C[i*n+j ] = dot;
        }
    }
    free(B2);
}

template <typename DType>
void gemmT_omp(DType *A, DType *B, DType *C, int n)
{   
	DType *B2;
	B2 = (DType*)malloc(sizeof(DType)*n*n);
    transpose(B,B2, n);
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
				DType dot = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B2[j*n+k];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
    free(B2);
}