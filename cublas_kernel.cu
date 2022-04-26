#include <math.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <algorithm>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
    
#ifdef DEBUG
#define CUSOLVER_CHECK(err) (HandlecusolverError(err, __FILE__, __LINE__))
#define CUDA_CHECK(err) (HandleError(err, __FILE__, __LINE__))
#define CUBLAS_CHECK(err) (HandleBlasError(err, __FILE__, __LINE__))
#else
#define CUSOLVER_CHECK(err) (err)
#define CUDA_CHECK(err) (err)
#define CUBLAS_CHECK(err) (err)
#endif

static void HandleBlasError(cublasStatus_t err, const char *file, int line)
{

    if (err != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cublasGetStatusString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}



static void HandlecusolverError(cusolverStatus_t err, const char *file, int line )
{

    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr, "ERROR: %d in %s at line %d, (error-code %d)\n",
                err, file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

static void HandleError(cudaError_t err, const char *file, int line)
{

    if (err != cudaSuccess)
    {
        fprintf(stderr, "ERROR: %s in %s at line %d (error-code %d)\n",
                cudaGetErrorString(err), file, line, err);
        fflush(stdout);
        exit(-1);
    }
}

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

    cublasOperation_t char_to_cublas_trans(char trans)
    {
        cublasOperation_t cuTrans;
        switch (trans)
        {
        case 'n':
        case 'N':
            cuTrans = CUBLAS_OP_N;
            break;
        case 't':
        case 'T':
            cuTrans = CUBLAS_OP_T;
            break;
        case 'c':
        case 'C':
            cuTrans = CUBLAS_OP_C;
            break;
        default:
            exit(-1);
        }
        return cuTrans;
    }

    __global__ void setting_array(cuDoubleComplex *dA, cuDoubleComplex *dB, cuDoubleComplex *dC, 
        cuDoubleComplex **dA_B, cuDoubleComplex **dB_B, cuDoubleComplex **dC_B, int batch_count,
                       int m, int n, int k){
        int idx=blockIdx.x*blockDim.x + threadIdx.x;
        if (idx<batch_count){
            dA_B[idx] = &dA[idx*m*k];
            dB_B[idx] = &dB[idx/2*k*n];
            dC_B[idx] = &dC[idx*m*n];
        }
    };

    void matmul_strided_batched_comp(int m, int n, int k, cuDoubleComplex *A, 
        cuDoubleComplex *B, cuDoubleComplex *C, int batch_count)
    {
        cuDoubleComplex alpha={1.0,0.0};
        cuDoubleComplex beta={0.0,0.0};
        int lda=m;
        int ldb=n;
        int ldc=k;
        char transa = 'n', transb = 'n';
        cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cuDoubleComplex **dA_B = nullptr, **dB_B = nullptr, **dC_B = nullptr;
        cublasHandle_t blasHandle;

        // CUDA timer
        cudaEvent_t start, stop;
        float elapsed_time;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUBLAS_CHECK(cublasCreate(&blasHandle));

        CUDA_CHECK(cudaMalloc((void **)&dA, batch_count * m * k * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&dB, batch_count * k * n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)&dC, batch_count * m * n * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **)(&dA_B), batch_count  * sizeof(cuDoubleComplex*)));
        CUDA_CHECK(cudaMalloc((void **)(&dB_B), batch_count  * sizeof(cuDoubleComplex*)));
        CUDA_CHECK(cudaMalloc((void **)(&dC_B), batch_count  * sizeof(cuDoubleComplex*)));

        CUDA_CHECK(cudaMemcpy(dA, A, batch_count * m * k * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B, batch_count * k * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        int nthreads=1024;
        int ngrids=(batch_count+nthreads-1)/nthreads;
        printf("employ %i grids and %i threads\n",ngrids,nthreads);
        setting_array<<<ngrids,nthreads>>>(dA,dB,dC,dA_B,dB_B,dC_B,batch_count,m,n,k);

        // C timer
        std::clock_t c_start = std::clock();
        struct timeval begin, end;
        gettimeofday(&begin, 0);
        int batchmax=batch_count;
        int nstep = 1;
        for (int l = 1; l <= nstep; l++){
        batch_count=l*batchmax/nstep;
        CUDA_CHECK(cudaEventRecord(start));
        // CUDA_CHECK(cudaDeviceSynchronize());
        int nloop = 10;
        for (int i = 0; i < nloop; i++){
            CUBLAS_CHECK(cublasZgemmBatched(blasHandle, char_to_cublas_trans(transa), char_to_cublas_trans(transb), m, n, k, &alpha, dA_B, lda, dB_B, ldb, &beta, dC_B, ldc, batch_count));
        };
        // CUDA_CHECK(cudaDeviceSynchronize());
        batch_count = batch_count;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("batchsize: %d, CUDA event time: %gs, avg time per matmul : %g \n",batch_count*nloop,elapsed_time/1000.0,elapsed_time/1000.0/double(batch_count*nloop));
    };
    // C timer: WALL time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    
    printf("Wall Time measured: %.3f seconds.\n", elapsed);
    // printf("batchsize: %d, CUDA event time: %gs \n",batch_count,elapsed_time/1000.0);
    
    CUDA_CHECK(cudaMemcpy(C, dC, batch_count*m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    // print_matrix(m,k,&A[m*k],m);
    // print_matrix(k,n,B,k);
    // print_matrix(n,n,&C[n*n],n);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    
    cublasDestroy(blasHandle);
    }