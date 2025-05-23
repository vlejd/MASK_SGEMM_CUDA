#pragma once

#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define cudaCheck(err) (cudaCheckInternal(err, __FILE__, __LINE__))
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

using u64 = unsigned long long;
using s64 = long long int;
using u32 = unsigned int;
using s32 = int;
using u16 = unsigned short;
using s16 = short;

#define WARP_SIZE 32


void cudaCheckInternal(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void zero_init_matrix(__half *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = 0.0;
    }
}

void randomize_matrix(__half *mat, int N, int seed)
{
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    srand(seed);
    for (int i = 0; i < N; i++)
    {
        // float tmp = (float)(rand() % 5) + 0.1 * (rand() % 5);
        // tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        float tmp = (float)(rand() % 4 + 1);
        mat[i] = __float2half(tmp);
    }
}

void generate_mask(int *mask, int M, int N, float density, int seed)
{
    srand(seed);
    for (int i = 0; i < M * N; i++)
    {
        mask[i] = (rand() % 100 < density * 100) ? 1 : 0;
    }
}

void apply_mask(__half *mat, int *mask, int M, int N)
{
    for (int i = 0; i < M * N; i++)
    {
        mat[i] = (mask[i] == 0) ? __half{0} : mat[i];
    }
}

template <typename T>
void transpose(T *src, T *dst, int K, int N)
{
    // souce is K*N, dest should be N*K
    //  transpose the matrix
    for (int iK = 0; iK < K; iK++)
    {
        for (int iN = 0; iN < N; iN++)
        {
            dst[iN * K + iK] = src[iK * N + iN];
        }
    }
}

int count_nonzeros(int *mask, int M, int N)
{
    int count = 0;
    for (int i = 0; i < M * N; i++)
    {
        if (mask[i] != 0)
            count++;
    }
    return count;
}

void to_csc(__half *src, int *mask, __half *dst, int *row_indices, int *col_starts, int M, int N)
{
    // convert the matrix to csc format
    // src is K*N, dst is nonzero_count
    // row_indices is nonzero_count
    // col_starts is N+1
    int count = 0;
    for (int iN = 0; iN < N; iN++)
    {
        col_starts[iN] = count;
        for (int iM = 0; iM < M; iM++)
        {
            if (mask[iM * N + iN] != 0)
            {
                row_indices[count] = iM;
                dst[count] = src[iM * N + iN];
                count++;
            }
        }
    }
    col_starts[N] = count;
}

void pack_rows(int *src, int *dst, int n_rows, int n_cols, int n_packed)
{
    for (int i_row = 0; i_row < n_rows; i_row++)
    {
        for (int i_col_group = 0; i_col_group < n_cols; i_col_group+=32)
        {
            int packed = 0;
            for (int i_group = 0; i_group < 32 && i_col_group + i_group < n_cols; i_group++)
            {
                packed |= (src[i_row * n_cols + i_col_group + i_group] << i_group);
            } 
            dst[i_row * n_packed + i_col_group / 32] = packed;
        }
    }
}


const std::string errLogFile = "matrixValidationFailure.txt";
const std::string dbgLogFile = "matrixValidationDebug.txt";

class Problem_InstanceFP16
{
public:

    // A is MxK
    // B is KxN
    // C is MxN
    int M, N, K, K_packed;
    int seed;
    __half *hA, *hB, *hBt, *hC, *hC_ref, *hBcsc;
    __half *dA, *dB, *dBt, *dC, *dC_ref, *dBcsc;
    int *hBcsc_col_starts, *hBcsc_rows;
    int *dBcsc_col_starts, *dBcsc_rows; 
    int nonzero_count;
    int *hMask, *hMask_t, *hPacked_mask_t;
    int *dMask, *dPacked_mask_t;
    // todo transpose mask
    float density;
    void get_result();
    void get_result_ref();
    Problem_InstanceFP16(int M, int N, int K, float density, int seed = 0);
    ~Problem_InstanceFP16();
};

Problem_InstanceFP16::Problem_InstanceFP16(int M, int K, int N, float density, int seed)
{
    this->M = M;
    this->N = N;
    this->K = K;
    this->density = density;
    this->seed = seed;
    this->K_packed = CEIL_DIV(this->K, 32);

    this->hA = (__half *)malloc(sizeof(__half) * this->M * this->K);
    this->hB = (__half *)malloc(sizeof(__half) * this->K * this->N);
    this->hBt = (__half *)malloc(sizeof(__half) * this->K * this->N);
    this->hC = (__half *)malloc(sizeof(__half) * this->M * this->N);
    this->hC_ref = (__half *)malloc(sizeof(__half) * this->M * this->N);
    this->hMask = (int *)malloc(sizeof(int) * this->K * this->N);
    this->hMask_t = (int *)malloc(sizeof(int) * this->N * this->K);
    this->hPacked_mask_t = (int *)malloc(sizeof(int) * this->N * this->K_packed);

    randomize_matrix(this->hA, this->M * this->K, this->seed);
    randomize_matrix(this->hB, this->K * this->N, this->seed + 1);
    zero_init_matrix(this->hC, this->M * this->N);
    zero_init_matrix(this->hC_ref, this->M * this->N);
    generate_mask(this->hMask, this->K, this->N, this->density, this->seed + 2);
    apply_mask(this->hB, this->hMask, this->K, this->N);
    transpose(this->hB, this->hBt, this->K, this->N);
    transpose(this->hMask, this->hMask_t, this->K, this->N);
    pack_rows(this->hMask_t, this->hPacked_mask_t, this->N, this->K, this->K_packed);

    cudaCheck(cudaMalloc((void **)&this->dA, sizeof(__half) * this->M * this->K));
    cudaCheck(cudaMalloc((void **)&this->dB, sizeof(__half) * this->K * this->N));
    cudaCheck(cudaMalloc((void **)&this->dBt, sizeof(__half) * this->K * this->N));
    cudaCheck(cudaMalloc((void **)&this->dC, sizeof(__half) * this->M * this->N));
    cudaCheck(cudaMalloc((void **)&this->dC_ref, sizeof(__half) * this->M * this->N));

    cudaCheck(cudaMemcpy(this->dA, this->hA, sizeof(__half) * this->M * this->K, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dB, this->hB, sizeof(__half) * this->K * this->N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dBt, this->hBt, sizeof(__half) * this->K * this->N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dC, this->hC, sizeof(__half) * this->M * this->N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dC_ref, this->hC_ref, sizeof(__half) * this->M * this->N, cudaMemcpyHostToDevice));

    this->nonzero_count = count_nonzeros(this->hMask, this->K, this->N);

    this->hBcsc = (__half *)malloc(sizeof(__half) * this->nonzero_count);
    this->hBcsc_rows = (int *)malloc(sizeof(int) * this->nonzero_count);
    this->hBcsc_col_starts = (int *)malloc(sizeof(int) * (N+1));

    to_csc(this->hB, this->hMask, this->hBcsc, this->hBcsc_rows, this->hBcsc_col_starts, this->K, this->N);

    cudaCheck(cudaMalloc((void **)&this->dBcsc, sizeof(__half) * this->nonzero_count));
    cudaCheck(cudaMalloc((void **)&this->dBcsc_rows, sizeof(int) * this->nonzero_count));
    cudaCheck(cudaMalloc((void **)&this->dBcsc_col_starts, sizeof(int) * (N+1)));
    cudaCheck(cudaMalloc((void **)&this->dMask, sizeof(int) * this->K* this->N));
    cudaCheck(cudaMalloc((void **)&this->dPacked_mask_t, sizeof(int) * this->N * this->K_packed));   
    
    cudaCheck(cudaMemcpy(this->dBcsc, this->hBcsc, sizeof(__half) * this->nonzero_count, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dBcsc_rows, this->hBcsc_rows, sizeof(int) * this->nonzero_count, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dBcsc_col_starts, this->hBcsc_col_starts, sizeof(int) * (N+1), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dMask, this->hMask, sizeof(int) * this->K * this->N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(this->dPacked_mask_t, this->hPacked_mask_t, sizeof(int) * this->N * this->K_packed, cudaMemcpyHostToDevice));

}

void Problem_InstanceFP16::get_result()
{
    cudaCheck(cudaMemcpy(this->hC, this->dC, sizeof(__half) * this->M * this->N, cudaMemcpyDeviceToHost));
}

void Problem_InstanceFP16::get_result_ref()
{
    cudaCheck(cudaMemcpy(this->hC_ref, this->dC_ref, sizeof(__half) * this->M * this->N, cudaMemcpyDeviceToHost));
}

Problem_InstanceFP16::~Problem_InstanceFP16()
{
    free(this->hA);
    free(this->hB);
    free(this->hBt);
    free(this->hC);
    cudaFree(this->dA);
    cudaFree(this->dB);
    cudaFree(this->dBt);
    cudaFree(this->dC);
}
template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs)
{
    int i;
    fs << std::setprecision(2)
       << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for (i = 0; i < M && i < 32; i++)
    {
        for (int j = 0; j < N && j < 32; j++)
        {
            fs << __half2float(A[i * N + j]);
            if (j != N - 1)
                fs << ", ";
        }
        if (i != M - 1)
            fs << ";\n";
    }
    fs << "]\n";
}

void print_int_matrix(const int *A, int M, int N, std::ofstream &fs)
{
    int i;
    fs << std::setprecision(2)
       << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for (i = 0; i < M && i < 32; i++)
    {
        for (int j = 0; j < N && j < 32; j++)
        {
            fs << A[i * N + j];
            if (j != N - 1)
                fs << ", ";
        }
        if (i != M - 1)
            fs << ";\n";
    }
    fs << "]\n";
}

void log_matrix_data(const std::string &fileName, const Problem_InstanceFP16 &pi)
{
    std::ofstream fs;
    fs.open(fileName);
    fs << "A:\n";
    print_matrix(pi.hA, pi.M, pi.K, fs);
    fs << "B:\n";
    print_matrix(pi.hB, pi.K, pi.N, fs);
    fs << "Bcsc:\n";
    print_matrix(pi.hBcsc, 1, pi.nonzero_count, fs);
    fs << "Bcsc_rows:\n";
    print_int_matrix(pi.hBcsc_rows, 1, pi.nonzero_count, fs);
    fs << "Bcsc_col_starts:\n";
    print_int_matrix(pi.hBcsc_col_starts, 1, pi.N + 1, fs);
    fs << "Bt:\n";
    print_matrix(pi.hBt, pi.N, pi.K, fs);
    fs << "Mask:\n";
    print_matrix(pi.hMask, pi.K, pi.N, fs);
    fs << "Packed_mask_t:\n";
    print_int_matrix(pi.hPacked_mask_t, pi.N, pi.K_packed, fs);
    fs << "C:\n";
    print_matrix(pi.hC, pi.M, pi.N, fs);
    fs << "Should:\n";
    print_matrix(pi.hC_ref, pi.M, pi.N, fs);
};

bool verify_result(Problem_InstanceFP16 &pi)
{
    double diff = 0.0;
    int i;
    for (i = 0; i < pi.N * pi.M; i++)
    {
        float a = __half2float(pi.hC[i]);
        float b = __half2float(pi.hC_ref[i]);
        diff = std::fabs(a - b);

        if (diff > 0.01)
        {
            std::cout << "Divergence! Should " << a << ", Is " << b
                      << " (Diff " << diff << ") at " << i << std::endl;
            return false;
        }
    }
    return true;
}


__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
  
    return val;
}

//function that computes prefix sum across the warp
__inline__ __device__ int warp_prefix_sum(int val) {
    int original_val = val;
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val-original_val;
}


// function that computes sum across the warp
__inline__ __device__ int warp_sum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__device__ __forceinline__ void blockReduceSum(float val, float *smem, int tid, int blockDimX) {
    // 1. do warpReduce sum
    val = warpReduceSum(val);
  
    // 2. do blockReduce sum
    if (blockDimX > warpSize) {
        int lane = tid % warpSize;
        int wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();
  
        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : 0.0f;
            val = warpReduceSum(val);
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    // __syncthreads();
    // sync not needed because only thread 0 reads from smem[0]
  }