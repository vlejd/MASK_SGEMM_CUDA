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
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


void cudaCheckInternal(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

const std::string errLogFile = "matrixValidationFailure.txt";
const std::string dbgLogFile = "matrixValidationDebug.txt";

class Problem_InstanceFP16
{
public:
    int M, N, K;
    int seed;
    __half *hA, *hB, *hBt, *hC, *hC_ref;
    __half *dA, *dB, *dBt, *dC, *dC_ref;
    int *hMask;
    int *dMask;
    float density;
    void get_result();
    void get_result_ref();
    Problem_InstanceFP16(int M, int N, int K, float density, int seed = 0);
    ~Problem_InstanceFP16();
};

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
        float tmp = (float)(rand() % 4);
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

Problem_InstanceFP16::Problem_InstanceFP16(int M, int N, int K, float density, int seed)
{
    this->M = M;
    this->N = N;
    this->K = K;
    this->density = density;
    this->seed = seed;

    this->hA = (__half *)malloc(sizeof(__half) * this->M * this->K);
    this->hB = (__half *)malloc(sizeof(__half) * this->K * this->N);
    this->hBt = (__half *)malloc(sizeof(__half) * this->K * this->N);
    this->hC = (__half *)malloc(sizeof(__half) * this->M * this->N);
    this->hC_ref = (__half *)malloc(sizeof(__half) * this->M * this->N);
    this->hMask = (int *)malloc(sizeof(int) * this->K * this->N);

    randomize_matrix(this->hA, this->M * this->K, this->seed);
    randomize_matrix(this->hB, this->K * this->N, this->seed + 1);
    zero_init_matrix(this->hC, this->M * this->N);
    zero_init_matrix(this->hC_ref, this->M * this->N);
    generate_mask(this->hMask, this->K, this->N, this->density, this->seed + 2);
    // apply_mask(this->hB, this->hMask, this->K, this->N);
    // transpose(this->hB, this->hBt, this->K, this->N);

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

void log_matrix_data(const std::string &fileName, const Problem_InstanceFP16 &pi)
{
    std::ofstream fs;
    fs.open(fileName);
    fs << "A:\n";
    print_matrix(pi.hA, pi.M, pi.K, fs);
    fs << "B:\n";
    print_matrix(pi.hB, pi.K, pi.N, fs);
    fs << "Bt:\n";
    print_matrix(pi.hBt, pi.K, pi.N, fs);
    fs << "Mask:\n";
    print_matrix(pi.hMask, pi.K, pi.N, fs);
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