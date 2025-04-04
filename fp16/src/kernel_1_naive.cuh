#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

__global__ void naive_kernel_fp16(int M, int N, int K, const __half *A,
                             const __half *B, __half *C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N)
    {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i)
        {
            tmp += __half2float(A[x * K + i]) * __half2float(B[i * N + y]);
        }
        C[x * N + y] = __float2half(tmp);
    }
}

void run_naive_kernel_fp16(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(CEIL_DIV(pi.M, 32), CEIL_DIV(pi.N, 32));
    dim3 blockDim(32, 32);
    naive_kernel_fp16<<<gridDim, blockDim>>>(pi.M, pi.N, pi.K, pi.dA, pi.dB, pi.dC);
}