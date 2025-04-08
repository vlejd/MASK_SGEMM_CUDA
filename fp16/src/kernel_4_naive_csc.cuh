#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

__global__ void naive_fp16_csc(int M, int N, int K, const __half *A,
                               const __half *Bcsc, const int *B_col_statrs, const int *B_rows,
                               __half *C)
{
    const uint iM = blockIdx.x;
    const uint iN = blockIdx.y;

    float tmp = 0.0;
    int col_start = B_col_statrs[iN];
    int col_end = B_col_statrs[iN + 1];

    for (int i = col_start; i < col_end; ++i)
    {
        int value_index = B_rows[i];
        __half value = Bcsc[i];
        tmp += __half2float(A[iM * K + value_index]) * __half2float(value);
    }
    C[iM * N + iN] = __float2half(tmp);
}

void run_naive_fp16_csc(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(pi.M, pi.N);
    dim3 blockDim(1);
    naive_fp16_csc<<<gridDim, blockDim>>>(
        pi.M, pi.N, pi.K, pi.dA,
        pi.dBcsc, pi.dBcsc_col_starts, pi.dBcsc_rows,
        pi.dC);
}
