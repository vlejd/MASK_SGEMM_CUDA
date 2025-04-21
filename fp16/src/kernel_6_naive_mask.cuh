#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

__global__ void naive_fp16_mask(int M, int N, int K, const __half *A,
                               const __half *Bcsc, const int *B_col_statrs, const int *B_mask,
                               __half *C)
{
    const uint iM = blockIdx.x;
    const uint iN = blockIdx.y;

    float tmp = 0.0;
    int value_index = B_col_statrs[iN];
    //printf("iM: %d, iN: %d, value_index: %d\n", iM, iN, value_index);

    for (int iK = 0; iK < K; ++iK)
    {
        int mask_value = B_mask[iK*N + iN];
        //printf("mask_value: %d\n, ik %d", mask_value, iK);
        if(mask_value > 0){
            __half a_value = A[iM * K + iK];
            __half b_value = Bcsc[value_index];
            tmp += __half2float(a_value) * __half2float(b_value);
            value_index++;
        }
    }
    C[iM * N + iN] = __float2half(tmp);
}

void run_naive_fp16_mask(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(pi.M, pi.N);
    dim3 blockDim(1);
    naive_fp16_mask<<<gridDim, blockDim>>>(
        pi.M, pi.N, pi.K, pi.dA,
        pi.dBcsc, pi.dBcsc_col_starts, pi.dMask,
        pi.dC);
}
