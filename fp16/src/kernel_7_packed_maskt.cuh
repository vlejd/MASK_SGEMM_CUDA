#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

__global__ void packed_fp16_mask_t(int M, int N, int K, int K_packed, const __half *A,
                               const __half *Bcsc, const int *B_col_statrs, const int *B_packed_mask_t,
                               __half *C)
{
    const uint iM = blockIdx.x;
    const uint iN = blockIdx.y;

    float tmp = 0.0;
    int value_index = B_col_statrs[iN];
    //printf("iM: %d, iN: %d, value_index: %d\n", iM, iN, value_index);

    for (int iK_packed = 0; iK_packed < K_packed; ++iK_packed)
    {
        int packed_mask_value = B_packed_mask_t[iN*K_packed + iK_packed];
        
        for(int i32=0; i32 < 32 && i32 + iK_packed*32 < K; i32++){
            int mask_value = (packed_mask_value >> i32) & 1;
            if(mask_value > 0){
                __half a_value = A[iM * K + iK_packed * 32 + i32];
                __half b_value = Bcsc[value_index];
                tmp += __half2float(a_value) * __half2float(b_value);
                value_index++;
            }
        }
    }
    C[iM * N + iN] = __float2half(tmp);
}

void run_packed_fp16_mask_t(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(pi.M, pi.N);
    dim3 blockDim(1);
    packed_fp16_mask_t<<<gridDim, blockDim>>>(
        pi.M, pi.N, pi.K, pi.K_packed, pi.dA,
        pi.dBcsc, pi.dBcsc_col_starts, pi.dPacked_mask_t,
        pi.dC);
}
