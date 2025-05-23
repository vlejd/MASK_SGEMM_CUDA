#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

#define WARP_COUNT 2

__global__ void packed_fp16_mask_t_warp_over_k_with_block(int M, int N, int K, int K_packed, const __half *A,
                               const __half *Bcsc, const int *B_col_statrs, const int *B_packed_mask_t,
                               __half *C)
{
    extern __shared__ float smem[];
    const uint iM = blockIdx.x;
    const uint iN = blockIdx.y;

    const uint thread_id = threadIdx.x;
    const uint warp_id = thread_id / 32;
    const uint lane_id = thread_id % 32;

    float tmp = 0.0;
    int value_index = B_col_statrs[iN];


    for (int iK_packed = 0; iK_packed < K_packed; iK_packed+=2)
    {
        int packed_mask_value_1 = B_packed_mask_t[iN*K_packed + iK_packed];
        int packed_mask_value_2 = B_packed_mask_t[iN*K_packed + iK_packed+1];

        int ones_count_1 = __popc(packed_mask_value_1);
        int ones_count_2 = __popc(packed_mask_value_2);
        int packed_mask_value = (warp_id == 0) ? packed_mask_value_1 : packed_mask_value_2;

        bool my_mask = (packed_mask_value >> lane_id) & 1;

        // compute the prefix sum of the mask
        int prefix_sum = warp_prefix_sum(my_mask);
        // potentially unload this to shared memory... 
        prefix_sum += (warp_id == 0) ? 0 : ones_count_1;

        if(my_mask){
            __half a_value = A[iM * K + iK_packed * 32 + lane_id + 32*warp_id];
            __half b_value = Bcsc[value_index+prefix_sum];
            tmp += __half2float(a_value) * __half2float(b_value);
        }

        value_index += ones_count_1 + ones_count_2; 
    }

    //tmp = warpReduceSum(tmp);
    blockReduceSum(tmp, smem, thread_id, blockDim.x);
    if (thread_id == 0) {
        // reduce the result across the warp
        __half sum = smem[0];  
        C[iM * N + iN] = __float2half(sum);
    }
}

void run_packed_fp16_mask_t_warp_over_k_with_block(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(pi.M, pi.N);
    dim3 blockDim(32*WARP_COUNT);
    size_t shared_mem_size = WARP_COUNT * sizeof(float);

    packed_fp16_mask_t_warp_over_k_with_block<<<gridDim, blockDim, shared_mem_size>>>(
        pi.M, pi.N, pi.K, pi.K_packed, pi.dA,
        pi.dBcsc, pi.dBcsc_col_starts, pi.dPacked_mask_t,
        pi.dC);
}
