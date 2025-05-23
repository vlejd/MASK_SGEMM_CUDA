#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

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


__global__ void packed_fp16_mask_t_warp_over_k(int M, int N, int K, int K_packed, const __half *A,
                               const __half *Bcsc, const int *B_col_statrs, const int *B_packed_mask_t,
                               __half *C)
{
    const uint iM = blockIdx.x;
    const uint iN = blockIdx.y;

    const uint thread_id = threadIdx.x;
    //const uint warp_id = thread_id / 32;
    const uint lane_id = thread_id % 32;

    float tmp = 0.0;
    int value_index = B_col_statrs[iN];
    //printf("iM: %d, iN: %d, value_index: %d\n", iM, iN, value_index);

    // load mask
    // compute the prefix sum of the mask
    // threads with ones load the value.



    for (int iK_packed = 0; iK_packed < K_packed; ++iK_packed)
    {
        int packed_mask_value = B_packed_mask_t[iN*K_packed + iK_packed];
        int ones_count = __popc(packed_mask_value);
        bool my_mask = (packed_mask_value >> lane_id) & 1;

        // compute the prefix sum of the mask
        int prefix_sum = warp_prefix_sum(my_mask);
        // potentially unload this to shared memory... 

        if(my_mask){
            __half a_value = A[iM * K + iK_packed * 32 + lane_id];
            __half b_value = Bcsc[value_index+prefix_sum];
            tmp += __half2float(a_value) * __half2float(b_value);
        }

        value_index += ones_count; 
    }

    tmp = warp_sum(tmp);
    if (lane_id == 0) {
        // reduce the result across the warp
        C[iM * N + iN] = __float2half(tmp);
    }
}

void run_packed_fp16_mask_t_warp_over_k(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(pi.M, pi.N);
    dim3 blockDim(32);
    packed_fp16_mask_t_warp_over_k<<<gridDim, blockDim>>>(
        pi.M, pi.N, pi.K, pi.K_packed, pi.dA,
        pi.dBcsc, pi.dBcsc_col_starts, pi.dPacked_mask_t,
        pi.dC);
}
