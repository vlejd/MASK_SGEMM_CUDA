#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

#define WARP_COUNT 1

__global__ void packed_fp16_mask_t_only_load(int M, int N, int K, int K_packed, const __half *A,
                               const __half *Bcsc, const int *B_col_statrs, const int *B_packed_mask_t,
                               __half *C)
{
    extern __shared__ float smem[];
    const uint iM = blockIdx.x;
    const uint iN = blockIdx.y;

    const uint thread_id = threadIdx.x;
    const uint warp_id = thread_id / 32;
    const uint lane_id = thread_id % 32;
    const uint blockDim_x = blockDim.x;

    float tmp = 0.0;
    int value_index = B_col_statrs[iN];
    int value_index_end = B_col_statrs[iN+1];
    int num_values = value_index_end - value_index;

    int mask_nums = 0;
    for (int iK_packed = 0; iK_packed < K_packed; iK_packed+=blockDim_x)
    {
        int mask = B_packed_mask_t[iN*K_packed + iK_packed+ thread_id];
        //mask_nums += __popc(mask);
        mask_nums += mask & 1;
    }

    int a_vals = 0;
    for (int iK=0; iK < K_packed; iK += blockDim_x)
    {
        __half a_value = A[iM * K + iK + thread_id];
        a_vals += __half2float(a_value);
    }

    int b_vals = 0;
    for (int iK = 0; iK < num_values; iK += blockDim_x)
    {
        __half b_value = Bcsc[value_index+thread_id + iK];
        b_vals += __half2float(b_value); 
    }

    //tmp = warpReduceSum(tmp);
    tmp = a_vals + b_vals + mask_nums;
    blockReduceSum(tmp, smem, thread_id, blockDim.x);
    if (thread_id == 0) {
        // reduce the result across the warp
        __half sum = smem[0];  
        C[iM * N + iN] = __float2half(sum);
    }
}

void run_packed_fp16_mask_t_only_load(Problem_InstanceFP16 &pi)
{
    dim3 gridDim(pi.M, pi.N);
    dim3 blockDim(32*WARP_COUNT);
    size_t shared_mem_size = WARP_COUNT * sizeof(float);

    packed_fp16_mask_t_only_load<<<gridDim, blockDim, shared_mem_size>>>(
        pi.M, pi.N, pi.K, pi.K_packed, pi.dA,
        pi.dBcsc, pi.dBcsc_col_starts, pi.dPacked_mask_t,
        pi.dC);
}
