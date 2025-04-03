#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "vector_kernels/vector_kernel_utils.cuh"


__global__ void coalesced_warp_block(int aM, int aN, int aK, float *A, float *B, float *C) {
    extern __shared__ float smem[];

    int M = aN;
    int N = aK;


    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    // each thread calculates its own partial output
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += B[bid * N + col] * A[col];
    }

    // block level sum reduction
    // only first thread reads the first location in shared memory
    // only first thread writes the output to global memory
    blockReduceSum(partial_sum, smem, tid, blockDim.x);
    if (tid == 0) {
        float sum = smem[0];
        C[bid] = sum;
    }
}
