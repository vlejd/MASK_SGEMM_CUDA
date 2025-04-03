#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "vector_kernels/vector_kernel_utils.cuh"


__global__ void coalesced_warp_sgmev(int aM, int aN, int aK, float *A, float *B, float *C) {
    int M = aN;
    int N = aK;

    assert(blockDim.x == warpSize);

    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    // each thread calculates its own partial output
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += B[bid * N + col] * A[col];
    }

    // warp level sum reduction
    // only first thread writes the output to global memory
    float sum = warpReduceSum(partial_sum);
    if (tid == 0) {
        C[bid] = sum;
    }
}
