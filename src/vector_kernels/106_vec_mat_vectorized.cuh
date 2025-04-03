#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "vector_kernels/vector_kernel_utils.cuh"



__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
  }

  return val;
}


__device__ __forceinline__ void blockReduceSum(float val, float *smem, int tid, int blockDimX) {
  // 1. do warpReduce sum
  val = warpReduceSum(val);

  // 2. do blockReduce sum
  if (blockDimX > warpSize) {
      int lane = tid % warpSize;
      int wid = tid / warpSize;
      if (lane == 0) {
          smem[wid] = val;
      }
      __syncthreads();

      if (tid < warpSize) {
          val = tid < VCEIL_DIV(blockDimX, warpSize) ? smem[tid] : 0.0f;
          val = warpReduceSum(val);
          if (tid == 0) smem[0] = val;
      }
  } else {
      if (tid == 0) smem[0] = val;
  }
  // __syncthreads();
  // sync not needed because only thread 0 reads from smem[0]
}


__global__ void vectorized_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int N, int K) {
  extern __shared__ float smem[];

  int bid = blockIdx.x;
  if (bid >= N) return;

  int tid = threadIdx.x;
  int n_float4s = K / 4;

  // cast the matrix and vector as float4
  // float4 holds multiple values (x, y, z, w)
  float4* mat_row = reinterpret_cast<float4*>(matd + bid * K);
  float4* vec = reinterpret_cast<float4*>(vecd);

  // each thread calculates its own partial output
  float partial_sum = 0.f;

// manual loop unrolling with a factor of 4
#pragma unroll 4
  for (int col = tid; col < n_float4s; col += blockDim.x) {
      float4 matval = mat_row[col];
      float4 vecval = vec[col];

      partial_sum += (matval.x * vecval.x +
                      matval.y * vecval.y +
                      matval.z * vecval.z +
                      matval.w * vecval.w);
  }

  // block level sum reduction
  // only first thread reads the first location in shared memory
  // only first thread writes the output to global memory
  blockReduceSum(partial_sum, smem, tid, blockDim.x);
  if (tid == 0) {
      float sum = smem[0];
      resd[bid] = sum;
  }
}