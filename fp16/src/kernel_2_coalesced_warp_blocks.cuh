#pragma once

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>




__global__ void coalesced_warp_block(int aM, int aN, int aK, __half *A, __half *B, __half *C) {
  extern __shared__ float smem[];

  int M = aN;
  int N = aK;


  int bid = blockIdx.x;
  if (bid >= M) return;

  int tid = threadIdx.x;
  // each thread calculates its own partial output
  float partial_sum = 0.f;
  for (int col = tid; col < N; col += blockDim.x) {
      partial_sum +=  __half2float(B[bid * N + col]) * __half2float(A[col]);
  }

  // block level sum reduction
  // only first thread reads the first location in shared memory
  // only first thread writes the output to global memory
  blockReduceSum(partial_sum, smem, tid, blockDim.x);
  if (tid == 0) {
      __half sum = smem[0];
      C[bid] = __float2half(sum);
  }
}

void run_coalesced_warp_block(Problem_InstanceFP16 &pi)
{
  int NUM_THREADS = 64;
  int warp_size = 32;

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(pi.N);
  size_t shared_mem_size = CEIL_DIV(blockDim.x, warp_size) * sizeof(float);

  coalesced_warp_block<<<gridDim, blockDim, shared_mem_size>>>(pi.M, pi.N, pi.K, pi.dA, pi.dBt, pi.dC);
}