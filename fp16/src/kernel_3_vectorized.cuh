#pragma once

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>


__inline__ __device__ float dot_product(float *a, float *b) {
  float sum = 0.f;
  __half2* a_half2 = reinterpret_cast<__half2*>(a);
  __half2* b_half2 = reinterpret_cast<__half2*>(b);

  // compute dot product of a_half2 and b_half2
  sum += __half2float(a_half2->x) * __half2float(b_half2->x);
  sum += __half2float(a_half2->y) * __half2float(b_half2->y);   

  return sum;
}

__global__ void vectorized_mem_load(__half* __restrict__ matd, __half* __restrict__ vecd, __half* __restrict__ resd, int M, int K, int N) {
  extern __shared__ float smem[];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int out_row = blockIdx.y;
  if (bid >= N) return;

  //int n_float4s = K / 4;
  int n_half8s = K / 8;

  // cast the matrix and vector as float4
  // float4 holds multiple values (x, y, z, w)
  float4* mat_row = reinterpret_cast<float4*>(matd + bid * K);
  float4* vec = reinterpret_cast<float4*>(out_row*K + vecd);

  // each thread calculates its own partial output
  float partial_sum = 0.f;

// manual loop unrolling with a factor of 4
//#pragma unroll 4
  for (int col = tid; col < n_half8s; col += blockDim.x) {
      float4 matval = mat_row[col];
      float4 vecval = vec[col];

      partial_sum += (dot_product(&matval.x, &vecval.x) +
                      dot_product(&matval.y, &vecval.y) +
                      dot_product(&matval.z, &vecval.z) +
                      dot_product(&matval.w, &vecval.w));
  }
  // block level sum reduction
  // only first thread reads the first location in shared memory
  // only first thread writes the output to global memory
  blockReduceSum(partial_sum, smem, tid, blockDim.x);
  if (tid == 0) {
      float sum = smem[0];
      resd[out_row * N + bid] = __float2half(sum);
  }
  
}

void run_vectorized_mem_load(Problem_InstanceFP16 &pi)
{
  __half* __restrict__ matd = pi.dBt; // TODO transpose
  __half* __restrict__ vecd = pi.dA;
  __half* __restrict__ resd = pi.dC;
  
  int NUM_THREADS = 64;
  int warp_size = 32;

  dim3 block_size(NUM_THREADS);
  dim3 grid_size(pi.N, pi.M);
  size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

  vectorized_mem_load<<<grid_size, block_size, shared_mem_size>>>(matd, vecd, resd, pi.M, pi.K, pi.N);
}