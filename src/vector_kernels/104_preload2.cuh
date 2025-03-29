#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "vector_kernels/vector_kernel_utils.cuh"

/*

Matrix sizes:
MxK * KxN = MxN


/*

__global__ void vector_preload()

*/

__global__ void vector_preload2(const int M, const int N, const int K, const float *A,
                                const float *B, float *C)
{

  // number of threads in a block
  const uint out_M = blockIdx.y;
  const uint out_N = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int THREAD_COUNT = 1024;

  // Preload full row of A into the shared memory
  extern __shared__ float As[THREAD_COUNT];
  extern __shared__ float As2[THREAD_COUNT];
  
  float *current_As = As;
  float *other_As = As2;
  float *tmp_As;
  

  // if statement is necessary to make things work under tile quantization

  double tmp = 0.0;
  for (int block_BK = 0; block_BK < VCEIL_DIV(K, THREAD_COUNT); block_BK++)
  {

    // Load A into shared memory
    int A_row = block_BK * THREAD_COUNT + threadIdx.x;
    if (A_row < K)
    {
      current_As[threadIdx.x] = A[out_M * K + A_row];
    }
    __syncthreads();

    for (int iBK = 0; iBK < THREAD_COUNT; ++iBK)
    {
      int iK = block_BK * THREAD_COUNT + iBK;
      if (iK < K && out_N < N)
      {
        tmp += current_As[iBK] * B[iK * N + out_N];
      }
    }
    tmp_As = current_As;
    current_As = other_As;
    other_As = tmp_As;
  }
  if (out_N < N)
  {
    C[out_M * N + out_N] = tmp;
  }
}
