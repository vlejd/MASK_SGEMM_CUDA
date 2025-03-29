#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void vector_preload(const int M, const int N, const int K, const float *A,
                               const float *B, float *C)
{

  // number of threads in a block
  const uint out_M = blockIdx.y;
  const uint out_N = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr int THREAD_COUNT = 1024;

  // Preload full row of A into the shared memory
  extern __shared__ float As[THREAD_COUNT];

  // if statement is necessary to make things work under tile quantization
  if (out_N < N)
  {
    float tmp = 0.0;
    for (int block_BK = 0; block_BK < CEIL_DIV(K, THREAD_COUNT); block_BK++)
    {

      // Load A into shared memory
      int A_row = block_BK * THREAD_COUNT + threadIdx.x;
      if (A_row < K)
      {
        As[threadIdx.x] = A[out_M * K + A_row];
      }
      __syncthreads();

      for (int iBK = 0; iBK < THREAD_COUNT; ++iBK)
      {
        int iK = block_BK * THREAD_COUNT + iBK;
        if (iK < K)
        {
          tmp += As[iBK] * B[iK * N + out_N];
        }
      }
    }
    // C = α*(A@B)+β*C
    C[out_M * N + out_N] = tmp;
  }
}