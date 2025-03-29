#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void vector_warping(int M, int N, int K, const float *A,
                            const float *B, float *C) {
  const uint out_M = blockIdx.y;
  const uint out_N = blockIdx.x * blockDim.x + threadIdx.x;
  
  // if statement is necessary to make things work under tile quantization
  if (out_N < N) {
    float tmp = 0.0;
    for (int iK = 0; iK < K; ++iK) {
      tmp += A[out_M * K + iK] * B[iK * N + out_N];
    }
    // C = α*(A@B)+β*C
    C[out_M * N + out_N] = tmp;
  }
}