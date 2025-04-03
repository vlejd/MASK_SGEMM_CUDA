#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "vector_kernels/vector_kernel_utils.cuh"


__global__ void transposed_naive(int M, int N, int K, const float *A,
    const float *B, float *C) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < N) {
      float sum = 0.0f;
      for (int col = 0; col < K; col++) {
          sum += B[row * K + col] * A[col];
      }
      C[row] = sum;
  }
}