#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>


void cudaCheckInternal(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N, int seed = 0);
void generate_mask(int *mask, int M, int N, float density, int seed = 0);
void apply_mask(float *mat, int *mask, int M, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}
bool verify_matrix(float *mat1, float *mat2, int N);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);

class Problem_InstanceFP32 {
  public:
    int M, N, K;
    int seed;
    float *hA, *hB, *hC, *hC_ref;
    float *dA, *dB, *dC, *dC_ref;
    int *hMask;
    int *dMask;
    float density;
    void get_result();
    void get_result_ref();
    Problem_InstanceFP32(int M, int N, int K, float density, int seed=0);
    ~Problem_InstanceFP32();
};

void run_vector_kernel(int kernel_num, Problem_InstanceFP32 &pi, float alpha, float beta);
