#pragma once

#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils.cuh>

void runCublasFP16(cublasHandle_t handle, Problem_InstanceFP16 &pi, bool ref)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    // cublasGemmEx
    __half *result_pointer;
    if (ref)
    {
        result_pointer = pi.dC_ref;
    }
    else
    {
        result_pointer = pi.dC;
    }
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        pi.N, pi.M, pi.K,
        &alpha,
        pi.dB, CUDA_R_16F, pi.N,
        pi.dA, CUDA_R_16F, pi.K,
        &beta,
        result_pointer, CUDA_R_16F, pi.N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS error: " << status << std::endl;
    }
}
