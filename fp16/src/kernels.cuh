#pragma once

#include <kernel_0_cublas.cuh>
#include <kernel_1_naive.cuh>
#include <utils.cuh>

void run_kernel_fp16(int kernel_num, Problem_InstanceFP16 &pi, cublasHandle_t handle, bool ref)
{
    switch (kernel_num)
    {
    case 0:
        runCublasFP16(handle, pi, ref);
        break;
    case 1:
        run_naive_kernel_fp16(pi);
        break;
    default:
        throw std::invalid_argument("Unknown kernel number");
    }
}