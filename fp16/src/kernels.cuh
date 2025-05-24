#pragma once

#include <kernel_0_cublas.cuh>
#include <kernel_1_naive.cuh>
#include <kernel_2_coalesced_warp_blocks.cuh>
#include <kernel_3_vectorized.cuh>
#include <kernel_4_naive_csc.cuh>
#include <kernel_6_naive_mask.cuh>
#include <kernel_7_packed_maskt.cuh>
#include <kernel_8_packed_maskt_warp_over_k.cuh>
#include <kernel_9_packed_maskt_warp_over_k_with_block.cuh>
#include <kernel_10_packed_maskt_only_load.cuh>
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
    case 2:
        run_coalesced_warp_block(pi);
        break;
    case 3:
        run_vectorized_mem_load(pi);
        break;
    case 4:
        run_naive_fp16_csc(pi);
        break;
    case 6:
        run_naive_fp16_mask(pi);
        break;
    case 7:
        run_packed_fp16_mask_t(pi);
        break;
    case 8:
        run_packed_fp16_mask_t_warp_over_k(pi);
        break;
    case 9:
        run_packed_fp16_mask_t_warp_over_k_with_block(pi);
        break;
    case 10:
        run_packed_fp16_mask_t_only_load(pi);
        break;
    default:
        throw std::invalid_argument("Unknown kernel number");
    }
}