# Overview

Fork of https://github.com/siboehm/SGEMM_CUDA , adjusted to perform `vector` times `sparse matrix` multiplication.  

# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A100-PCIE-40GB (Ampere):

GFLOPs at dense matrices A: 4096x4096, B:4096x4096.
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |   `290.7` | 2.0%                           |
| 2: GMEM Coalescing                  |  `3006.9` | 21.3%                          |
| 3: SMEM Caching                     |  `4859.7` | 34.4%                          |
| 4: 1D Blocktiling                   |  `9260.8` | 65.7%                          |
| 5: 2D Blocktiling                   | `12276.0` | 87.1%                          |
| 8: Avoid Bank Conflicts (Offset)    | `12728.0` | 90.3%                          |
| 7: Avoid Bank Conflicts (Linearize) | `12740.6` | 90.4%                          |
| 9: Autotuning                       | `12844.7` | 91.1%                          |
| 6: Vectorized Mem Access            | `13033.9` | 92.4%                          |
| 11: Double Buffering                | `13192.7` | 93.6%                          |
| 10: Warptiling                      | `13462.2` | 95.5%                          |
| 0: cuBLAS                           | `14092.0` | 100%                           |
<!-- benchmark_results -->


GFLOPs at dense matrices A: 1x4096, B:4096x4096.
Other kernels do not run nativelly on this shape.
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |    `25.5` |                                |
| 2: GMEM Coalescing                  |    `53.5` |                                |
| 3: SMEM Caching                     |    `48.9` |                                |
| 4: 1D Blocktiling                   |    `33.7` |                                |
| 5: 2D Blocktiling                   |    `25.8` |                                |
| 0: cuBLAS                           |   `417.3` | 100.0%                         |
<!-- benchmark_results -->


GFLOPs at dense matrices A: 2x4096, B:4096x4096.
Other kernels do not run nativelly on this shape.
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |    `51.3` |                                |
| 2: GMEM Coalescing                  |   `106.5` |                                |
| 3: SMEM Caching                     |    `97.8` |                                |
| 4: 1D Blocktiling                   |    `67.5` |                                |
| 5: 2D Blocktiling                   |    `51.6` |                                |
| 0: cuBLAS                           |   `454.2` | 100.0%                         |


GFLOPs at dense matrices A: 4x4096, B:4096x4096.
Other kernels do not run nativelly on this shape.
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |    `75.0` |                                |
| 2: GMEM Coalescing                  |   `207.8` |                                |
| 3: SMEM Caching                     |   `195.5` |                                |
| 4: 1D Blocktiling                   |   `135.0` |                                |
| 5: 2D Blocktiling                   |   `103.2` |                                |
| 0: cuBLAS                           |   `932.3` | 100.0%                         |


GFLOPs at dense matrices A: 8x4096, B:4096x4096.
Other kernels do not run nativelly on this shape.
<!-- benchmark_results -->
| Kernel                              |  GFLOPs/s | Performance relative to cuBLAS |
|:------------------------------------|----------:|:-------------------------------|
| 1: Naive                            |    `89.0` |                                |
| 2: GMEM Coalescing                  |   `423.6` |                                |
| 3: SMEM Caching                     |   `391.2` |                                |
| 4: 1D Blocktiling                   |   `270.0` |                                |
| 5: 2D Blocktiling                   |   `206.3` |                                |
| 0: cuBLAS                           |  `1837.3` | 100.0%                         |



## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.


## Sparse implementation

We are interested in sparse matrices instead.
Interesting densities (1-sparsity) are: 0.5, 0.25, 0.12, 0.06
 - N=1, K=4096, M=4096, fp32 
 - N=1, K=4096, M=4096, fp16 
 - N=1, K=4096, M=4096, int8
 - N=1, K=4096, M=4096, int4




## TODO list

Create checkboxes

- [x] problem object, basic object
  - [x] C++ data
  - [x] cuda data
  - [x] processing
  - [x] C++ pocessed data
  - [x] C++ cuda data
- [ ] gen sparse representation
- [X] run cubas for reference
- [X] run CSR representation for reference
- [ ] run algo



