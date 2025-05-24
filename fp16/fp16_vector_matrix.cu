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

#include "utils.cuh"
#include "kernels.cuh"

int main(int argc, char **argv)
{
    if (argc == 1 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0] << " <kernel_num> [debug_arg] [density_arg]\n";
        exit(EXIT_FAILURE);
    }

    int kernel_num = std::stoi(argv[1]);
    int debug_arg = 0;
    if (argc >= 3)
    {
        debug_arg = std::stoi(argv[2]);
    }

    printf("Debug argument: %d\n", debug_arg);
    bool debug = debug_arg;

    int density_arg = 25;
    if(argc >= 4)
    {
        density_arg = std::stoi(argv[3]);
    }
    float density = float(density_arg) / 100.0f;
    printf("Density: %d%%\n", density_arg);


    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL)
    {
        deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    cublasHandle_t handle;
    if (cublasCreate(&handle))
    {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // cuBLAS FLOPs ceiling is reached at 8192
    // std::vector<int> SIZE = {1<<10, 1<<11, 1<<12, 1<<13};
    // std::vector<int> SIZE = {1 << 12};
    std::vector<int> SIZE = {1 << 12};

    
    int repeat_times = 50;
    int num_problems = 5;
    if (debug){
        repeat_times = 1;
        num_problems = 1;
    }
    int M,N,K;
    // A @ B = C -> [MxK] @ [KxN] = [MxN]
    Problem_InstanceFP16 *problem_instances[num_problems];

    for (int size : SIZE)
    {
        // TODO generate multiple problems and cycle through them.
        if(debug){
            M = 1;
            K = 32;
            N = 32;
        } else {
            M = 1;
            K = size;
            N = size;
        }

        for (int i = 0; i < num_problems; i++)
        {
            Problem_InstanceFP16* pi_pointer = new Problem_InstanceFP16(M, K, N, density, 42+i);
            problem_instances[i] = pi_pointer;
        }
        Problem_InstanceFP16 &pi = *problem_instances[0];
        std::cout << "dimensions(M,K,N) " << pi.M << "," << pi.K << "," << pi.N << std::endl;
        run_kernel_fp16(0, pi, handle, true);
        run_kernel_fp16(kernel_num, pi, handle, false);

        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
        pi.get_result();
        pi.get_result_ref();

        if (!verify_result(pi))
        {
            std::cout << "=============================" << std::endl;
            std::cout << "Different result than cuBBLAS" << std::endl;
            std::cout << "=============================" << std::endl;
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            log_matrix_data(errLogFile, pi);
        }
        else
        {
            std::cout << " Logging debug output into " << dbgLogFile << "\n";
            log_matrix_data(dbgLogFile, pi);
        }

        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++)
        {
            Problem_InstanceFP16 &pi = *problem_instances[j%num_problems];
            // TODO cycle the problem instances
            run_kernel_fp16(kernel_num, pi, handle, false);
        }

        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // Convert to seconds

        long flops = 2 * pi.M * pi.N * pi.K;
        printf(
            "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%dX%dX%d).\n\n",
            elapsed_time / repeat_times,
            (repeat_times * flops * 1e-9) / elapsed_time, pi.M, pi.K, pi.N);
        fflush(stdout);
        for (int i = 0; i < num_problems; i++)
        {
            delete problem_instances[i];
        }

    }
    cublasDestroy(handle);

    return 0;
};