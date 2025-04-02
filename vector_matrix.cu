#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheckInternal(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";
const std::string dbgLogFile = "matrixValidationDebug.txt";

void log_matrix_data(const std::string &fileName, const Problem_InstanceFP32 &pi) {
  std::ofstream fs;
  fs.open(fileName);
  fs << "A:\n";
  print_matrix(pi.hA, pi.M, pi.N, fs);
  fs << "B:\n";
  print_matrix(pi.hB, pi.M, pi.N, fs);
  fs << "Mask:\n";
  print_matrix(pi.hMask, pi.M, pi.N, fs);
  fs << "C:\n";
  print_matrix(pi.hC, pi.M, pi.N, fs);
  fs << "Should:\n";
  print_matrix(pi.hC_ref, pi.M, pi.N, fs);
};


int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  bool normal_kernel = (kernel_num >= 0 && kernel_num <= 12);
  bool vector_kernel = (kernel_num >= 101);
  if (!(normal_kernel || vector_kernel)) {
    std::cerr << "Please enter a valid kernel number (0-12) or (101-?)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  //std::vector<int> SIZE = {1<<10, 1<<11, 1<<12, 1<<13};
  std::vector<int> SIZE = {1<<12};

  // GEMM input parameters, C=α*AB+β*C
  float alpha = 1, beta=0; 

  float density = 0.25;
  int repeat_times = 50;
  for (int size : SIZE) {
    //problem instance
    Problem_InstanceFP32 pi(1, size, size, density, 42);

    std::cout << "dimensions(m,n,k) " << pi.M << "," << pi.K << "," << pi.N << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (true) {  // kernel_num != 0
      // run_kernel with problem_instance
      run_kernel(0, pi.M, pi.N, pi.K, alpha, pi.dA, pi.dB, beta, pi.dC_ref, handle);
      if(kernel_num < 100){
        run_kernel(kernel_num, pi.M, pi.N, pi.K, alpha, pi.dA, pi.dB, beta, pi.dC, handle);
      } else {
        run_vector_kernel(kernel_num, pi, alpha, beta);
      }

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      pi.get_result();
      pi.get_result_ref();

      if (!verify_matrix(pi.hC_ref, pi.hC, pi.M * pi.N)) {
        std::cout
            << "@@@@ Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;

        if (pi.M <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          log_matrix_data(errLogFile, pi);
        }
        // exit(EXIT_FAILURE);
      } else {
        if (pi.M <= 128 && true) {
          std::cout << " Logging debug output into " << dbgLogFile << "\n";
          log_matrix_data(dbgLogFile, pi);
        }
      }
    }

    cudaEventRecord(beg);
    if(kernel_num < 100){
      for (int j = 0; j < repeat_times; j++) {
        run_kernel(kernel_num, pi.M, pi.N, pi.K, alpha, pi.dA, pi.dB, beta, 
                  pi.dC, handle);
      }
    } else {
      for (int j = 0; j < repeat_times; j++) {
        run_vector_kernel(kernel_num, pi, alpha, beta);
      }
    }
    
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * pi.M * pi.N * pi.K;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ldX%ldX%ld).\n\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, pi.M, pi.K, pi.N);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    // cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
  }

  cublasDestroy(handle);

  return 0;
};
