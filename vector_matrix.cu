#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

//object that holds the data for matrix multiplication
// it has array A, B, result C, and their cuda counterparts
// it also has the dimensions of the matrices
// it has a function to run the matrix multiplication
// use prefix h for host and d for device

class Problem_InstanceFP32{
public:
  int m, n, k;
  int seed;
  float *hA, *hB, *hC, *hC_ref;
  float *dA, *dB, *dC, *dC_ref;
  int *hMask;
  int *dMask;
  float density;


  Problem_InstanceFP32(int m, int n, int k, float density, int seed=0){
    this->m = m;
    this->n = n;
    this->k = k;
    this->density = density;
    this->seed = seed;

    hA = (float *)malloc(sizeof(float) * m * k);
    hB = (float *)malloc(sizeof(float) * k * n);
    hC = (float *)malloc(sizeof(float) * m * n);
    hC_ref = (float *)malloc(sizeof(float) * m * n);
    hMask = (int *)malloc(sizeof(int) * m * n);

    randomize_matrix(hA, m * k, seed);
    randomize_matrix(hB, k * n, seed);
    zero_init_matrix(hC, m * n);
    zero_init_matrix(hC_ref, m * n);
    generate_mask(hMask, m, n, density, seed);
    apply_mask(hB, hMask, m, n);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * m * k));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * k * n));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * m * n));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * m * n));

    cudaCheck(cudaMemcpy(dA, hA, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, hB, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, hC, sizeof(float) * m * n, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, hC_ref, sizeof(float) * m * n, cudaMemcpyHostToDevice));
  }

  void get_result(){
    cudaCheck(cudaMemcpy(hC, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
  }

  void get_result_ref(){
    cudaCheck(cudaMemcpy(hC_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
  }

  ~Problem_InstanceFP32(){
    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
};


int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
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
  //std::vector<int> SIZE = {2048, 4096, 1<<13};
  std::vector<int> SIZE = {32};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  // GEMM input parameters, C=α*AB+β*C
  float alpha = 1, beta=0; 

  int repeat_times = 50;
  for (int size : SIZE) {
    //problem instance
    Problem_InstanceFP32 pi(1, size, size, 42);

    std::cout << "dimensions(m,n,k) " << pi.m << "," << pi.k << "," << pi.n << ", alpha: " << alpha
              << ", beta: " << beta << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      // run_kernel with problem_instance
      run_kernel(0, pi.m, pi.n, pi.k, alpha, pi.dA, pi.dB, beta, pi.dC_ref, handle);
      run_kernel(kernel_num, pi.m, pi.n, pi.k, alpha, pi.dA, pi.dB, beta, pi.dC, handle);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      pi.get_result();
      pi.get_result_ref();

      if (!verify_matrix(pi.hC_ref, pi.hC, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(pi.hA, pi.m, pi.n, fs);
          fs << "B:\n";
          print_matrix(pi.hB, pi.m, pi.n, fs);
          fs << "C:\n";
          print_matrix(pi.hC, pi.m, pi.n, fs);
          fs << "Should:\n";
          print_matrix(pi.hC_ref, pi.m, pi.n, fs);
        }
        // exit(EXIT_FAILURE);
      }
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(kernel_num, pi.m, pi.n, pi.k, alpha, pi.dA, pi.dB, beta, 
                pi.dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * pi.m * pi.n * pi.k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld,%ld,%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, pi.m, pi.k, pi.n);
    fflush(stdout);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    // cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
  }

  cublasDestroy(handle);

  return 0;
};
