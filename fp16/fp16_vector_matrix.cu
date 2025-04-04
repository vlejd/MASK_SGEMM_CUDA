#include <cuda_fp16.h>
#include <iostream>

__global__ void add_half_arrays(const __half* a, const __half* b, __half* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = __hadd(a[idx], b[idx]);  // Element-wise addition for half-precision
    }
}

int main() {
    int size = 1024;  // Number of elements
    int bytes = size * sizeof(__half);  // Memory size

    // Allocate host memory
    __half *h_a, *h_b, *h_c;
    h_a = new __half[size];
    h_b = new __half[size];
    h_c = new __half[size];

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = __float2half(i);  // Convert float to half
        h_b[i] = __float2half(i * 2);  // Convert float to half
    }

    // Allocate device memory
    __half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch Kernel (using 256 threads per block)
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_half_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy result back to CPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print some results
    for(int i=0; i<10; i++){
        std::cout << "h_c[" << i << "] = " << __half2float(h_c[i]) << std::endl;
    }

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}