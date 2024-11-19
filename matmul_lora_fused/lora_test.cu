#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>  // For std::min

#define EPSILON 1e-2  // Adjusted for FP16 precision

// CUDA kernel to add Wx and BAx (FP16 support)
__global__ void add_matrices_fp16(const __half* Wx, const __half* BAx, __half* y, int size_m, int size_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_m * size_b) {
        y[idx] = __hadd(Wx[idx], BAx[idx]);
    }
}

// Function to initialize an FP32 array with random values and convert to FP16
void initialize_random_fp16(float* array_fp32, __half* array_fp16, int size, float min_value, float max_value) {
    for (int i = 0; i < size; ++i) {
        array_fp32[i] = min_value + static_cast<float>(rand()) / RAND_MAX * (max_value - min_value);
        array_fp16[i] = __float2half(array_fp32[i]);
    }
}

// Reference implementation using cuBLAS with FP16
void lora_reference_cublas_fp16(
    const __half* d_W, const __half* d_x, const __half* d_B, const __half* d_A,
    __half* d_y, int size_m, int size_d, int size_b, int size_r, cublasHandle_t handle) {

    // Device pointers for intermediate results
    __half* d_Wx;
    __half* d_Ax;
    __half* d_BAx;

    size_t size_Wx = size_m * size_b * sizeof(__half);
    size_t size_Ax = size_r * size_b * sizeof(__half);
    size_t size_BAx = size_m * size_b * sizeof(__half);

    cudaMalloc(&d_Wx, size_Wx);
    cudaMalloc(&d_Ax, size_Ax);
    cudaMalloc(&d_BAx, size_BAx);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Step 1: Compute Wx = W * x (size_m x size_b)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        size_m, size_b, size_d,
        &alpha, d_W, CUDA_R_16F, size_m, d_x, CUDA_R_16F, size_d,
        &beta, d_Wx, CUDA_R_16F, size_m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Step 2: Compute Ax = A * x (size_r x size_b)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        size_r, size_b, size_d,
        &alpha, d_A, CUDA_R_16F, size_r, d_x, CUDA_R_16F, size_d,
        &beta, d_Ax, CUDA_R_16F, size_r,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Step 3: Compute BAx = B * Ax (size_m x size_b)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        size_m, size_b, size_r,
        &alpha, d_B, CUDA_R_16F, size_m, d_Ax, CUDA_R_16F, size_r,
        &beta, d_BAx, CUDA_R_16F, size_m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Step 4: Add Wx and BAx (y = Wx + BAx)
    int threadsPerBlock = 256;
    int blocksPerGrid = (size_m * size_b + threadsPerBlock - 1) / threadsPerBlock;
    add_matrices_fp16<<<blocksPerGrid, threadsPerBlock>>>(d_Wx, d_BAx, d_y, size_m, size_b);

    // Free intermediate results
    cudaFree(d_Wx);
    cudaFree(d_Ax);
    cudaFree(d_BAx);
}

int main() {
    // Seed for reproducible random numbers
    srand(42);

    // Sizes (example values)
    int size_m = 1024; // Rows of W and B
    int size_d = 512;  // Columns of W and A
    int size_b = 256;  // Columns of x
    int size_r = 128;  // Rank (columns of B, rows of A)

    // Allocate host memory
    size_t size_W = size_m * size_d * sizeof(__half);
    size_t size_B = size_m * size_r * sizeof(__half);
    size_t size_A = size_r * size_d * sizeof(__half);
    size_t size_x = size_d * size_b * sizeof(__half);
    size_t size_y = size_m * size_b * sizeof(__half);

    float *h_W_fp32 = (float*)malloc(size_m * size_d * sizeof(float));
    float *h_B_fp32 = (float*)malloc(size_m * size_r * sizeof(float));
    float *h_A_fp32 = (float*)malloc(size_r * size_d * sizeof(float));
    float *h_x_fp32 = (float*)malloc(size_d * size_b * sizeof(float));

    __half *h_W = (__half*)malloc(size_W);
    __half *h_B = (__half*)malloc(size_B);
    __half *h_A = (__half*)malloc(size_A);
    __half *h_x = (__half*)malloc(size_x);
    __half *h_y_ref = (__half*)malloc(size_y);
    __half *h_y_user = (__half*)malloc(size_y);

    // Initialize host arrays with random FP32 values and convert to FP16
    initialize_random_fp16(h_W_fp32, h_W, size_m * size_d, -1.0f, 1.0f);
    initialize_random_fp16(h_B_fp32, h_B, size_m * size_r, -1.0f, 1.0f);
    initialize_random_fp16(h_A_fp32, h_A, size_r * size_d, -1.0f, 1.0f);
    initialize_random_fp16(h_x_fp32, h_x, size_d * size_b, -1.0f, 1.0f);

    // Allocate device memory
    __half *d_W, *d_B, *d_A, *d_x, *d_y_ref, *d_y_user;
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_y_ref, size_y);
    cudaMalloc(&d_y_user, size_y);

    // Copy data from host to device
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warmup and timing
    int trials = 5;
    double min_time_ms = 1e9; // Initialize with a large value

    for (int i = 0; i < trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        lora_reference_cublas_fp16(d_W, d_x, d_B, d_A, d_y_ref, size_m, size_d, size_b, size_r, handle);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start; // Convert to milliseconds

        if (i > 0) { // Ignore the first trial for warmup
            min_time_ms = std::min(min_time_ms, elapsed.count());
        }
    }

    std::cout << "Minimum execution time after warmup: " << min_time_ms << " ms." << std::endl;

    // Clean up
    cudaFree(d_W);
    cudaFree(d_B);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y_ref);
    cudaFree(d_y_user);
    cublasDestroy(handle);
    free(h_W_fp32);
    free(h_B_fp32);
    free(h_A_fp32);
    free(h_x_fp32);
    free(h_W);
    free(h_B);
    free(h_A);
    free(h_x);
    free(h_y_ref);
    free(h_y_user);

    return 0;
}
