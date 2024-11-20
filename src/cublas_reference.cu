#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "utils.h"
#include "impl.h"

// Namespace for cuBLAS reference implementation
namespace cublas_reference {

// Function that performs the LoRA forward pass using cuBLAS
void launch_cublas_reference(const __half* d_W, const __half* d_x, const __half* d_B, const __half* d_A,
                             __half* d_y, const Dimensions& dims) {
    // Create cuBLAS handle locally
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Device pointers for intermediate results
    __half* d_Wx;
    __half* d_Ax;
    __half* d_BAx;

    size_t size_Wx = dims.size_m * dims.size_b * sizeof(__half);
    size_t size_Ax = dims.size_r * dims.size_b * sizeof(__half);
    size_t size_BAx = dims.size_m * dims.size_b * sizeof(__half);

    cudaMalloc(&d_Wx, size_Wx);
    cudaMalloc(&d_Ax, size_Ax);
    cudaMalloc(&d_BAx, size_BAx);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Step 1: Compute Wx = W * x (size_m x size_b)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        dims.size_m, dims.size_b, dims.size_d,
        &alpha, d_W, CUDA_R_16F, dims.size_m, d_x, CUDA_R_16F, dims.size_d,
        &beta, d_Wx, CUDA_R_16F, dims.size_m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Step 2: Compute Ax = A * x (size_r x size_b)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        dims.size_r, dims.size_b, dims.size_d,
        &alpha, d_A, CUDA_R_16F, dims.size_r, d_x, CUDA_R_16F, dims.size_d,
        &beta, d_Ax, CUDA_R_16F, dims.size_r,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Step 3: Compute BAx = B * Ax (size_m x size_b)
    cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        dims.size_m, dims.size_b, dims.size_r,
        &alpha, d_B, CUDA_R_16F, dims.size_m, d_Ax, CUDA_R_16F, dims.size_r,
        &beta, d_BAx, CUDA_R_16F, dims.size_m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Step 4: Add Wx and BAx (y = Wx + BAx)
    int threadsPerBlock = 256;
    int blocksPerGrid = (dims.size_m * dims.size_b + threadsPerBlock - 1) / threadsPerBlock;
    add_matrices_fp16<<<blocksPerGrid, threadsPerBlock>>>(d_Wx, d_BAx, d_y, dims.size_m, dims.size_b);

    // Free intermediate results
    cudaFree(d_Wx);
    cudaFree(d_Ax);
    cudaFree(d_BAx);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

}  // namespace cublas_reference
