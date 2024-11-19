#include <iostream>
#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

// Define the data type
using Element = cutlass::half_t;

// -------------------------
// 3. Kernels for u = A x and Y = B u
// -------------------------

// Define the CUTLASS GEMM template for u = A x
using GemmAx = cutlass::gemm::device::Gemm<
    Element,                          // Data-type of A matrix
    cutlass::layout::RowMajor,        // Layout of A matrix
    Element,                          // Data-type of B matrix
    cutlass::layout::RowMajor,        // Layout of B matrix
    Element,                          // Data-type of C matrix
    cutlass::layout::RowMajor         // Layout of C matrix
>;

// Define the CUTLASS GEMM template for Y = B u
using GemmBu = cutlass::gemm::device::Gemm<
    Element,                          // Data-type of A matrix
    cutlass::layout::RowMajor,        // Layout of A matrix
    Element,                          // Data-type of B matrix
    cutlass::layout::RowMajor,        // Layout of B matrix
    Element,                          // Data-type of C matrix
    cutlass::layout::RowMajor         // Layout of C matrix
>;

void compute_Ax(
    const Element* A,     // [r x n]
    const Element* x,     // [n x b]
    Element* u,           // [r x b]
    int r, int n, int b) {

    // Create a CUTLASS GEMM operator for u = A x
    GemmAx gemm_operator;

    // Define the GEMM problem size
    cutlass::gemm::GemmCoord problem_size(r, b, n);

    // Define the GEMM arguments
    GemmAx::Arguments arguments{
        problem_size,
        {A, n},       // TensorRef for A
        {x, b},       // TensorRef for x
        {u, b},       // TensorRef for u (destination)
        {u, b},       // TensorRef for u (source for accumulation)
        {1.0f, 0.0f}  // Alpha and Beta
    };

    // Launch the GEMM computation
    cutlass::Status status = gemm_operator(arguments);

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM Ax failed: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void compute_Bu(
    const Element* B,     // [m x r]
    const Element* u,     // [r x b]
    Element* Y,           // [m x b]
    int m, int r, int b) {

    // Create a CUTLASS GEMM operator for Y = B u
    GemmBu gemm_operator;

    // Define the GEMM problem size
    cutlass::gemm::GemmCoord problem_size(m, b, r);

    // Define the GEMM arguments
    GemmBu::Arguments arguments{
        problem_size,
        {B, r},       // TensorRef for B
        {u, b},       // TensorRef for u
        {Y, b},       // TensorRef for Y (destination)
        {Y, b},       // TensorRef for Y (source for accumulation)
        {1.0f, 0.0f}  // Alpha and Beta
    };

    // Launch the GEMM computation
    cutlass::Status status = gemm_operator(arguments);

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM Bu failed: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// -------------------------
// 4. Kernel for Y = B (A x)
// -------------------------

void compute_BAx(
    const Element* B,     // [m x r]
    const Element* A,     // [r x n]
    const Element* x,     // [n x b]
    Element* Y,           // [m x b]
    int m, int n, int b, int r) {

    // Allocate intermediate result u = A x
    Element* u;
    cudaMalloc((void**)&u, sizeof(Element) * r * b);

    // Compute u = A x
    compute_Ax(A, x, u, r, n, b);

    // Compute Y = B u
    compute_Bu(B, u, Y, m, r, b);

    // Free the intermediate result
    cudaFree(u);
}

// -------------------------
// 5. Main Function for Benchmarking
// -------------------------

int main() {
    // Matrix dimensions (example values; adjust as needed)
    int m = 1024;  // Number of rows in W and Y
    int n = 512;   // Number of columns in W and rows in x
    int b = 256;   // Number of columns in x and Y
    int r = 128;   // Number of columns in A and rows in B

    // Allocate host memory
    size_t size_W = m * n * sizeof(Element);
    size_t size_x = n * b * sizeof(Element);
    size_t size_Y = m * b * sizeof(Element);
    size_t size_B = m * r * sizeof(Element);
    size_t size_A = r * n * sizeof(Element);

    Element* h_W = (Element*)malloc(size_W);
    Element* h_x = (Element*)malloc(size_x);
    Element* h_Y = (Element*)malloc(size_Y);
    Element* h_B = (Element*)malloc(size_B);
    Element* h_A = (Element*)malloc(size_A);

    // Initialize host matrices (omitted for brevity)
    // ...

    // Allocate device memory
    Element *d_W, *d_x, *d_Y, *d_B, *d_A;
    cudaMalloc((void**)&d_W, size_W);
    cudaMalloc((void**)&d_x, size_x);
    cudaMalloc((void**)&d_Y, size_Y);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_A, size_A);

    // Copy data from host to device
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    // -------------------------
    // Benchmarking the Fused Kernel
    // -------------------------

    // Zero out d_Y
    cudaMemset(d_Y, 0, size_Y);

    // Launch the fused kernel
    launch_fused_kernel(d_W, d_x, d_B, d_A, d_Y, m, n, b, r);

    // Copy result back to host (if needed)
    // cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // -------------------------
    // Benchmarking the Individual Kernels
    // -------------------------

    // Zero out d_Y
    cudaMemset(d_Y, 0, size_Y);

    // Compute Y = W x
    compute_Wx(d_W, d_x, d_Y, m, n, b);

    // Compute u = A x
    Element* d_u;
    cudaMalloc((void**)&d_u, sizeof(Element) * r * b);
    compute_Ax(d_A, d_x, d_u, r, n, b);

    // Compute Y += B u
    compute_Bu(d_B, d_u, d_Y, m, r, b);

    // Free intermediate result
    cudaFree(d_u);

    // Copy result back to host (if needed)
    // cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // -------------------------
    // Clean Up
    // -------------------------

    // Free device memory
    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_Y);
    cudaFree(d_B);
    cudaFree(d_A);

    // Free host memory
    free(h_W);
    free(h_x);
    free(h_Y);
    free(h_B);
    free(h_A);

    return 0;
}
