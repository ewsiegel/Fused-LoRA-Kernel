#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <cutlass/gemm/device/gemm.h>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)


namespace matmul_cutlass {

void launch_matmul_cutlass(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    __half const *a,
    __half const *b,
    __half *c,
    void *workspace) {
    // Define data types for the computation.
    using ElementInputA = __half;
    using ElementInputB = __half;
    using ElementOutput = __half;
    using ElementAccumulator = __half;
    using ElementCompute = __half;

    // Define the layouts of the matrices (row-major in this case).
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // Define the GEMM operation.
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator>;

    // Create GEMM arguments.
    ElementCompute alpha = 1.0f;
    ElementCompute beta = 0.0f;
    typename Gemm::Arguments args(
        {size_i, size_j, size_k},
        {a, size_k},     // Tensor A (device pointer and leading dimension)
        {b, size_j},     // Tensor B (device pointer and leading dimension)
        {c, size_j},     // Tensor C (device pointer and leading dimension)
        {c, size_j},     // Tensor D (output tensor)
        {alpha, beta}      // Scalars used in the epilogue
    );
    Gemm gemm_op;

    // Check and init GEMM.
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Unsupported operation\n";
        return;
    }

    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Failed to init GEMM\n";
        return;
    }

    // Launch GEMM.
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cout << "Failed to run GEMM\n";
        return;
    }
}

}; // namespace matmul_cutlass

int main(int argc, char **argv) {
    printf("nothing to do\n");
}
