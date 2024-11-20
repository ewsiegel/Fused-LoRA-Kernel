//#include <iostream>
//#include <cuda_runtime.h>
//#include <mma.h>  // For nvcuda::wmma
//
//// Include CUTLASS headers
//#include <cutlass/cutlass.h>
//#include <cutlass/layout/matrix.h>
//
//#include "impl.h"  // Make sure this includes the correct Dimensions struct
//
//namespace fused_lora_sequential {
//
//using ElementInput = half;          // half is equivalent to __half
//using ElementOutput = half;
//using ElementCompute = float;
//
//// Define the matrix layouts
//constexpr wmma::layout_t LayoutA = wmma::col_major;
//constexpr wmma::layout_t LayoutB = wmma::row_major;
//
//// Define the tile sizes
//constexpr int M = 16;
//constexpr int N = 16;
//constexpr int K = 16;
//
//// Kernel using WMMA Tensor Cores
//__global__ void fused_lora_sequential_kernel(
//    const ElementInput* __restrict__ W,  // [m x n]
//    const ElementInput* __restrict__ x,  // [n x b]
//    const ElementInput* __restrict__ B,  // [m x r]
//    const ElementInput* __restrict__ A,  // [r x n]
//    ElementOutput* __restrict__ Y,       // [m x b]
//    int m, int n, int b, int r) {
//
//    using namespace nvcuda::wmma;
//
//    // Calculate global warp ID
//    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
//    int lane_id = threadIdx.x % 32;
//
//    int num_warp_m = (m + M - 1) / M;
//    int num_warp_n = (b + N - 1) / N;
//
//    int warp_m = warp_id % num_warp_m;
//    int warp_n = warp_id / num_warp_m;
//
//    if (warp_m >= num_warp_m || warp_n >= num_warp_n)
//        return;
//
//    // Pointers to the tile in global memory
//    const ElementInput* W_tile_ptr = W + warp_m * M * n;
//    const ElementInput* x_tile_ptr = x + warp_n * N;
//    const ElementInput* A_tile_ptr = A;
//    const ElementInput* B_tile_ptr = B + warp_m * M * r;
//
//    // Allocate shared memory for u
//    extern __shared__ ElementCompute shared_u[]; // Shared memory size to be specified in kernel launch
//
//    // Compute u = A x
//    fragment<matrix_a, K, N, K, ElementInput, LayoutA> frag_Ax_A;
//    fragment<matrix_b, K, N, K, ElementInput, LayoutB> frag_Ax_B;
//    fragment<accumulator, K, N, K, ElementCompute> frag_u;
//
//    fill_fragment(frag_u, 0.0f);
//
//    for (int k = 0; k < n; k += K) {
//        // Load A and x for computing u
//        load_matrix_sync(frag_Ax_A, A_tile_ptr + k, n);
//        load_matrix_sync(frag_Ax_B, x_tile_ptr + k * b, b);
//
//        mma_sync(frag_u, frag_Ax_A, frag_Ax_B, frag_u);
//    }
//
//    // Store u to shared memory
//    store_matrix_sync(&shared_u[0], frag_u, N, mem_row_major);
//
//    __syncwarp();
//
//    // Compute v = B u
//    fragment<matrix_a, M, N, K, ElementInput, LayoutA> frag_Bu_A;
//    fragment<matrix_b, M, N, K, ElementInput, LayoutB> frag_Bu_B;
//    fragment<accumulator, M, N, K, ElementCompute> frag_v;
//
//    fill_fragment(frag_v, 0.0f);
//
//    for (int k = 0; k < r; k += K) {
//        // Load B and u for computing v
//        load_matrix_sync(frag_Bu_A, B_tile_ptr + k, r);
//        load_matrix_sync(frag_Bu_B, reinterpret_cast<ElementInput*>(&shared_u[k * N]), N);
//
//        mma_sync(frag_v, frag_Bu_A, frag_Bu_B, frag_v);
//    }
//
//    // Compute W x
//    fragment<matrix_a, M, N, K, ElementInput, LayoutA> frag_Wx_A;
//    fragment<matrix_b, M, N, K, ElementInput, LayoutB> frag_Wx_B;
//    fragment<accumulator, M, N, K, ElementCompute> frag_C;
//
//    fill_fragment(frag_C, 0.0f);
//
//    for (int k = 0; k < n; k += K) {
//        // Load W and x for computing W x
//        load_matrix_sync(frag_Wx_A, W_tile_ptr + k, n);
//        load_matrix_sync(frag_Wx_B, x_tile_ptr + k * b, b);
//
//        mma_sync(frag_C, frag_Wx_A, frag_Wx_B, frag_C);
//    }
//
//    // Accumulate frag_C and frag_v
//    for (int i = 0; i < frag_C.num_elements; ++i) {
//        frag_C.x[i] += frag_v.x[i];
//    }
//
//    // Store the result to Y
//    store_matrix_sync(Y + warp_m * M * b + warp_n * N, frag_C, b, mem_row_major);
//}
//
//void launch_fused_lora_sequential(
//    const __half* d_W, const __half* d_x,
//    const __half* d_B, const __half* d_A,
//    __half* d_Y, const Dimensions& dims) {
//
//    int m = dims.size_m;
//    int n = dims.size_d;
//    int b = dims.size_b;
//    int r = dims.size_r;
//
//    // Calculate number of tiles
//    int num_tiles_m = (m + M - 1) / M;
//    int num_tiles_n = (b + N - 1) / N;
//
//    int total_warps = num_tiles_m * num_tiles_n;
//    int threads_per_warp = 32;
//    int warps_per_block = 4; // Adjust based on occupancy
//    int threads_per_block = warps_per_block * threads_per_warp;
//    int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
//
//    // Calculate shared memory size
//    size_t shared_mem_size = K * N * sizeof(ElementCompute); // For shared_u
//
//    // Launch the kernel
//    fused_lora_sequential_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
//        reinterpret_cast<const ElementInput*>(d_W),
//        reinterpret_cast<const ElementInput*>(d_x),
//        reinterpret_cast<const ElementInput*>(d_B),
//        reinterpret_cast<const ElementInput*>(d_A),
//        reinterpret_cast<ElementOutput*>(d_Y),
//        m, n, b, r);
//
//    // Check for errors
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) {
//        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
//        exit(EXIT_FAILURE);
//    }
//}
//
//} // namespace fused_lora_sequential
