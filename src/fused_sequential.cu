#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>  // For nvcuda::wmma

#include "impl.h"  // Ensure this includes the correct Dimensions struct

namespace fused_sequential {

using ElementInput = half;          // 'half' is equivalent to '__half'
using ElementOutput = half;
using ElementCompute = float;

// Use WMMA namespace
using namespace nvcuda::wmma;

// Define the matrix layouts as type aliases
using LayoutA = row_major;
using LayoutB = row_major;

// Define the tile sizes (must be multiples of 16 for Tensor Cores)
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// Kernel using WMMA Tensor Cores
__global__ void fused_sequential_kernel(
    const ElementInput* __restrict__ W,  // [m x n]
    const ElementInput* __restrict__ x,  // [n x b]
    const ElementInput* __restrict__ B,  // [m x r]
    const ElementInput* __restrict__ A,  // [r x n]
    ElementOutput* __restrict__ Y,       // [m x b]
    int m, int n, int b, int r) {

    // Using WMMA namespace inside the kernel
    using namespace nvcuda::wmma;

    // Calculate global warp ID and warp ID within the block
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_id_in_block = (threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    int num_warp_m = (m + M - 1) / M;
    int num_warp_n = (b + N - 1) / N;

    int warp_m = global_warp_id % num_warp_m;
    int warp_n = global_warp_id / num_warp_m;

    if (warp_m >= num_warp_m || warp_n >= num_warp_n)
        return;

    // Pointers to the tile in global memory
    const ElementInput* W_tile_ptr = W + warp_m * M * n;
    const ElementInput* x_tile_ptr = x + warp_n * N * b;
    const ElementInput* A_tile_ptr = A;
    const ElementInput* B_tile_ptr = B + warp_m * M * r;

    // Shared memory allocation
    extern __shared__ char shared_mem[];

    // Calculate per-warp shared memory size
    size_t shared_mem_per_warp = K * N * sizeof(ElementCompute) + K * N * sizeof(ElementInput);

    // Calculate the warp's shared memory offset
    size_t warp_shared_mem_offset = warp_id_in_block * shared_mem_per_warp;

    // Offsets within the per-warp shared memory
    size_t shared_u_offset = 0;
    size_t shared_u_half_offset = shared_u_offset + K * N * sizeof(ElementCompute);

    // Pointers to shared memory for this warp
    ElementCompute* shared_u = reinterpret_cast<ElementCompute*>(shared_mem + warp_shared_mem_offset + shared_u_offset);
    ElementInput* shared_u_half = reinterpret_cast<ElementInput*>(shared_mem + warp_shared_mem_offset + shared_u_half_offset);

    // Compute u = A x
    fragment<matrix_a, M, K, K, ElementInput, LayoutA> frag_Ax_A;
    fragment<matrix_b, K, N, K, ElementInput, LayoutB> frag_Ax_B;
    fragment<accumulator, M, N, K, ElementCompute> frag_u;

    fill_fragment(frag_u, 0.0f);

    for (int k = 0; k < n; k += K) {
        // Load A and x for computing u
        load_matrix_sync(frag_Ax_A, A_tile_ptr + k, n);
        load_matrix_sync(frag_Ax_B, x_tile_ptr + k, b);

        mma_sync(frag_u, frag_Ax_A, frag_Ax_B, frag_u);
    }

    // Convert frag_u to __half and store to shared_u_half
    for (int i = 0; i < frag_u.num_elements; ++i) {
        shared_u[i] = frag_u.x[i];  // Store as float
        shared_u_half[i] = __float2half(frag_u.x[i]);  // Convert to half
    }

    __syncwarp();

    // Compute v = B u
    fragment<matrix_a, M, K, K, ElementInput, LayoutA> frag_Bu_A;
    fragment<matrix_b, K, N, K, ElementInput, LayoutB> frag_Bu_B;
    fragment<accumulator, M, N, K, ElementCompute> frag_v;

    fill_fragment(frag_v, 0.0f);

    for (int k = 0; k < r; k += K) {
        // Load B and u for computing v
        load_matrix_sync(frag_Bu_A, B_tile_ptr + k, r);
        load_matrix_sync(frag_Bu_B, shared_u_half + k, N);

        mma_sync(frag_v, frag_Bu_A, frag_Bu_B, frag_v);
    }

    // Compute W x
    fragment<matrix_a, M, K, K, ElementInput, LayoutA> frag_Wx_A;
    fragment<matrix_b, K, N, K, ElementInput, LayoutB> frag_Wx_B;
    fragment<accumulator, M, N, K, ElementCompute> frag_C;

    fill_fragment(frag_C, 0.0f);

    for (int k = 0; k < n; k += K) {
        // Load W and x for computing W x
        load_matrix_sync(frag_Wx_A, W_tile_ptr + k, n);
        load_matrix_sync(frag_Wx_B, x_tile_ptr + k, b);

        mma_sync(frag_C, frag_Wx_A, frag_Wx_B, frag_C);
    }

    // Accumulate frag_C and frag_v
    for (int i = 0; i < frag_C.num_elements; ++i) {
        frag_C.x[i] += frag_v.x[i];
    }

    // Store the result to Y
    int output_row = warp_m * M;
    int output_col = warp_n * N;
    int row_stride = b;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            int y_idx = (output_row + i) * row_stride + (output_col + j);
            if ((output_row + i) < m && (output_col + j) < b) {
                Y[y_idx] = __float2half(frag_C.x[idx]);
            }
        }
    }
}

void launch_fused_sequential(
    const __half* d_W, const __half* d_x,
    const __half* d_B, const __half* d_A,
    __half* d_Y, const Dimensions& dims) {

    int m = dims.size_m;
    int n = dims.size_d;
    int b = dims.size_b;
    int r = dims.size_r;

    // Calculate number of tiles
    int num_tiles_m = (m + M - 1) / M;
    int num_tiles_n = (b + N - 1) / N;

    int total_warps = num_tiles_m * num_tiles_n;
    int threads_per_warp = 32;
    int warps_per_block = 4; // Adjust based on occupancy and hardware capabilities
    int threads_per_block = warps_per_block * threads_per_warp;
    int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    // Calculate shared memory size per warp and per block
    size_t shared_mem_per_warp = (K * N * sizeof(ElementCompute)) + (K * N * sizeof(ElementInput));
    size_t shared_mem_size = shared_mem_per_warp * warps_per_block;

    // Launch the kernel
    fused_sequential_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_W,
        d_x,
        d_B,
        d_A,
        d_Y,
        m, n, b, r);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Synchronize to catch errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

} // namespace fused_sequential
