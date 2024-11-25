#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>  // For nvcuda::wmma
#include <wmma_extension/operators.hpp>

#include "impl.h"  // Ensure this includes the correct Dimensions struct

namespace fused_concurrent {

using ElementInput = half;          // 'half' is equivalent to '__half'
using ElementOutput = half;
using ElementCompute = half;

// Use WMMA namespace
using namespace nvcuda::wmma;

// Define the matrix layouts as type aliases
using LayoutA = col_major;
using LayoutB = col_major;

// Define the tile sizes (must be multiples of 16 for Tensor Cores)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Kernel using WMMA Tensor Cores
__global__ void fused_concurrent_kernel(
    const ElementInput* __restrict__ W,  // [m x n]
    const ElementInput* __restrict__ x,  // [n x b]
    const ElementInput* __restrict__ B,  // [m x r]
    const ElementInput* __restrict__ A,  // [r x n]
    ElementOutput* __restrict__ Y,       // [m x b]
    int m, int n, int b, int r) {

    // Using WMMA namespace inside the kernel
    using namespace nvcuda::wmma;

    // Coordinates for the output tile
    int row_start = blockIdx.x * WMMA_M;//tile_row * WMMA_M;
    int col_start = blockIdx.y * WMMA_N;//tile_col * WMMA_N;

    // Allocate shared memory
    extern __shared__ char shared_mem[];
    half* shared_u = reinterpret_cast<half*>(shared_mem);
    half* shared_tmp = reinterpret_cast<half*>(shared_mem + r * WMMA_N * sizeof(half));

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Wx_C;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Bu_C;

    // compute Wx
    if (threadIdx.x < 32) {
        fragment<matrix_a, WMMA_M, WMMA_K, WMMA_K, ElementInput, LayoutA> frag_Wx_A;
        fragment<matrix_b, WMMA_K, WMMA_N, WMMA_K, ElementInput, LayoutB> frag_Wx_B;
        fill_fragment(frag_Wx_C, 0.0f);
        for(int k = 0; k < n; k += WMMA_K){
            // compute v = Wx
            // inside this loop to reuse more of x
            int w_row = row_start;
            int w_col = k;
            int x_row = k;
            int x_col = col_start;

            const ElementInput* W_tile_ptr = W + w_col * m + w_row;
            const ElementInput* x_tile_ptr = x + x_col * n + x_row;

            load_matrix_sync(frag_Wx_A, W_tile_ptr, m);
            load_matrix_sync(frag_Wx_B, x_tile_ptr, n);

            mma_sync(frag_Wx_C, frag_Wx_A, frag_Wx_B, frag_Wx_C);
        }
        //store_matrix_sync(shared_tmp, frag_Wx_C, WMMA_M, mem_col_major);
    } else {
        // Fragments for A and x
        fragment<matrix_a, WMMA_M, WMMA_K, WMMA_K, ElementInput, LayoutA> frag_Ax_A;
        fragment<matrix_b, WMMA_K, WMMA_N, WMMA_K, ElementInput, LayoutB> frag_Ax_B;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Ax_C;
        // Compute u = Ax
        for(int row = 0; row < r; row += WMMA_M){
            fill_fragment(frag_Ax_C, 0.0f);
            for(int k = 0; k < n; k += WMMA_K){
                int a_row = row;
                int a_col = k;
                int x_row = k;
                int x_col = col_start;

                const ElementInput* A_tile_ptr = A + a_col * r + a_row;
                const ElementInput* x_tile_ptr = x + x_col * n + x_row;

                load_matrix_sync(frag_Ax_A, A_tile_ptr, r);
                load_matrix_sync(frag_Ax_B, x_tile_ptr, n);

                mma_sync(frag_Ax_C, frag_Ax_A, frag_Ax_B, frag_Ax_C);
            }
            store_matrix_sync(shared_u + row, frag_Ax_C, r, mem_col_major);
        }

        // Compute v = Bu
        fragment<matrix_a, WMMA_M, WMMA_K, WMMA_K, ElementInput, LayoutA> frag_Bu_A;
        fragment<matrix_b, WMMA_K, WMMA_N, WMMA_K, ElementInput, LayoutB> frag_Bu_B;
        fill_fragment(frag_Bu_C, 0.0f);

        for(int k = 0; k < r; k += WMMA_K){
            int b_row = row_start;
            int b_col = k;
            int u_row = k;
            int u_col = 0;

            const ElementInput* B_tile_ptr = B + b_col * m + b_row;
            const half* u_tile_ptr = shared_u + u_col * r + u_row;

            load_matrix_sync(frag_Bu_A, B_tile_ptr, m);
            load_matrix_sync(frag_Bu_B, u_tile_ptr, r);

            mma_sync(frag_Bu_C, frag_Bu_A, frag_Bu_B, frag_Bu_C);
        }

        //TODO write frag_Bu_C to shmem
        store_matrix_sync(shared_tmp, frag_Bu_C, WMMA_M, mem_col_major);
    }

    __syncthreads();

    if (threadIdx.x < 32) {
    //if (threadIdx.x >= 32) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Bu_C;
        load_matrix_sync(frag_Bu_C, shared_tmp, WMMA_M, mem_col_major);
        //fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Wx_C;
        //load_matrix_sync(frag_Wx_C, shared_tmp, WMMA_M, mem_col_major);

        // accumulate
        frag_Wx_C = frag_Wx_C + frag_Bu_C;

        // Store the result to Y manually, converting from float to half
        ElementOutput* Y_tile_ptr = Y + col_start * m + row_start;
        store_matrix_sync(Y_tile_ptr, frag_Wx_C, m, mem_col_major);
    }
}

void launch_fused_concurrent(
    const __half* d_W, const __half* d_x,
    const __half* d_B, const __half* d_A,
    __half* d_Y, const Dimensions& dims) {

    int m = dims.size_m;
    int n = dims.size_d;
    int b = dims.size_b;
    int r = dims.size_r;

    int threads_per_block = 2*32;
    dim3 gridSize(m / WMMA_M, b / WMMA_N);
    size_t shared_mem_size = r * WMMA_N * sizeof(half) + WMMA_M * WMMA_N * sizeof(half);

    // Check shared memory size
    int max_shared_mem_per_block;
    cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    if(shared_mem_size > static_cast<size_t>(max_shared_mem_per_block)){
        std::cerr << "Shared memory per block exceeds the maximum limit." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Launch the kernel
    fused_concurrent_kernel<<<gridSize, threads_per_block, shared_mem_size>>>(
        d_W,
        d_x,
        d_B,
        d_A,
        d_Y,
        m, n, b, r);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // Synchronize to catch errors
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

} // namespace fused_concurrent
