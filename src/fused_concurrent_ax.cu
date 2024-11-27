#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <wmma_extension/operators.hpp>
#include <cooperative_groups.h>

#include "impl.h"


namespace cg = cooperative_groups;

namespace fused_concurrent_ax {

using ElementInput = half;
using ElementOutput = half;
using ElementCompute = half;

using namespace nvcuda::wmma;

using LayoutA = col_major;
using LayoutB = col_major;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void fused_concurrent_ax_kernel(
    const ElementInput* __restrict__ W,  // [m x n]
    const ElementInput* __restrict__ x,  // [n x b]
    const ElementInput* __restrict__ B,  // [m x r]
    const ElementInput* __restrict__ A,  // [r x n]
    ElementOutput* __restrict__ Y,       // [m x b]
    int m, int n, int b, int r,
    half *shared_tmp,
    const int LORA_BLOCKS_R,
    const int LORA_BLOCKS_B) {

    using namespace nvcuda::wmma;

    cg::grid_group grid = cg::this_grid();

    int row_start = (blockIdx.x / (b/WMMA_N)) * WMMA_M;
    int col_start = (blockIdx.x % (b/WMMA_N)) * WMMA_N;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Wx_C;

    int lora_block_start = gridDim.x - LORA_BLOCKS_R*LORA_BLOCKS_B;

    // compute Wx
    if (blockIdx.x < lora_block_start) {
        fragment<matrix_a, WMMA_M, WMMA_K, WMMA_K, ElementInput, LayoutA> frag_Wx_A;
        fragment<matrix_b, WMMA_K, WMMA_N, WMMA_K, ElementInput, LayoutB> frag_Wx_B;
        fill_fragment(frag_Wx_C, 0.0f);
        for(int k = 0; k < n; k += WMMA_K){
            // compute v = Wx
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
    } else {
        int splits_b = max(16, b / LORA_BLOCKS_B);
        int start_b = ((blockIdx.x - lora_block_start) / LORA_BLOCKS_R) * splits_b;
        int end_b = start_b + splits_b;
        int stop_b = min(b, end_b);

        int splits_r = max(16, r / LORA_BLOCKS_R);
        int start_r = ((blockIdx.x - lora_block_start) % LORA_BLOCKS_R) * splits_r;
        int end_r = start_r + splits_r;
        int stop_r = min(r, end_r);

        for (int col = start_b; col < stop_b; col+=WMMA_N) {
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
                    int x_col = col;

                    const ElementInput* A_tile_ptr = A + a_col * r + a_row;
                    const ElementInput* x_tile_ptr = x + x_col * n + x_row;

                    load_matrix_sync(frag_Ax_A, A_tile_ptr, r);
                    load_matrix_sync(frag_Ax_B, x_tile_ptr, n);

                    mma_sync(frag_Ax_C, frag_Ax_A, frag_Ax_B, frag_Ax_C);
                }
                store_matrix_sync(shared_tmp + col * r + row, frag_Ax_C, r, mem_col_major);
            }
        }
    }

    grid.sync();

    if (blockIdx.x < lora_block_start) {
        //load from rxb u intermediate
        //and do Y = v + Bu
        //for output Y mxb, m = row_start, b = col_start
        //B is mxr
        //so for (m, b) output tile, we need row m of B and column b of u
        fragment<matrix_a, WMMA_M, WMMA_K, WMMA_K, ElementInput, LayoutA> frag_Bu_A;
        fragment<matrix_b, WMMA_K, WMMA_N, WMMA_K, ElementInput, LayoutB> frag_Bu_B;
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ElementCompute> frag_Bu_C;
        fill_fragment(frag_Bu_C, 0.0f);
        for(int k = 0; k < r; k += WMMA_K){
            int b_row = row_start;
            int b_col = k;
            int u_row = k;
            int u_col = col_start;

            const ElementInput* B_tile_ptr = B + b_col * m + b_row;
            const half* u_tile_ptr = shared_tmp + u_col * r + u_row;

            load_matrix_sync(frag_Bu_A, B_tile_ptr, m);
            load_matrix_sync(frag_Bu_B, u_tile_ptr, r);

            mma_sync(frag_Bu_C, frag_Bu_A, frag_Bu_B, frag_Bu_C);
        }

        // accumulate
        frag_Wx_C = frag_Wx_C + frag_Bu_C;

        // Store the result to Y manually, converting from float to half
        ElementOutput* Y_tile_ptr = Y + col_start * m + row_start;
        store_matrix_sync(Y_tile_ptr, frag_Wx_C, m, mem_col_major);
    }
}

void launch_fused_concurrent_ax(
    const __half* d_W, const __half* d_x,
    const __half* d_B, const __half* d_A,
    __half* d_Y, const Dimensions& dims, __half* d_tmp) {

    int m = dims.size_m;
    int n = dims.size_d;
    int b = dims.size_b;
    int r = dims.size_r;

    int threads_per_block = 32;
    int LORA_BLOCKS_R = (r + 15) / 16;
    int LORA_BLOCKS_B = (b + 15) / 16;
    dim3 gridSize(m / WMMA_M * b / WMMA_N + LORA_BLOCKS_R*LORA_BLOCKS_B);

    void* kernelArgs[] = {
        &d_W,
        &d_x,
        &d_B,
        &d_A,
        &d_Y,
        &m, &n, &b, &r,
        &d_tmp,
        &LORA_BLOCKS_R,
        &LORA_BLOCKS_B
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)fused_concurrent_ax_kernel,
        gridSize,
        threads_per_block,
        kernelArgs
    );

    // Check for errors
    if(err != cudaSuccess){
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

}

} // namespace fused_concurrent_ax
