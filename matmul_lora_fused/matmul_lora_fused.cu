#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/wmma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/fragment.h>

template <
    typename scalar_t,      // Data type (e.g., cutlass::half_t)
    int WarpSizeM,          // Tile size M dimension per warp
    int WarpSizeN,          // Tile size N dimension per warp
    int WarpSizeK           // Tile size K dimension per warp
>
__global__ void fused_wxab_ax_bax_kernel(
    const scalar_t* __restrict__ W,  // [m x n]
    const scalar_t* __restrict__ x,  // [n x b]
    const scalar_t* __restrict__ B,  // [m x r]
    const scalar_t* __restrict__ A,  // [r x n]
    scalar_t* __restrict__ Y,        // [m x b] Output
    int m, int n, int b, int r) {

    // Compute the warp and lane indices
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Compute the tile indices this warp will compute
    int tile_m = (blockIdx.x * blockDim.x / 32 + warp_id) * WarpSizeM;
    int tile_n = blockIdx.y * WarpSizeN;

    // Guard against out-of-bounds tiles
    if (tile_m >= m || tile_n >= b)
        return;

    // Define fragment types
    using FragmentA = cutlass::Array<scalar_t, WarpSizeK * WarpSizeN / 32>;
    using FragmentB = cutlass::Array<scalar_t, WarpSizeM * WarpSizeK / 32>;
    using FragmentC = cutlass::Array<scalar_t, WarpSizeM * WarpSizeN / 32>;

    // Shared memory for operands
    __shared__ scalar_t shared_W[WarpSizeM * WarpSizeK];
    __shared__ scalar_t shared_x[WarpSizeK * WarpSizeN];
    __shared__ scalar_t shared_B[WarpSizeM * WarpSizeK];
    __shared__ scalar_t shared_A[WarpSizeK * WarpSizeN];

    // Pointers to the current tile
    const scalar_t* ptr_W = W + tile_m * n;
    const scalar_t* ptr_B = B + tile_m * r;
    const scalar_t* ptr_x = x + tile_n;
    const scalar_t* ptr_A = A;

    // Accumulators for Wx and B(Ax)
    FragmentC accum_Wx;
    FragmentC accum_BAx;

    // Initialize accumulators to zero
    cutlass::arch::fill_fragment(accum_Wx, scalar_t(0));
    cutlass::arch::fill_fragment(accum_BAx, scalar_t(0));

    // Loop over K dimension for Wx
    for (int k = 0; k < n; k += WarpSizeK) {
        // Load fragments of W and x into shared memory
        if (lane_id < WarpSizeM * WarpSizeK) {
            int row = lane_id / WarpSizeK;
            int col = lane_id % WarpSizeK;
            shared_W[lane_id] = ptr_W[row * n + k + col];
        }
        if (lane_id < WarpSizeK * WarpSizeN) {
            int row = lane_id / WarpSizeN;
            int col = lane_id % WarpSizeN;
            shared_x[lane_id] = ptr_x[(k + row) * b + tile_n + col];
        }
        __syncthreads();

        // Perform Wx = W * x
        FragmentA frag_W;
        FragmentB frag_x;
        cutlass::arch::load_matrix_sync(frag_W, shared_W, WarpSizeK);
        cutlass::arch::load_matrix_sync(frag_x, shared_x, WarpSizeN);
        cutlass::arch::mma_sync(accum_Wx, frag_W, frag_x, accum_Wx);
        __syncthreads();
    }

    // Compute u = A * x
    FragmentC accum_Ax;
    cutlass::arch::fill_fragment(accum_Ax, scalar_t(0));

    for (int k = 0; k < n; k += WarpSizeK) {
        // Load fragments of A and x into shared memory
        if (lane_id < WarpSizeK * WarpSizeN) {
            int row = lane_id / WarpSizeN;
            int col = lane_id % WarpSizeN;
            shared_A[lane_id] = ptr_A[row * n + k + col];
        }
        // Reuse shared_x from previous computation
        __syncthreads();

        // Perform Ax = A * x
        FragmentA frag_A;
        FragmentB frag_x;
        cutlass::arch::load_matrix_sync(frag_A, shared_A, WarpSizeK);
        cutlass::arch::load_matrix_sync(frag_x, shared_x, WarpSizeN);
        cutlass::arch::mma_sync(accum_Ax, frag_A, frag_x, accum_Ax);
        __syncthreads();
    }

    // Compute BAx = B * (A * x)
    for (int k = 0; k < r; k += WarpSizeK) {
        // Load fragments of B and accum_Ax into shared memory
        if (lane_id < WarpSizeM * WarpSizeK) {
            int row = lane_id / WarpSizeK;
            int col = lane_id % WarpSizeK;
            shared_B[lane_id] = ptr_B[row * r + k + col];
        }
        // Load accum_Ax into shared memory
        if (lane_id < WarpSizeK * WarpSizeN) {
            shared_A[lane_id] = accum_Ax[lane_id];
        }
        __syncthreads();

        // Perform BAx = B * (A * x)
        FragmentA frag_B;
        FragmentB frag_u;
        cutlass::arch::load_matrix_sync(frag_B, shared_B, WarpSizeK);
        cutlass::arch::load_matrix_sync(frag_u, shared_A, WarpSizeN);
        cutlass::arch::mma_sync(accum_BAx, frag_B, frag_u, accum_BAx);
        __syncthreads();
    }

    // Accumulate Wx and BAx
    for (int i = 0; i < accum_Wx.size(); ++i) {
        accum_Wx[i] = accum_Wx[i] + accum_BAx[i];
    }

    // Write the result back to global memory
    scalar_t* ptr_Y = Y + tile_m * b + tile_n;
    if (lane_id < WarpSizeM * WarpSizeN) {
        int row = lane_id / WarpSizeN;
        int col = lane_id % WarpSizeN;
        ptr_Y[row * b + col] = accum_Wx[lane_id];
    }
}

// Host code to launch the kernel
void launch_fused_kernel(
    const cutlass::half_t* W, const cutlass::half_t* x,
    const cutlass::half_t* B, const cutlass::half_t* A,
    cutlass::half_t* Y,
    int m, int n, int b, int r) {

    // Define warp-level tile sizes
    constexpr int WarpSizeM = 16;
    constexpr int WarpSizeN = 16;
    constexpr int WarpSizeK = 16;

    // Calculate grid and block dimensions
    int grid_dim_x = (m + WarpSizeM - 1) / WarpSizeM;
    int grid_dim_y = (b + WarpSizeN - 1) / WarpSizeN;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    dim3 block_dim(32);  // One warp per block

    // Launch the kernel
    fused_wxab_ax_bax_kernel<cutlass::half_t, WarpSizeM, WarpSizeN, WarpSizeK>
        <<<grid_dim, block_dim>>>(
            W, x, B, A, Y, m, n, b, r);

    // Check for errors (omitted for brevity)
}
