#pragma once

#include "utils.h"

namespace cublas_reference {
    void launch_cublas_reference(
        const __half* d_W,
        const __half* d_x,
        const __half* d_B,
        const __half* d_A,
        __half* d_y,
        const Dimensions& dims
    );
}

namespace fused_sequential {
    void launch_fused_sequential(
        const __half* d_W,
        const __half* d_x,
        const __half* d_B,
        const __half* d_A,
        __half* d_y,
        const Dimensions& dims
    );
}

//////////////////////////////////////////
// TODO ADD IMPLEMENTATION STRUCTS HERE //
//////////////////////////////////////////
// Struct for cuBLAS reference implementation
struct CublasReference {
    constexpr static char const* name = "cublas_reference";

    static void run(const __half* d_W, const __half* d_x, const __half* d_B, const __half* d_A,
                    __half* d_y, const Dimensions& dims, void* workspace = nullptr) {
        cublas_reference::launch_cublas_reference(d_W, d_x, d_B, d_A, d_y, dims);
    }
};

struct FusedSequential {
    constexpr static char const* name = "fused_sequential";

    static void run(const __half* d_W, const __half* d_x, const __half* d_B, const __half* d_A,
                    __half* d_y, const Dimensions& dims, void* workspace = nullptr) {
        fused_sequential::launch_fused_sequential(d_W, d_x, d_B, d_A, d_y, dims);
    }
};
