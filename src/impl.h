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

namespace fused_concurrent {
    void launch_fused_concurrent(
        const __half* d_W,
        const __half* d_x,
        const __half* d_B,
        const __half* d_A,
        __half* d_y,
        const Dimensions& dims
    );
}

namespace fused_concurrent_asymmetric {
    void launch_fused_concurrent_asymmetric(
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

struct FusedConcurrent {
    constexpr static char const* name = "fused_concurrent";

    static void run(const __half* d_W, const __half* d_x, const __half* d_B, const __half* d_A,
                    __half* d_y, const Dimensions& dims, void* workspace = nullptr) {
        fused_concurrent::launch_fused_concurrent(d_W, d_x, d_B, d_A, d_y, dims);
    }
};

struct FusedConcurrentAsymmetric {
    constexpr static char const* name = "fused_concurrent_asymmetric";

    static void run(const __half* d_W, const __half* d_x, const __half* d_B, const __half* d_A,
                    __half* d_y, const Dimensions& dims, void* workspace = nullptr) {
        fused_concurrent_asymmetric::launch_fused_concurrent_asymmetric(d_W, d_x, d_B, d_A, d_y, dims);
    }
};
