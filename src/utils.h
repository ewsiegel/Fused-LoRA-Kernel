#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Structure to define matrix dimensions
struct Dimensions {
    int size_m;  // Rows of W and B
    int size_d;  // Columns of W and A
    int size_b;  // Columns of x
    int size_r;  // Rank (columns of B, rows of A)
};

__global__ void add_matrices_fp16(const __half* Wx, const __half* BAx, __half* y, int size_m, int size_b);
