#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "utils.h"

// CUDA kernel to add Wx and BAx (FP16 support)
__global__ void add_matrices_fp16(const __half* Wx, const __half* BAx, __half* y, int size_m, int size_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_m * size_b) {
        y[idx] = __hadd(Wx[idx], BAx[idx]);
    }
}
