#include <cuda_runtime.h>
#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    printf("Hello from CPU!\n");
    helloFromGPU<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
