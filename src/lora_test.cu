#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>  // For std::min
#include <map>
#include <tuple>
#include <vector>
#include <string>

#include "utils.h"
#include "impl.h"

#define EPSILON_PER_MAC 1e-6
#define CORRECTNESS true

// Function to initialize an FP32 array with random values and convert to FP16
void initialize_random_fp16(float* array_fp32, __half* array_fp16, int size, float min_value, float max_value, unsigned int seed) {
    std::srand(seed);
    for (int i = 0; i < size; ++i) {
        array_fp32[i] = min_value + static_cast<float>(rand()) / RAND_MAX * (max_value - min_value);
        array_fp16[i] = __float2half(array_fp32[i]);
    }
}

int round16(int n) {
    return ((n + 15) / 16) * 16;
}

// Struct to store benchmark results
struct BenchmarkResults {
    char const* name;
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, __half*> outputs;
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, double> elapsed_ms;
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, double> speedup;
};

template <typename Impl>
BenchmarkResults benchmark(const std::vector<Dimensions>& dimensions_list, int trials) {
    BenchmarkResults results;
    results.name = Impl::name;

    printf("Benchmarking %s\n", Impl::name);

    for (const auto& dims : dimensions_list) {
        printf("Benchmarking %d x %d x %d x %d\n", dims.size_m, dims.size_d, dims.size_b, dims.size_r);

        // Allocate unpadded host arrays for W, B, A, x
        size_t size_W = dims.size_m * dims.size_d * sizeof(__half);
        size_t size_B = dims.size_m * dims.size_r * sizeof(__half);
        size_t size_A = dims.size_r * dims.size_d * sizeof(__half);
        size_t size_x = dims.size_d * dims.size_b * sizeof(__half);

        float *h_W_fp32 = (float*)malloc(dims.size_m * dims.size_d * sizeof(float));
        float *h_B_fp32 = (float*)malloc(dims.size_m * dims.size_r * sizeof(float));
        float *h_A_fp32 = (float*)malloc(dims.size_r * dims.size_d * sizeof(float));
        float *h_x_fp32 = (float*)malloc(dims.size_d * dims.size_b * sizeof(float));

        __half *h_W = (__half*)malloc(size_W);
        __half *h_B = (__half*)malloc(size_B);
        __half *h_A = (__half*)malloc(size_A);
        __half *h_x = (__half*)malloc(size_x);

        // Initialize random values
        unsigned int seed = 42;
        initialize_random_fp16(h_W_fp32, h_W, dims.size_m * dims.size_d, -1.0f, 1.0f, seed);
        initialize_random_fp16(h_B_fp32, h_B, dims.size_m * dims.size_r, -1.0f, 1.0f, seed);
        initialize_random_fp16(h_A_fp32, h_A, dims.size_r * dims.size_d, -1.0f, 1.0f, seed);
        initialize_random_fp16(h_x_fp32, h_x, dims.size_d * dims.size_b, -1.0f, 1.0f, seed);

        // Pad dimensions to the nearest multiple of 16
        int padded_m = round16(dims.size_m);
        int padded_d = round16(dims.size_d);
        int padded_b = round16(dims.size_b);
        int padded_r = round16(dims.size_r);

        // Calculate padded sizes
        size_t size_W_padded = padded_m * padded_d * sizeof(__half);
        size_t size_B_padded = padded_m * padded_r * sizeof(__half);
        size_t size_A_padded = padded_r * padded_d * sizeof(__half);
        size_t size_x_padded = padded_d * padded_b * sizeof(__half);
        size_t size_y_padded = padded_m * padded_b * sizeof(__half);

        // Allocate padded host arrays
        __half *h_W_padded = (__half*)calloc(padded_m * padded_d, sizeof(__half));
        __half *h_B_padded = (__half*)calloc(padded_m * padded_r, sizeof(__half));
        __half *h_A_padded = (__half*)calloc(padded_r * padded_d, sizeof(__half));
        __half *h_x_padded = (__half*)calloc(padded_d * padded_b, sizeof(__half));
        __half *h_y_padded = (__half*)calloc(padded_m * padded_b, sizeof(__half));

        // Copy and pad input matrices
        for (int i = 0; i < dims.size_m; ++i)
            for (int j = 0; j < dims.size_d; ++j)
                h_W_padded[i * padded_d + j] = h_W[i * dims.size_d + j];
        for (int i = 0; i < dims.size_m; ++i)
            for (int j = 0; j < dims.size_r; ++j)
                h_B_padded[i * padded_r + j] = h_B[i * dims.size_r + j];
        for (int i = 0; i < dims.size_r; ++i)
            for (int j = 0; j < dims.size_d; ++j)
                h_A_padded[i * padded_d + j] = h_A[i * dims.size_d + j];
        for (int i = 0; i < dims.size_d; ++i)
            for (int j = 0; j < dims.size_b; ++j)
                h_x_padded[i * padded_b + j] = h_x[i * dims.size_b + j];

        // Allocate device memory
        __half *d_W, *d_B, *d_A, *d_x, *d_y, *d_tmp;
        cudaMalloc(&d_W, size_W_padded);
        cudaMalloc(&d_B, size_B_padded);
        cudaMalloc(&d_A, size_A_padded);
        cudaMalloc(&d_x, size_x_padded);
        cudaMalloc(&d_y, size_y_padded);
        cudaMalloc(&d_tmp, Impl::get_workspace_size(padded_m, padded_d, padded_b, padded_r));

        // Copy padded matrices to device
        cudaMemcpy(d_W, h_W_padded, size_W_padded, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B_padded, size_B_padded, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, h_A_padded, size_A_padded, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x_padded, size_x_padded, cudaMemcpyHostToDevice);

        // Run kernel and measure time
        double min_time_ms = 1e9;

        for (int i = 0; i < trials; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            Impl::run(d_W, d_x, d_B, d_A, d_y, {padded_m, padded_d, padded_b, padded_r}, d_tmp);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            if (i > 0) { // Ignore the first trial for warmup
                min_time_ms = std::min(min_time_ms, elapsed.count());
            }
        }

        // Copy the result back to host
        cudaMemcpy(h_y_padded, d_y, size_y_padded, cudaMemcpyDeviceToHost);

        // Extract the valid results
        __half *h_y_valid = (__half*)malloc(dims.size_m * dims.size_b * sizeof(__half));
        for (int i = 0; i < dims.size_m; ++i)
            for (int j = 0; j < dims.size_b; ++j)
                h_y_valid[i * dims.size_b + j] = h_y_padded[i * padded_b + j];

        // Save results
        auto dim_tuple = std::make_tuple(dims.size_m, dims.size_d, dims.size_b, dims.size_r);
        results.outputs[dim_tuple] = h_y_valid;
        results.elapsed_ms[dim_tuple] = min_time_ms;

        // Free device and host memory
        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_tmp);
        free(h_W);
        free(h_B);
        free(h_A);
        free(h_x);
        free(h_W_fp32);
        free(h_B_fp32);
        free(h_A_fp32);
        free(h_x_fp32);
        free(h_W_padded);
        free(h_B_padded);
        free(h_A_padded);
        free(h_x_padded);
        free(h_y_padded);
    }

    printf("Done with %s\n", Impl::name);

    return results;
}

int main() {
    std::vector<Dimensions> dimensions_list = {
        // {8192, 8192, ...}
        {8192, 8192, 1, 1}, {8192, 8192, 1, 2}, {8192, 8192, 1, 4}, {8192, 8192, 1, 8}, {8192, 8192, 1, 16}, {8192, 8192, 1, 32}, {8192, 8192, 1, 64},
        {8192, 8192, 2, 1}, {8192, 8192, 2, 2}, {8192, 8192, 2, 4}, {8192, 8192, 2, 8}, {8192, 8192, 2, 16}, {8192, 8192, 2, 32}, {8192, 8192, 2, 64},
        {8192, 8192, 4, 1}, {8192, 8192, 4, 2}, {8192, 8192, 4, 4}, {8192, 8192, 4, 8}, {8192, 8192, 4, 16}, {8192, 8192, 4, 32}, {8192, 8192, 4, 64},
        {8192, 8192, 8, 1}, {8192, 8192, 8, 2}, {8192, 8192, 8, 4}, {8192, 8192, 8, 8}, {8192, 8192, 8, 16}, {8192, 8192, 8, 32}, {8192, 8192, 8, 64},
        {8192, 8192, 16, 1}, {8192, 8192, 16, 2}, {8192, 8192, 16, 4}, {8192, 8192, 16, 8}, {8192, 8192, 16, 16}, {8192, 8192, 16, 32}, {8192, 8192, 16, 64},
        {8192, 8192, 32, 1}, {8192, 8192, 32, 2}, {8192, 8192, 32, 4}, {8192, 8192, 32, 8}, {8192, 8192, 32, 16}, {8192, 8192, 32, 32}, {8192, 8192, 32, 64},

        // {4096, 4096, ...}
        {4096, 4096, 1, 1}, {4096, 4096, 1, 2}, {4096, 4096, 1, 4}, {4096, 4096, 1, 8}, {4096, 4096, 1, 16}, {4096, 4096, 1, 32}, {4096, 4096, 1, 64},
        {4096, 4096, 2, 1}, {4096, 4096, 2, 2}, {4096, 4096, 2, 4}, {4096, 4096, 2, 8}, {4096, 4096, 2, 16}, {4096, 4096, 2, 32}, {4096, 4096, 2, 64},
        {4096, 4096, 4, 1}, {4096, 4096, 4, 2}, {4096, 4096, 4, 4}, {4096, 4096, 4, 8}, {4096, 4096, 4, 16}, {4096, 4096, 4, 32}, {4096, 4096, 4, 64},
        {4096, 4096, 8, 1}, {4096, 4096, 8, 2}, {4096, 4096, 8, 4}, {4096, 4096, 8, 8}, {4096, 4096, 8, 16}, {4096, 4096, 8, 32}, {4096, 4096, 8, 64},
        {4096, 4096, 16, 1}, {4096, 4096, 16, 2}, {4096, 4096, 16, 4}, {4096, 4096, 16, 8}, {4096, 4096, 16, 16}, {4096, 4096, 16, 32}, {4096, 4096, 16, 64},
        {4096, 4096, 32, 1}, {4096, 4096, 32, 2}, {4096, 4096, 32, 4}, {4096, 4096, 32, 8}, {4096, 4096, 32, 16}, {4096, 4096, 32, 32}, {4096, 4096, 32, 64},

        // {2048, 2048, ...}
        {2048, 2048, 1, 1}, {2048, 2048, 1, 2}, {2048, 2048, 1, 4}, {2048, 2048, 1, 8}, {2048, 2048, 1, 16}, {2048, 2048, 1, 32}, {2048, 2048, 1, 64},
        {2048, 2048, 2, 1}, {2048, 2048, 2, 2}, {2048, 2048, 2, 4}, {2048, 2048, 2, 8}, {2048, 2048, 2, 16}, {2048, 2048, 2, 32}, {2048, 2048, 2, 64},
        {2048, 2048, 4, 1}, {2048, 2048, 4, 2}, {2048, 2048, 4, 4}, {2048, 2048, 4, 8}, {2048, 2048, 4, 16}, {2048, 2048, 4, 32}, {2048, 2048, 4, 64},
        {2048, 2048, 8, 1}, {2048, 2048, 8, 2}, {2048, 2048, 8, 4}, {2048, 2048, 8, 8}, {2048, 2048, 8, 16}, {2048, 2048, 8, 32}, {2048, 2048, 8, 64},
        {2048, 2048, 16, 1}, {2048, 2048, 16, 2}, {2048, 2048, 16, 4}, {2048, 2048, 16, 8}, {2048, 2048, 16, 16}, {2048, 2048, 16, 32}, {2048, 2048, 16, 64},
        {2048, 2048, 32, 1}, {2048, 2048, 32, 2}, {2048, 2048, 32, 4}, {2048, 2048, 32, 8}, {2048, 2048, 32, 16}, {2048, 2048, 32, 32}, {2048, 2048, 32, 64},

        // {1024, 1024, ...}
        {1024, 1024, 1, 1}, {1024, 1024, 1, 2}, {1024, 1024, 1, 4}, {1024, 1024, 1, 8}, {1024, 1024, 1, 16}, {1024, 1024, 1, 32}, {1024, 1024, 1, 64},
        {1024, 1024, 2, 1}, {1024, 1024, 2, 2}, {1024, 1024, 2, 4}, {1024, 1024, 2, 8}, {1024, 1024, 2, 16}, {1024, 1024, 2, 32}, {1024, 1024, 2, 64},
        {1024, 1024, 4, 1}, {1024, 1024, 4, 2}, {1024, 1024, 4, 4}, {1024, 1024, 4, 8}, {1024, 1024, 4, 16}, {1024, 1024, 4, 32}, {1024, 1024, 4, 64},
        {1024, 1024, 8, 1}, {1024, 1024, 8, 2}, {1024, 1024, 8, 4}, {1024, 1024, 8, 8}, {1024, 1024, 8, 16}, {1024, 1024, 8, 32}, {1024, 1024, 8, 64},
        {1024, 1024, 16, 1}, {1024, 1024, 16, 2}, {1024, 1024, 16, 4}, {1024, 1024, 16, 8}, {1024, 1024, 16, 16}, {1024, 1024, 16, 32}, {1024, 1024, 16, 64},
        {1024, 1024, 32, 1}, {1024, 1024, 32, 2}, {1024, 1024, 32, 4}, {1024, 1024, 32, 8}, {1024, 1024, 32, 16}, {1024, 1024, 32, 32}, {1024, 1024, 32, 64},

        // {512, 512, ...}
        {512, 512, 1, 1}, {512, 512, 1, 2}, {512, 512, 1, 4}, {512, 512, 1, 8}, {512, 512, 1, 16}, {512, 512, 1, 32}, {512, 512, 1, 64},
        {512, 512, 2, 1}, {512, 512, 2, 2}, {512, 512, 2, 4}, {512, 512, 2, 8}, {512, 512, 2, 16}, {512, 512, 2, 32}, {512, 512, 2, 64},
        {512, 512, 4, 1}, {512, 512, 4, 2}, {512, 512, 4, 4}, {512, 512, 4, 8}, {512, 512, 4, 16}, {512, 512, 4, 32}, {512, 512, 4, 64},
        {512, 512, 8, 1}, {512, 512, 8, 2}, {512, 512, 8, 4}, {512, 512, 8, 8}, {512, 512, 8, 16}, {512, 512, 8, 32}, {512, 512, 8, 64},
        {512, 512, 16, 1}, {512, 512, 16, 2}, {512, 512, 16, 4}, {512, 512, 16, 8}, {512, 512, 16, 16}, {512, 512, 16, 32}, {512, 512, 16, 64},
        {512, 512, 32, 1}, {512, 512, 32, 2}, {512, 512, 32, 4}, {512, 512, 32, 8}, {512, 512, 32, 16}, {512, 512, 32, 32}, {512, 512, 32, 64},

        // {256, 256, ...}
        {256, 256, 1, 1}, {256, 256, 1, 2}, {256, 256, 1, 4}, {256, 256, 1, 8}, {256, 256, 1, 16}, {256, 256, 1, 32}, {256, 256, 1, 64},
        {256, 256, 2, 1}, {256, 256, 2, 2}, {256, 256, 2, 4}, {256, 256, 2, 8}, {256, 256, 2, 16}, {256, 256, 2, 32}, {256, 256, 2, 64},
        {256, 256, 4, 1}, {256, 256, 4, 2}, {256, 256, 4, 4}, {256, 256, 4, 8}, {256, 256, 4, 16}, {256, 256, 4, 32}, {256, 256, 4, 64},
        {256, 256, 8, 1}, {256, 256, 8, 2}, {256, 256, 8, 4}, {256, 256, 8, 8}, {256, 256, 8, 16}, {256, 256, 8, 32}, {256, 256, 8, 64},
        {256, 256, 16, 1}, {256, 256, 16, 2}, {256, 256, 16, 4}, {256, 256, 16, 8}, {256, 256, 16, 16}, {256, 256, 16, 32}, {256, 256, 16, 64},
        {256, 256, 32, 1}, {256, 256, 32, 2}, {256, 256, 32, 4}, {256, 256, 32, 8}, {256, 256, 32, 16}, {256, 256, 32, 32}, {256, 256, 32, 64},

        // {128, 128, ...}
        {128, 128, 1, 1}, {128, 128, 1, 2}, {128, 128, 1, 4}, {128, 128, 1, 8}, {128, 128, 1, 16}, {128, 128, 1, 32}, {128, 128, 1, 64},
        {128, 128, 2, 1}, {128, 128, 2, 2}, {128, 128, 2, 4}, {128, 128, 2, 8}, {128, 128, 2, 16}, {128, 128, 2, 32}, {128, 128, 2, 64},
        {128, 128, 4, 1}, {128, 128, 4, 2}, {128, 128, 4, 4}, {128, 128, 4, 8}, {128, 128, 4, 16}, {128, 128, 4, 32}, {128, 128, 4, 64},
        {128, 128, 8, 1}, {128, 128, 8, 2}, {128, 128, 8, 4}, {128, 128, 8, 8}, {128, 128, 8, 16}, {128, 128, 8, 32}, {128, 128, 8, 64},
        {128, 128, 16, 1}, {128, 128, 16, 2}, {128, 128, 16, 4}, {128, 128, 16, 8}, {128, 128, 16, 16}, {128, 128, 16, 32}, {128, 128, 16, 64},
        {128, 128, 32, 1}, {128, 128, 32, 2}, {128, 128, 32, 4}, {128, 128, 32, 8}, {128, 128, 32, 16}, {128, 128, 32, 32}, {128, 128, 32, 64},
    };
    std::vector<BenchmarkResults> results;

    // Add benchmarks for various implementations
    results.push_back(benchmark<CublasReference>(dimensions_list, 5));
    results.push_back(benchmark<FusedSequential>(dimensions_list, 5));
    results.push_back(benchmark<FusedConcurrent>(dimensions_list, 5));
    results.push_back(benchmark<FusedConcurrentAsymmetric>(dimensions_list, 5));
    results.push_back(benchmark<FusedConcurrentAx>(dimensions_list, 5));
    // Add other implementations here, e.g., results.push_back(benchmark<OtherImpl>(dimensions_list, 5));

    // Reference output
    const BenchmarkResults& reference_results = results[0];

    // Print results
    for (size_t impl_index = 0; impl_index < results.size(); ++impl_index) {
        auto& result = results[impl_index];
        std::cout << "Implementation: " << result.name << "\n";
        printf("  %-6s  %-6s  %-6s  %-6s  %-12s  %-14s\n", "size_m", "size_d", "size_b", "size_r", "elapsed_ms", "tflop_per_sec");
        printf("  %-6s  %-6s  %-6s  %-6s  %-12s  %-14s\n", "------", "------", "------", "------", "-----------", "--------------");

        for (const auto& [dims, elapsed_ms] : result.elapsed_ms) {
            auto [size_m, size_d, size_b, size_r] = dims;
            double tflop = 2.0 * size_m * size_d * size_b * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);

            double ref_elapsed_ms = reference_results.elapsed_ms.at(dims);
            double speedup = ref_elapsed_ms / elapsed_ms;
            result.speedup[dims] = speedup;

            printf("  %-6d  %-6d  %-6d  %-6d  %.2f %-14.2f\n",
                   size_m, size_d, size_b, size_r, elapsed_ms, tflop_per_sec);

#if CORRECTNESS
            if (impl_index > 0) { // Skip the reference implementation itself
                const __half* ref_output = reference_results.outputs.at(dims);
                const __half* impl_output = result.outputs.at(dims);

                // Compare outputs
                bool is_correct = true;
                size_t total_elements = size_m * size_b;
                for (size_t i = 0; i < total_elements; ++i) {
                    float ref_val = __half2float(ref_output[i]);
                    float impl_val = __half2float(impl_output[i]);
                    int macs = (
                        size_m * size_b * size_d +
                        size_r * size_b * size_d +
                        size_m * size_b * size_r
                    );
                    float atol = macs * EPSILON_PER_MAC;
                    if (std::abs(ref_val - impl_val) > atol) {
                        printf("%f != %f\n", ref_val, impl_val);
                        is_correct = false;
                        break;
                    }
                }

                // Print warning if incorrect
                if (!is_correct) {
                    std::cout << "WARNING: Output mismatch for dimensions ("
                              << size_m << ", " << size_d << ", " << size_b << ", " << size_r
                              << ") in implementation " << result.name << "\n";
                }
            }
#endif
        }

        std::cout << std::endl;
    }

    // Print a summary table of speedups
    std::cout << "Speedup Summary Table\n";
    printf("  %-5s %-5s %-5s %-5s", "m", "d", "b", "r");

    // Print implementation names as headers
    for (const auto& result : results) {
        printf(" %-12s", result.name);
    }
    printf(" %-15s\n", "Fastest Impl");

    // Print separator line
    printf("  %-5s %-5s %-5s %-5s", "---", "---", "---", "---");
    for (size_t impl_index = 0; impl_index < results.size(); ++impl_index) {
        printf(" %-12s", "------------");
    }
    printf(" %-15s\n", "---------------");

    // Iterate through all problem sizes and print speedups
    const auto& reference_dims = results[0].elapsed_ms; // Assume all results share the same problem sizes
    for (const auto& [dims, _] : reference_dims) {
        auto [size_m, size_d, size_b, size_r] = dims;
        printf("  %-5d %-5d %-5d %-5d", size_m, size_d, size_b, size_r);

        double fastest_time = 1e9;
        const char* fastest_impl = "N/A";

        // Print speedups for each implementation
        for (const auto& result : results) {
            if (result.elapsed_ms.find(dims) != result.elapsed_ms.end()) {
                double elapsed = result.elapsed_ms.at(dims);
                double speedup = reference_results.elapsed_ms.at(dims) / elapsed;
                printf(" %-12.2f", speedup);

                // Track the fastest implementation
                if (elapsed < fastest_time) {
                    fastest_time = elapsed;
                    fastest_impl = result.name;
                }
            } else {
                printf(" %-12s", "N/A");
            }
        }

        // Print the fastest implementation
        printf(" %-15s\n", fastest_impl);
    }

    // Free the stored outputs
    for (auto& result : results) {
        for (auto& [dims, output] : result.outputs) {
            free(output);
        }
    }

    return 0;
}

