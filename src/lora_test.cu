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

#define EPSILON 1e-6
#define CORRECTNESS true

// Function to initialize an FP32 array with random values and convert to FP16
void initialize_random_fp16(float* array_fp32, __half* array_fp16, int size, float min_value, float max_value, unsigned int seed) {
    std::srand(seed);
    for (int i = 0; i < size; ++i) {
        array_fp32[i] = min_value + static_cast<float>(rand()) / RAND_MAX * (max_value - min_value);
        array_fp16[i] = __float2half(array_fp32[i]);
    }
}

// Struct to store benchmark results
struct BenchmarkResults {
    char const* name;
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, __half*> outputs;
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, double> elapsed_ms;
};

// General benchmark function
template <typename Impl>
BenchmarkResults benchmark(const std::vector<Dimensions>& dimensions_list, int trials) {
    BenchmarkResults results;
    results.name = Impl::name;

    for (const auto& dims : dimensions_list) {
        size_t size_W = dims.size_m * dims.size_d * sizeof(__half);
        size_t size_B = dims.size_m * dims.size_r * sizeof(__half);
        size_t size_A = dims.size_r * dims.size_d * sizeof(__half);
        size_t size_x = dims.size_d * dims.size_b * sizeof(__half);
        size_t size_y = dims.size_m * dims.size_b * sizeof(__half);

        float *h_W_fp32 = (float*)malloc(dims.size_m * dims.size_d * sizeof(float));
        float *h_B_fp32 = (float*)malloc(dims.size_m * dims.size_r * sizeof(float));
        float *h_A_fp32 = (float*)malloc(dims.size_r * dims.size_d * sizeof(float));
        float *h_x_fp32 = (float*)malloc(dims.size_d * dims.size_b * sizeof(float));

        __half *h_W = (__half*)malloc(size_W);
        __half *h_B = (__half*)malloc(size_B);
        __half *h_A = (__half*)malloc(size_A);
        __half *h_x = (__half*)malloc(size_x);
        __half *h_y = (__half*)malloc(size_y);

        unsigned int seed = 42;
        initialize_random_fp16(h_W_fp32, h_W, dims.size_m * dims.size_d, -1000.0f, 1000.0f, seed);
        initialize_random_fp16(h_B_fp32, h_B, dims.size_m * dims.size_r, -1000.0f, 1000.0f, seed);
        initialize_random_fp16(h_A_fp32, h_A, dims.size_r * dims.size_d, -1000.0f, 1000.0f, seed);
        initialize_random_fp16(h_x_fp32, h_x, dims.size_d * dims.size_b, -1000.0f, 1000.0f, seed);

        __half *d_W, *d_B, *d_A, *d_x, *d_y;
        cudaMalloc(&d_W, size_W);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_x, size_x);
        cudaMalloc(&d_y, size_y);

        cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

        double min_time_ms = 1e9;

        for (int i = 0; i < trials; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            Impl::run(d_W, d_x, d_B, d_A, d_y, dims);  // Call the templated LoRA implementation
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            if (i > 0) { // Ignore the first trial for warmup
                min_time_ms = std::min(min_time_ms, elapsed.count());
            }
        }

        cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost);

        // Save results
        auto dim_tuple = std::make_tuple(dims.size_m, dims.size_d, dims.size_b, dims.size_r);
        results.outputs[dim_tuple] = h_y;
        results.elapsed_ms[dim_tuple] = min_time_ms;

        cudaFree(d_W);
        cudaFree(d_B);
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        free(h_W_fp32);
        free(h_B_fp32);
        free(h_A_fp32);
        free(h_x_fp32);
        free(h_W);
        free(h_B);
        free(h_A);
        free(h_x);
        // Do not free h_y as it's stored in results.outputs
    }

    return results;
}

int main() {
    std::vector<Dimensions> dimensions_list = {
        {2048, 2048, 16, 16},
        {2048, 2048, 16, 32},
        {2048, 2048, 16, 64},
        {2048, 2048, 16, 128},
        {2048, 2048, 32, 16},
        {2048, 2048, 32, 32},
        {2048, 2048, 32, 64},
        {2048, 2048, 32, 128},
        {2048, 2048, 64, 16},
        {2048, 2048, 64, 32},
        {2048, 2048, 64, 64},
        {2048, 2048, 64, 128},
        {2048, 2048, 128, 16},
        {2048, 2048, 128, 32},
        {2048, 2048, 128, 64},
        {2048, 2048, 128, 128},
        {2048, 2048, 256, 16},
        {2048, 2048, 256, 32},
        {2048, 2048, 256, 64},
        {2048, 2048, 256, 128},
        {2048, 2048, 512, 16},
        {2048, 2048, 512, 32},
        {2048, 2048, 512, 64},
        {2048, 2048, 512, 128},
        {2048, 2048, 1024, 16},
        {2048, 2048, 1024, 32},
        {2048, 2048, 1024, 64},
        {2048, 2048, 1024, 128},

        {1024, 1024, 16, 16},
        {1024, 1024, 16, 32},
        {1024, 1024, 16, 64},
        {1024, 1024, 16, 128},
        {1024, 1024, 32, 16},
        {1024, 1024, 32, 32},
        {1024, 1024, 32, 64},
        {1024, 1024, 32, 128},
        {1024, 1024, 64, 16},
        {1024, 1024, 64, 32},
        {1024, 1024, 64, 64},
        {1024, 1024, 64, 128},
        {1024, 1024, 128, 16},
        {1024, 1024, 128, 32},
        {1024, 1024, 128, 64},
        {1024, 1024, 128, 128},
        {1024, 1024, 256, 16},
        {1024, 1024, 256, 32},
        {1024, 1024, 256, 64},
        {1024, 1024, 256, 128},
        {1024, 1024, 512, 16},
        {1024, 1024, 512, 32},
        {1024, 1024, 512, 64},
        {1024, 1024, 512, 128},
        {1024, 1024, 1024, 16},
        {1024, 1024, 1024, 32},
        {1024, 1024, 1024, 64},
        {1024, 1024, 1024, 128},

        {512, 512, 16, 16},
        {512, 512, 16, 32},
        {512, 512, 16, 64},
        {512, 512, 16, 128},
        {512, 512, 32, 16},
        {512, 512, 32, 32},
        {512, 512, 32, 64},
        {512, 512, 32, 128},
        {512, 512, 64, 16},
        {512, 512, 64, 32},
        {512, 512, 64, 64},
        {512, 512, 64, 128},
        {512, 512, 128, 16},
        {512, 512, 128, 32},
        {512, 512, 128, 64},
        {512, 512, 128, 128},
        {512, 512, 256, 16},
        {512, 512, 256, 32},
        {512, 512, 256, 64},
        {512, 512, 256, 128},
        {512, 512, 512, 16},
        {512, 512, 512, 32},
        {512, 512, 512, 64},
        {512, 512, 512, 128},
        {512, 512, 1024, 16},
        {512, 512, 1024, 32},
        {512, 512, 1024, 64},
        {512, 512, 1024, 128},

        {256, 256, 16, 16},
        {256, 256, 16, 32},
        {256, 256, 16, 64},
        {256, 256, 16, 128},
        {256, 256, 32, 16},
        {256, 256, 32, 32},
        {256, 256, 32, 64},
        {256, 256, 32, 128},
        {256, 256, 64, 16},
        {256, 256, 64, 32},
        {256, 256, 64, 64},
        {256, 256, 64, 128},
        {256, 256, 128, 16},
        {256, 256, 128, 32},
        {256, 256, 128, 64},
        {256, 256, 128, 128},
        {256, 256, 256, 16},
        {256, 256, 256, 32},
        {256, 256, 256, 64},
        {256, 256, 256, 128},
        {256, 256, 512, 16},
        {256, 256, 512, 32},
        {256, 256, 512, 64},
        {256, 256, 512, 128},
        {256, 256, 1024, 16},
        {256, 256, 1024, 32},
        {256, 256, 1024, 64},
        {256, 256, 1024, 128},

        {128, 128, 16, 16},
        {128, 128, 16, 32},
        {128, 128, 16, 64},
        {128, 128, 16, 128},
        {128, 128, 32, 16},
        {128, 128, 32, 32},
        {128, 128, 32, 64},
        {128, 128, 32, 128},
        {128, 128, 64, 16},
        {128, 128, 64, 32},
        {128, 128, 64, 64},
        {128, 128, 64, 128},
        {128, 128, 128, 16},
        {128, 128, 128, 32},
        {128, 128, 128, 64},
        {128, 128, 128, 128},
        {128, 128, 256, 16},
        {128, 128, 256, 32},
        {128, 128, 256, 64},
        {128, 128, 256, 128},
        {128, 128, 512, 16},
        {128, 128, 512, 32},
        {128, 128, 512, 64},
        {128, 128, 512, 128},
        {128, 128, 1024, 16},
        {128, 128, 1024, 32},
        {128, 128, 1024, 64},
        {128, 128, 1024, 128},
    };
    std::vector<BenchmarkResults> results;

    // Add benchmarks for various implementations
    results.push_back(benchmark<CublasReference>(dimensions_list, 5));
    results.push_back(benchmark<FusedSequential>(dimensions_list, 5));
    // Add other implementations here, e.g., results.push_back(benchmark<OtherImpl>(dimensions_list, 5));

    // Reference output
    const BenchmarkResults& reference_results = results[0];

    // Print results
    for (size_t impl_index = 0; impl_index < results.size(); ++impl_index) {
        const auto& result = results[impl_index];
        std::cout << "Implementation: " << result.name << "\n";
        printf("  %-6s  %-6s  %-6s  %-6s  %-12s  %-14s\n", "size_m", "size_d", "size_b", "size_r", "elapsed_ms", "tflop_per_sec");
        printf("  %-6s  %-6s  %-6s  %-6s  %-12s  %-14s\n", "------", "------", "------", "------", "-----------", "--------------");

        for (const auto& [dims, elapsed_ms] : result.elapsed_ms) {
            auto [size_m, size_d, size_b, size_r] = dims;
            double tflop = 2.0 * size_m * size_d * size_b * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);

            double ref_elapsed_ms = reference_results.elapsed_ms.at(dims);
            double speedup = ref_elapsed_ms / elapsed_ms;

            printf("  %-6d  %-6d  %-6d  %-6d  %.2f (%.2fx speedup) %-14.2f\n",
                   size_m, size_d, size_b, size_r, elapsed_ms, speedup, tflop_per_sec);

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
                    if (std::abs(ref_val - impl_val) > EPSILON) {
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

    // Free the stored outputs
    for (auto& result : results) {
        for (auto& [dims, output] : result.outputs) {
            free(output);
        }
    }

    return 0;
}

