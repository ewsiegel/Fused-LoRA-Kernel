#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

// Define the data type
using Element = cutlass::half_t;

// Define the CUTLASS GEMM template
using GemmWx = cutlass::gemm::device::Gemm<
    Element,                          // Data-type of A matrix
    cutlass::layout::RowMajor,        // Layout of A matrix
    Element,                          // Data-type of B matrix
    cutlass::layout::RowMajor,        // Layout of B matrix
    Element,                          // Data-type of C matrix
    cutlass::layout::RowMajor         // Layout of C matrix
>;

void compute_Wx(
    Element* W,     // [m x n]
    Element* x,     // [n x b]
    Element* Y,     // [m x b]
    int m, int n, int b) {

    // Create a CUTLASS GEMM operator
    GemmWx gemm_operator;

    // Define the GEMM problem size
    cutlass::gemm::GemmCoord problem_size(m, b, n);

    // Define the GEMM arguments
    GemmWx::Arguments arguments{
        problem_size,
        {W, n},           // TensorRef for W
        {x, b},           // TensorRef for x
        {Y, b},           // TensorRef for Y (destination)
        {Y, b},           // TensorRef for Y (source for accumulation)
        {1.0f, 0.0f}      // Alpha and Beta
    };

    // Launch the GEMM computation
    cutlass::Status status = gemm_operator(arguments);

    // Check for errors
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM Wx failed: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}
