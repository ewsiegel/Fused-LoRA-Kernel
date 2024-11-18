# Makefile for Optimized CUDA Programs with Output in build Directory

# Compiler
NVCC = nvcc

# Target Executable
TARGET = matmul_mixed_precision

# Build Directory
BUILD_DIR = build

# Source Files
SRCS = $(wildcard *.cu)

# Object Files
OBJS = $(SRCS:%.cu=$(BUILD_DIR)/%.o)

# Compute Capability (adjust for your GPU architecture)
ARCH = sm_86

# Optimization Flags
OPT_FLAGS = -O3 --use_fast_math --ftz=true --prec-div=false --maxrregcount=32

# Debugging and Profiling Flags (comment out in production)
# DEBUG_FLAGS = -G -lineinfo

# Generate PTX code for multiple architectures (optional)
GENCODE_FLAGS = 

# Default Flags
NVCC_FLAGS = $(OPT_FLAGS) $(GENCODE_FLAGS) -Xptxas -v -arch=$(ARCH)

# Host Compiler Options
HOST_FLAGS = -std=c++17 -Wall

# Linker Flags (if needed)
LDFLAGS = 

# Build Rules
all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler "$(HOST_FLAGS)" -c $< -o $@

# Clean Build
clean:
	rm -rf $(BUILD_DIR)

# Dependency Generation (Optional)
depend: $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -M $^ > $(BUILD_DIR)/Makefile.deps

-include $(BUILD_DIR)/Makefile.deps

# Phony Targets
.PHONY: all clean depend
