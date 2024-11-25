# Compiler
NVCC = nvcc

# Target Executable
TARGET = matmul_lora_fused

# Build Directory
BUILD_DIR = build

# Source Directory
SRC_DIR = src

# Source Files
SRCS = $(wildcard $(SRC_DIR)/*.cu)

# Object Files
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Compute Capability (adjust for your GPU architecture)
ARCH = sm_86

# Optimization Flags
OPT_FLAGS = -O3 --use_fast_math --ftz=true --prec-div=false

# Debugging and Profiling Flags (comment out in production)
# DEBUG_FLAGS = -G -lineinfo

# Generate PTX code for multiple architectures (optional)
GENCODE_FLAGS = 

# Default Flags
NVCC_FLAGS = -I./cutlass/include -I./cutlass/tools/util/include -I./wmma_extension/include $(OPT_FLAGS) $(GENCODE_FLAGS) -Xptxas -v -arch=$(ARCH)

# Host Compiler Options
HOST_FLAGS = -std=c++17 -Wall

# Linker Flags (if needed)
LDFLAGS = -lcublas 

# Build Rules
all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
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
