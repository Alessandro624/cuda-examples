# Makefile for CUDA Profiling Tools
# Usage:
#   make            - Build all tools
#   make clean      - Remove built files
#   make test       - Run basic tests to verify environment

# Configuration
NVCC := nvcc
PYTHON := python3

# Fallback to common architectures if detection fails
ifeq ($(GPU_ARCH),)
    GPU_ARCH := 52 61 75 86
    ARCH_FLAGS := $(foreach arch,$(GPU_ARCH),-gencode arch=compute_$(arch),code=sm_$(arch))
else
    ARCH_FLAGS := -arch=sm_$(GPU_ARCH)
endif

# Compiler flags
NVCC_FLAGS := -O3 $(ARCH_FLAGS) -std=c++11
CUDA_LIBS := -lcudart

# Targets
TARGETS := gpu_info

.PHONY: all clean test install help

all: $(TARGETS)
	@echo "✓ All tools built successfully"
	@echo "Run './profile_cuda.sh --help' for usage"

gpu_info: gpu_info.cu
	@echo "Building GPU info tool for architecture $(GPU_ARCH)..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(CUDA_LIBS)
	@echo "✓ gpu_info built"

clean:
	@echo "Cleaning build files..."
	rm -f $(TARGETS)
	rm -rf profiling_results_*
	rm -f *.o
	@echo "✓ Clean complete"

test: all
	@echo "Running basic tests..."
	@echo "1. Testing GPU info tool..."
	@./gpu_info > /dev/null && echo "  ✓ GPU info works" || echo "  ✗ GPU info failed"
	@echo "2. Checking Python dependencies..."
	@$(PYTHON) -c "import sys; print('  ✓ Python', sys.version.split()[0])"
	@echo "3. Checking profiler availability..."
	@which ncu > /dev/null && echo "  ✓ Nsight Compute (ncu) available" || \
	 which nvprof > /dev/null && echo "  ✓ nvprof available" || \
	 echo "  ✗ No profiler found (install CUDA toolkit)"
	@echo "4. Checking gnuplot..."
	@which gnuplot > /dev/null && echo "  ✓ gnuplot available" || \
	 echo "  ⚠ gnuplot not found (plots will not be generated)"

help:
	@echo "CUDA Profiling Tools - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build all tools"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make test         - Run basic environment tests"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Detected GPU architecture: $(GPU_ARCH)"
	@echo "NVCC flags: $(NVCC_FLAGS)"
	@echo ""
	@echo "Examples:"
	@echo "  make && ./profile_cuda.sh --dir ./build"
	@echo "  make test"

# Architecture-specific builds
gpu_info_pascal: ARCH_FLAGS = -arch=sm_60
gpu_info_pascal: gpu_info

gpu_info_volta: ARCH_FLAGS = -arch=sm_70
gpu_info_volta: gpu_info

gpu_info_turing: ARCH_FLAGS = -arch=sm_75
gpu_info_turing: gpu_info

gpu_info_ampere: ARCH_FLAGS = -arch=sm_80
gpu_info_ampere: gpu_info

gpu_info_hopper: ARCH_FLAGS = -arch=sm_90
gpu_info_hopper: gpu_info

# Debug build
debug: NVCC_FLAGS = -g -G -O0 $(ARCH_FLAGS)
debug: $(TARGETS)
	@echo "✓ Debug build complete"
