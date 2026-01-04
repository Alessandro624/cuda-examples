CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2

NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2

# Targets
TARGET_CPU ?= heat_transfer_cli
TARGET_CUDA ?= heat_transfer_cuda

all: $(TARGET_CPU) $(TARGET_CUDA)

cpu: $(TARGET_CPU)

cuda: $(TARGET_CUDA)

$(TARGET_CPU): heat_transfer_cli.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_CUDA): heat_transfer_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA) *.o

clean-logs:
	rm -f *.log *.dat

clean-all: clean clean-logs

.PHONY: all cpu cuda clean clean-logs clean-all
