NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2
TARGETS = vectAdd_errors errorCudaMemcpy

all: $(TARGETS)

vectAdd_errors: vectAdd_errors.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

errorCudaMemcpy: errorCudaMemcpy.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGETS) *.o

clean-logs:
	rm -f *.log

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
