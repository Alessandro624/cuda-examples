NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2
TARGETS = convolution1D convolution2D

all: $(TARGETS)

convolution1D: convolution1D.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

convolution2D: convolution2D.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGETS) *.o

clean-logs:
	rm -f *.log

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
