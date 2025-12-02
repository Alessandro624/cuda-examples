NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++11
TARGET ?= matrixVectMul

all: $(TARGET)

$(TARGET): matrixVectMul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

clean-logs:
	rm -f *.log

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
