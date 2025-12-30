NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2
TARGET = parallelHistogram

all: $(TARGET)

$(TARGET): parallelHistogram.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) *.o

clean-logs:
	rm -f *.log

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
