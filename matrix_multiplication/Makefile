NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2
TARGET ?= matrixMul

all: $(TARGET)

$(TARGET): matrixMul.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) *.o

clean-logs:
	rm -f *.log *.dat *.csv *.png

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
