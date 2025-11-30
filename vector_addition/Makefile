NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2
TARGET ?= vectAdd

all: $(TARGET)

$(TARGET): vectAdd.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -f $(TARGET) *.o

clean-logs:
	rm -f *.log

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
