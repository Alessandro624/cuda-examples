NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O2
TARGET ?= deviceSpec

all: $(TARGET)
	
$(TARGET): deviceSpec.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
