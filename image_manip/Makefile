NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++11
LDFLAGS ?= -lpng -lz
TARGETS ?= imageBlur imageToGrayscale

all: $(TARGETS)

imageBlur: imageBlur.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

imageToGrayscale: imageToGrayscale.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

clean-logs:
	rm -f *.log

clean-all: clean clean-logs

.PHONY: all clean clean-logs clean-all
