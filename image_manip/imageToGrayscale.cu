/*
 * imageToGrayscale.cu - convert PNG to grayscale on the GPU.
 *
 * Usage: imageToGrayscale [--infile INFILE] [--outfile OUTFILE]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <png.h>
#include <cuda_runtime.h>
#include "png_helpers.h"
#include "../common/cli_utils.h"

__global__ void colortoGrayscaleConvertionKernel(unsigned char *Pout, const unsigned char *Pin, int width, int height, int channels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        int offset = (row * width + col) * channels;
        float r = Pin[offset + 0];
        float g = Pin[offset + 1];
        float b = Pin[offset + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        for (int c = 0; c < channels; ++c)
        {
            if (channels == 4 && c == 3)
                Pout[offset + c] = Pin[offset + c];
            else
                Pout[offset + c] = gray;
        }
    }
}

/**
 * main - parse flags, load PNG, run grayscale kernel and write result.
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--infile INFILE] [--outfile OUTFILE]\n", argv[0]);
        printf("  infile: input PNG (default: test_image.png)\n");
        printf("  outfile: output PNG (default: test_image_gray.png)\n");
        return 0;
    }

    const char *infile = "test_image.png";
    const char *outfile = "test_image_gray.png";
    const char *v = cli_find_flag_value(argc, argv, "infile");
    if (v)
        infile = v;
    v = cli_find_flag_value(argc, argv, "outfile");
    if (v)
        outfile = v;

    int width = 0, height = 0, channels = 0;
    unsigned char *Pin_h = read_png(infile, &width, &height, &channels);
    if (!Pin_h)
    {
        fprintf(stderr, "Error loading PNG %s\n", infile);
        return 1;
    }
    printf("Loaded image %s: %dx%d channels=%d\n", infile, width, height, channels);

    unsigned char *Pout_h = (unsigned char *)malloc((size_t)width * height * channels);
    if (!Pout_h)
    {
        fprintf(stderr, "Failed to allocate output buffer\n");
        free(Pin_h);
        return 1;
    }

    unsigned char *Pin_d = NULL, *Pout_d = NULL;
    size_t size = (size_t)width * height * channels * sizeof(unsigned char);
    CHECK_CUDA(cudaMalloc((void **)&Pin_d, size));
    CHECK_CUDA(cudaMalloc((void **)&Pout_d, size));
    CHECK_CUDA(cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice));

    dim3 dimGrid((width + 15) / 16, (height + 15) / 16, 1);
    dim3 dimBlock(16, 16, 1);
    colortoGrayscaleConvertionKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height, channels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));

    if (write_png(outfile, Pout_h, width, height, channels) != 0)
        fprintf(stderr, "Failed to write PNG %s\n", outfile);
    else
        printf("Wrote grayscale image to %s\n", outfile);

    CHECK_CUDA(cudaFree(Pin_d));
    CHECK_CUDA(cudaFree(Pout_d));
    free(Pin_h);
    free(Pout_h);
    return 0;
}
