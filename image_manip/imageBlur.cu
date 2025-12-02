/*
 * imageBlur.cu - apply a small box blur to a PNG using the GPU.
 *
 * Usage: imageBlur [--infile INFILE] [--outfile OUTFILE]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <png.h>
#include <cuda_runtime.h>
#include "png_helpers.h"
#include "../common/cli_utils.h"

// to change the blur effect
#define BLUR_SIZE 3

__global__ void blurImageKernelCoarsening(unsigned char *Pout, const unsigned char *Pin, int width, int height, int channels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        for (int c = 0; c < channels; ++c)
        {
            int pixVal = 0, pixels = 0;
            // average of the surrounding box
            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow)
            {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol)
                {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    // to prevent invalid image pixels
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                    {
                        int offset = (curRow * width + curCol) * channels;
                        pixVal += Pin[offset + c];
                        ++pixels;
                    }
                }
            }
            int offsetOut = (row * width + col) * channels;
            Pout[offsetOut + c] = (unsigned char)(pixVal / pixels);
        }
    }
}

void blurImage(unsigned char *Pout, unsigned char *Pin, int width, int height, int channels)
{
    unsigned char *Pin_d = NULL, *Pout_d = NULL;
    size_t size = (size_t)width * height * channels * sizeof(unsigned char);
    // allocating device memory
    CHECK_CUDA(cudaMalloc((void **)&Pin_d, size));
    CHECK_CUDA(cudaMalloc((void **)&Pout_d, size));
    // copying from host to device memory
    CHECK_CUDA(cudaMemcpy(Pin_d, Pin, size, cudaMemcpyHostToDevice));
    // number of blocks to call (2D)
    dim3 dimGrid((width + 15) / 16, (height + 15) / 16, 1);
    // number of threads in each block (2D)
    dim3 dimBlock(16, 16, 1);
    // calling kernel with coarsening
    blurImageKernelCoarsening<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height, channels);
    CHECK_CUDA(cudaGetLastError());
    // copying from device to host memory
    CHECK_CUDA(cudaMemcpy(Pout, Pout_d, size, cudaMemcpyDeviceToHost));
    // freeing device memory
    CHECK_CUDA(cudaFree(Pin_d));
    CHECK_CUDA(cudaFree(Pout_d));
}

/**
 * main - parse flags, load PNG, run blur kernel and write result.
 */
int main(int argc, char **argv)
{
    if (cli_has_help(argc, argv))
    {
        printf("Usage: %s [--infile INFILE] [--outfile OUTFILE]\n", argv[0]);
        printf("  infile: input PNG (default: test_image.png)\n");
        printf("  outfile: output PNG (default: test_image_blurred.png)\n");
        return 0;
    }

    const char *infile = "test_image.png";
    const char *outfile = "test_image_blurred.png";
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

    blurImage(Pout_h, Pin_h, width, height, channels);

    if (write_png(outfile, Pout_h, width, height, channels) != 0)
        fprintf(stderr, "Failed to write PNG %s\n", outfile);
    else
        printf("Wrote blurred image to %s\n", outfile);

    free(Pin_h);
    free(Pout_h);
    return 0;
}
