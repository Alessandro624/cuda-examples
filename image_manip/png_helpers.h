#ifndef IMAGE_MANIP_PNG_HELPERS_H
#define IMAGE_MANIP_PNG_HELPERS_H

/*
 * png_helpers.h
 *
 * Header-only helpers for reading/writing PNG via libpng and a CHECK_CUDA macro.
 * Functions are `static` to allow inclusion in multiple translation units.
 */

#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#ifndef CHECK_CUDA
#include <cuda_runtime.h>
#define CHECK_CUDA(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = (call);                                                                      \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)
#endif

static unsigned char *read_png(const char *filename, int *width, int *height, int *channels)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open %s\n", filename);
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        return NULL;
    }
    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(fp);
        return NULL;
    }
    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    png_uint_32 w = png_get_image_width(png, info);
    png_uint_32 h = png_get_image_height(png, info);
    int color_type = png_get_color_type(png, info);
    int bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    // Force an 8-bit RGB(A) layout
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    png_uint_32 rowbytes = png_get_rowbytes(png, info);
    unsigned char *image_data = (unsigned char *)malloc((size_t)rowbytes * h);
    if (!image_data)
    {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return NULL;
    }
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * h);
    for (unsigned int y = 0; y < h; ++y)
        row_pointers[y] = image_data + y * rowbytes;

    png_read_image(png, row_pointers);

    int ch = (int)(rowbytes / w);
    *width = (int)w;
    *height = (int)h;
    *channels = ch;

    free(row_pointers);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return image_data;
}

static int write_png(const char *filename, unsigned char *image_data, int width, int height, int channels)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return -1;
    }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
    {
        fclose(fp);
        return -1;
    }
    png_infop info = png_create_info_struct(png);
    if (!info)
    {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return -1;
    }
    if (setjmp(png_jmpbuf(png)))
    {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return -1;
    }

    png_init_io(png, fp);

    int color_type = (channels == 4) ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png, info, width, height, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    int rowbytes = width * channels;
    for (int y = 0; y < height; ++y)
        row_pointers[y] = image_data + y * rowbytes;

    png_set_rows(png, info, row_pointers);
    png_write_png(png, info, PNG_TRANSFORM_IDENTITY, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 0;
}

#endif // IMAGE_MANIP_PNG_HELPERS_H
