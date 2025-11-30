// Common CLI and utility helpers used across examples
#ifndef CLI_UTILS_H
#define CLI_UTILS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static inline int is_positive_integer_str(const char *s)
{
    if (!s || *s == '\0')
        return 0;
    const char *p = s;
    if (*p == '+')
        ++p;
    while (*p)
    {
        if (*p < '0' || *p > '9')
            return 0;
        ++p;
    }
    return 1;
}

// Check whether help was requested (-h or --help)
static inline int cli_has_help(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
            return 1;
    }
    return 0;
}

// Find a flag value for a key in argv. Supports `--key=value` and `--key value`.
// Returns pointer to the value string if found, otherwise NULL.
static inline const char *cli_find_flag_value(int argc, char **argv, const char *key)
{
    size_t keylen = strlen(key);
    for (int i = 1; i < argc; ++i)
    {
        const char *a = argv[i];
        if (strncmp(a, "--", 2) != 0)
            continue;
        const char *k = a + 2;
        const char *eq = strchr(k, '=');
        if (eq)
        {
            size_t klen = (size_t)(eq - k);
            if (klen == keylen && strncmp(k, key, keylen) == 0)
                return eq + 1;
        }
        else
        {
            if (strncmp(k, key, keylen) == 0 && k[keylen] == '\0')
            {
                if (i + 1 < argc)
                    return argv[i + 1];
                return NULL;
            }
        }
    }
    return NULL;
}

// Compute number of blocks as ceil(n / threads) for integer types
static inline int compute_blocks_from_elements(long long n, int threads)
{
    if (threads <= 0)
        return 1;
    long long b = (n + (long long)threads - 1) / (long long)threads;
    return (int)(b > 0 ? b : 1);
}

// Clamp threads to device maxThreadsPerBlock and print a warning if clamped.
static inline int clamp_threads_to_device(int threads)
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return threads;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return threads;
    int maxT = prop.maxThreadsPerBlock;
    if (threads > maxT)
    {
        fprintf(stderr, "Warning: requested threads=%d > device maxThreadsPerBlock=%d; clamping to %d\n", threads, maxT, maxT);
        return maxT;
    }
    return threads;
}

#endif // CLI_UTILS_H
