# Device Specification

Purpose: utility to enumerate CUDA devices and print hardware limits and properties useful for tuning kernels and understanding the platform.

Build:

```bash
cd device_specification
make
```

Programs and usage:

- `deviceSpec [--device device_index]` : print properties for all devices or for the supplied device index.

Run examples (local runner):

```bash
./run.sh --device=0    # print device 0 only
# No args prints all devices
./run.sh
```

Notes:

- Uses `cudaGetDeviceProperties` to collect a broad set of fields: memory, SMPs, registers, warp size, clock rates, compute capability, PCI IDs, ECC and concurrency flags. Useful baseline for kernel tuning.
- No external libraries required beyond the CUDA toolkit.
