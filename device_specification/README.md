# Device Specification

Utility to enumerate CUDA devices and print hardware limits and properties useful for tuning kernels and understanding the platform.

## Build

```bash
cd device_specification
make
```

## Usage

```bash
./deviceSpec [--device DEVICE_INDEX]
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--device` | Device index to query | All devices |

## Run

```bash
# Print all devices
./run.sh

# Print device 0 only
./run.sh --device=0
```

## Profiling

This is a query utility; profiling is not typically needed.

## Notes

- Uses `cudaGetDeviceProperties` to collect a broad set of fields: memory, SMPs, registers, warp size, clock rates, compute capability, PCI IDs, ECC and concurrency flags.
- Useful baseline for kernel tuning.
- No external libraries required beyond the CUDA toolkit.
