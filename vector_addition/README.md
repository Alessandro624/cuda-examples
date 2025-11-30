# Vector Addition (vectAdd)

Small demos that implement vector addition on the GPU. Variants include a baseline implementation, a grid-stride kernel with a thread granularity parameter, and Unified Memory versions (with and without prefetch).

Build:

```bash
cd vector_addition
make
```

Programs and usage:

- `vectAdd [--mode M] [--n N] [--threads T] [--granularity G]` : run vectAdd examples
  - `mode` can select implementation or behavior (see source `vectAdd.cu` or launch with `-h` for available modes).
  - Examples:
    - Flag-style: `./vectAdd --mode=2 --n=1000000 --threads=256 --granularity=4`

Run examples (local runner):

```bash
./run.sh --mode=2 --n=1000000 --threads=256 --granularity=4
```

Profile with nvprof:

```bash
./profile_nvprof.sh --mode=1
```

Notes:

- Use `NVCCFLAGS` in the `Makefile` to tune compilation flags for your hardware.
