# Image Manipulation

CUDA examples that load PNG images with `libpng`, run GPU kernels (blur and grayscale), and write PNG outputs.

## Build

```bash
cd image_manip
make
```

Requires `libpng-dev` (or equivalent) installed on the system.

## Usage

### imageBlur

Apply a box blur filter on GPU:

```bash
./imageBlur [--infile IN.png] [--outfile OUT.png]
```

### imageToGrayscale

Convert to grayscale on GPU:

```bash
./imageToGrayscale [--infile IN.png] [--outfile OUT.png]
```

## Run

```bash
./run.sh imageBlur --infile=input.png --outfile=output.png
./run.sh imageToGrayscale --infile=input.png --outfile=gray.png
```

## Profiling

```bash
./profile_nvprof.sh imageBlur --infile=input.png --outfile=output.png
```

## Notes

- Binaries link with `-lpng -lz`. If your system puts headers/libraries in non-standard locations, update `Makefile` accordingly.
- Outputs keep the same number of channels as the input (RGB/RGBA).
- Use `NVCCFLAGS` in the `Makefile` to tune compilation flags.
