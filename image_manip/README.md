# Image Manipulation (libpng)

Simple CUDA examples that load PNG images with `libpng`, run GPU kernels (blur and grayscale), and write PNG outputs.

Build:

```bash
cd image_manip
make
```

Programs and usage:

- `imageBlur [--infile IN.png] [--outfile OUT.png]` : apply a small box blur (GPU)
- `imageToGrayscale [--infile IN.png] [--outfile OUT.png]` : convert to grayscale on GPU

Run examples (local runner):

```bash
./run.sh imageBlur --infile=input.png --outfile=output.png
```

Profile with nvprof:

```bash
./profile_nvprof.sh imageBlur --infile=input.png --outfile=output.png
```

Notes:

- These examples use `libpng` from the system. Ensure `libpng-dev` (or equivalent) is installed and visible to the compiler.
- The binaries link with `-lpng -lz`. If your system puts headers/libraries in non-standard locations, update `Makefile` accordingly.
- Outputs keep the same number of channels as the input (RGB/RGBA).
