# CUDA example collection

This repository collects small, focused CUDA example programs and helper scripts used for learning and benchmarking. Each subdirectory contains a single example (source, README, and helper scripts).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Build](#quick-build)
- [How to run examples](#how-to-run-examples)
- [Profiling](#profiling)
- [Repository layout and links](#repository-layout-and-links)
- [CLI conventions](#cli-conventions)
- [CI / GitHub Actions](#ci--github-actions)
- [Contributing](#contributing)

## Prerequisites

- Linux (Ubuntu recommended for scripts in this repo)
- NVIDIA CUDA toolkit (nvcc) installed and on `PATH` for local builds
- `make` and standard build tools (`gcc`, `g++`, `make`)
- `nvprof` (or your preferred NVIDIA profiler) if you want to profile; profiling scripts in each directory call `nvprof` by default

If you plan to use the included GitHub Actions workflow, the workflow builds inside an NVIDIA CUDA Docker image so you don't need CUDA installed locally for CI builds.

## Quick Build

From the project root run:

```bash
make -j$(nproc)
```

This will run `make` in every subdirectory that provides a Makefile and build the example binaries.

## How to run examples

Each example directory contains a `run.sh` helper script and a README with example invocations. Most binaries accept an explicit `--help` flag that prints usage.

Example:

```bash
cd vector_addition
./vectAdd --mode 0 --n 1024 --threads 128 --granularity 1
```

Note: binaries accept flags only (no positional fallback). If a directory provides a `run.sh`, it maps convenient script arguments to the program flags when present.

## Profiling

Per-directory profiling scripts are provided and named `profile_nvprof.sh`. They call `nvprof` and save profiler outputs. Example usage (from a subdirectory):

```bash
./profile_nvprof.sh --n 4096 --threads 256
```

If you do not have `nvprof`, install the CUDA toolkit, or run the GitHub Actions CI which builds the project inside a CUDA container.

## Repository layout and links

Click the folders below for the example README files and more details:

- [`Vector Addition`](vector_addition/) — vector add example
- [`Error Handling`](error_handling/) — examples showing CUDA error handling
- [`Device Specification`](device_specification/) — device query and capability examples
- [`Image Manipulation`](image_manip/) — image processing examples (blur, grayscale); includes `stb` helper headers
- [`Matrix-Vector Multiplication`](matrix_vector_multiplication/) — matrix-vector multiplication example
- [`Matrix Multiplication`](matrix_multiplication/) — matrix multiplication example
- [`Convolution`](convolution/) — convolution examples (1D & 2D)

Each folder includes a `README.md` with per-example instructions.

## CLI conventions

- All example binaries use flag-style CLI (e.g., `--n 1024`, `--threads 128`).
- Centralized CLI helpers live in `common/cli_utils.h` and are used across examples for consistent parsing and validation.

## CI / GitHub Actions

A GitHub Actions workflow is included at `.github/workflows/ci.yml`. The workflow builds the project inside an NVIDIA CUDA Docker image and uploads artifacts. It runs on `push` and `pull_request` to `main`/`dev`.

## Contributing

- Make changes in a feature branch, run `make`, and add tests or smoke-tests if appropriate.
- Open a PR with a clear description and small, focused commits.

---
