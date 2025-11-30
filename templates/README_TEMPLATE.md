# Example module: {MODULE_NAME}

Short description
-----------------
Provide a one-line description of what this example demonstrates (e.g. "Simple vector addition using CUDA kernels").

Prerequisites
-------------
- CUDA toolkit (nvcc) installed for local builds, or rely on CI container
- `make`, `gcc`/`g++`, and `libpng-dev` if this example uses PNG I/O

Build
-----
This project uses the top-level `Makefile` and per-directory Makefiles.

To build just this example:

```bash
make -C {MODULE_DIR}
```

Run / Usage
-----------
Each example binary supports flag-style CLI. Replace the placeholders below.

```bash
./{BINARY_NAME} --help
./{BINARY_NAME} --n 1024 --threads 128 --mode 0
```

If the directory includes `run.sh`, you can use that helper (there is a [`template`](run.sh.template) for it in this directory):

```bash
./run.sh
```

Profiling
---------
If you want to profile this example, use the [`provided profiling template`](profile_nvprof.sh.template) as a basis and make it executable. The project contains `profile_nvprof.sh` examples in some directories.

Example (after making the script executable):

```bash
./profile_nvprof.sh --n 4096 --threads 256
```

Notes / Expected output
-----------------------
Describe what users should expect to see (e.g., verification messages, runtime outputs, or produced files). Include any tolerances for numeric checks if applicable.

Placeholders to replace in this template
---------------------------------------
- `{MODULE_NAME}` — human-readable module name
- `{MODULE_DIR}` — relative directory path (e.g. `vector_addition`)
- `{BINARY_NAME}` — compiled binary filename (e.g. `vectAdd`)

Contact / Author if you want
----------------
Author: <your name or team>
Date: {DATE}
