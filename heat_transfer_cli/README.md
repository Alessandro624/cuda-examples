# Example module: Heat Transfer CLI

Short description
-----------------
Heat transfer simulation using a 2D grid with hot top and bottom boundaries. Includes both CPU reference implementation and CUDA GPU parallel versions using Unified Memory.

**CUDA Implementations:**
1. **Global** - Straightforward parallelization with CUDA Unified Memory
2. **Tiled** - Shared memory tiled parallelization without halo cells
3. **Tiled_wH** - Tiled parallelization with halo cells in shared memory

Prerequisites
-------------
- CUDA toolkit (nvcc) installed
- `g++` compiler with C++11 support
- `make`
- `gnuplot` (optional, for visualization)

Build
-----
This project uses the top-level `Makefile` and per-directory Makefiles.

To build all versions (CPU and CUDA):

```bash
make -C heat_transfer_cli
```

To build only CPU or CUDA version:

```bash
make -C heat_transfer_cli cpu   # CPU only
make -C heat_transfer_cli cuda  # CUDA only
```

Run / Usage
-----------

### CPU Version
```bash
./heat_transfer_cli
```

### CUDA Version
```bash
./heat_transfer_cuda [options]
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--mode M` | Kernel mode: 0=Global, 1=Tiled, 2=Tiled_wH | 0 |
| `--block BX BY` | Block dimensions | 16 16 |
| `--steps N` | Number of simulation steps | 10000 |
| `--rows R` | Grid rows | 256 |
| `--cols C` | Grid columns | 4096 |
| `--hotrows T B` | Hot rows at top and bottom | 2 2 |
| `--temp T` | Initial hot temperature | 20.0 |
| `--save` | Save initial and final configurations | off |
| `--verify` | Verify result against CPU reference | off |

**Supported Block Configurations:**
- 8x8, 8x16, 8x32
- 16x8, 16x16, 16x32
- 32x8, 32x16, 32x32

**Examples:**
```bash
# Global kernel with 16x16 blocks
./heat_transfer_cuda --mode 0 --block 16 16

# Tiled kernel with 32x8 blocks
./heat_transfer_cuda --mode 1 --block 32 8

# Tiled with halo, verification enabled, save output
./heat_transfer_cuda --mode 2 --block 16 32 --verify --save

# Custom grid size
./heat_transfer_cuda --mode 0 --rows 512 --cols 2048 --steps 5000
```

Or use the helper script:

```bash
./run.sh --mode 1 --block 16 16
BIN=heat_transfer_cli ./run.sh  # Run CPU version
```

Profiling
---------
Use the provided profiling script:

```bash
./profile_nvprof.sh --mode 0 --block 16 16
./profile_nvprof.sh --mode 1 --block 32 32
./profile_nvprof.sh --mode 2 --block 8 8 --verify
```

Visualization
-------------
To visualize the simulation results, use gnuplot:

```bash
gnuplot
plot 'temperature_step_0.dat' matrix with image       # Initial state
plot 'temperature_step_10000.dat' matrix with image   # Final state
quit
```

Algorithm
---------
The simulation uses a 9-point stencil for heat diffusion:
- Each cell's new temperature is computed as a weighted average of its 8 neighbors
- Direct neighbors (N, S, E, W) have weight 4/20
- Diagonal neighbors have weight 1/20

**Kernel Modes:**
- **Global (0)**: Each thread computes one cell, reads directly from global memory
- **Tiled (1)**: Tiles loaded to shared memory, boundary threads read halo from global memory
- **Tiled_wH (2)**: Full halo cells loaded to shared memory by boundary threads

Configuration (defaults)
------------------------
- `nSteps`: 10000 simulation steps
- `gridRows`: 256 rows
- `gridCols`: 4096 columns
- `nHotTopRows`: 2 hot rows at top
- `nHotBottomRows`: 2 hot rows at bottom
- `initialHotTemperature`: 20.0

Notes / Expected output
-----------------------
```
========================================
Heat Transfer CUDA Simulation
========================================
Mode:          Global
Block size:    16x16
Grid size:     256x4096
Steps:         10000
Hot rows:      top=2, bottom=2
Temperature:   20
========================================
CUDA grid:     256x16 blocks
CUDA block:    16x16 threads
========================================
Simulation in progress... 
Simulation loop elapsed time: X ms (corresponding to Y s)
```
