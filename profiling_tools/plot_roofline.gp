#!/usr/bin/env gnuplot
#
# Generic Roofline Plot Generator
# Automatically adapts to any number of versions/implementations
#

reset

# Check if output directory argument is provided, otherwise use current directory
if (!exists("OUTDIR")) OUTDIR = "."
if (!exists("PRECISION")) PRECISION = "FP32"

# Load GPU specifications
specs_file = OUTDIR."/roofline_specs.gp"
if (system("test -f ".specs_file) eq "") {
    load specs_file
} else {
    print "Warning: roofline_specs.gp not found, using defaults"
    bw_dram = 200.0
    bw_l2 = 2000.0
    bw_shared = 2000.0
    peak_flops = 200.0
}

data_file = OUTDIR."/roofline_data.dat"

# Check if data file exists
if (system("test -f ".data_file) ne "") {
    print "Error: ".data_file." not found"
    exit gnuplot
}

# --- Enhanced Visual Styling ---
set title sprintf("Roofline Model (%s)", PRECISION) font ",18" enhanced
set xlabel "Arithmetic Intensity (FLOP/Byte)" font ",13" offset 0,-0.5
set ylabel "Performance (GFLOP/s)" font ",13" offset -1,0
set logscale xy
set grid xtics ytics mxtics mytics lc rgb "#d0d0d0" lw 0.8 lt 1
set border 3 lw 1.5 lc rgb "#333333"
set tics nomirror
set key bottom right box opaque font ",10" samplen 2

# Fixed range for roofline (ensures complete lines are visible)
ai_min = 0.0001
ai_max = 10.0
perf_min = 0.01
perf_max = peak_flops * 1.6

set xrange [ai_min:ai_max]
set yrange [perf_min:perf_max]

# --- Roofline Functions ---
roof(bw, x) = (bw * x < peak_flops) ? bw * x : peak_flops

# --- Roofline line styles ---
set style line 10 lc rgb '#C0392B' lw 3.0 dt 1  # DRAM - solid red
set style line 11 lc rgb '#E74C3C' lw 2.5 dt 2  # L2 - dashed red
set style line 12 lc rgb '#F39C12' lw 2.0 dt 4  # Shared - dotted orange

# --- Color palette for data points (auto-cycling) ---
# Define a palette of distinct, colorblind-friendly colors
array colors[12] = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12', \
                    '#1ABC9C', '#E91E63', '#00BCD4', '#8BC34A', '#FF5722', \
                    '#673AB7', '#795548']

array ptypes[12] = [7, 5, 9, 11, 13, 4, 6, 8, 10, 12, 14, 3]  # Different point types

# Function to get color index (wraps around)
color_idx(n) = ((n - 1) % 12) + 1
ptype_idx(n) = ((n - 1) % 12) + 1

# --- Label positioning ---
angle = 35
lx_dram = 0.008
lx_l2 = 0.004
lx_shared = 0.003

# Set labels with rotation like original
set label 1 sprintf("DRAM: %.0f GB/s", bw_dram) at lx_dram, roof(bw_dram, lx_dram)*1.3 \
    tc rgb '#C0392B' rotate by angle font ",11" front

set label 2 sprintf("L2: %.0f GB/s", bw_l2) at lx_l2, roof(bw_l2, lx_l2)*1.3 \
    tc rgb '#E74C3C' rotate by angle font ",11" front

set label 3 sprintf("Shared: %.0f GB/s", bw_shared) at lx_shared, roof(bw_shared, lx_shared)*1.3 \
    tc rgb '#F39C12' rotate by angle font ",11" front

# Peak FLOPS label (horizontal portion)
set label 4 sprintf("Peak: %.0f GFLOP/s", peak_flops) at 1.0, peak_flops * 1.3 \
    tc rgb '#2C3E50' font ",12" front

# --- Plot settings ---
width_px = 1200
height_px = 800

# Build output filename (lowercase precision manually)
prec_lower = (PRECISION eq "FP32") ? "fp32" : (PRECISION eq "FP64") ? "fp64" : "fp16"

# PNG output with cairo for better quality
set terminal pngcairo size width_px,height_px enhanced font 'Arial,12' background rgb '#fafafa'
set output OUTDIR.'/roofline_'.prec_lower.'.png'

# --- Plot rooflines and data points ---
# Simple approach: plot all data points with same style, use labels for identification
plot roof(bw_dram, x) ls 10 title sprintf('DRAM (%.0f GB/s)', bw_dram), \
     roof(bw_l2, x) ls 11 title sprintf('L2 (%.0f GB/s)', bw_l2), \
     roof(bw_shared, x) ls 12 title sprintf('Shared (%.0f GB/s)', bw_shared), \
     data_file using 2:3 with points lc rgb '#3498DB' pt 7 ps 2.0 lw 2.5 title 'Kernels', \
     data_file using 2:($3*1.18):1 with labels center font ',10' tc rgb '#2C3E50' notitle

# SVG output with same styling
set terminal svg size width_px,height_px enhanced font 'Arial,12' background rgb '#fafafa'
set output OUTDIR.'/roofline_'.prec_lower.'.svg'
replot
