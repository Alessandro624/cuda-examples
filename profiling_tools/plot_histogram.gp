#!/usr/bin/env gnuplot
#
# Generic Timing Histogram Generator
# Automatically adapts to data and creates performance comparison
#

reset

# Check if output directory argument is provided
if (!exists("OUTDIR")) OUTDIR = "."
if (!exists("TITLE_SUFFIX")) TITLE_SUFFIX = ""

data_file = OUTDIR."/time_data.dat"

# Check if data file exists
if (system("test -f ".data_file) ne "") {
    print "Error: ".data_file." not found"
    exit gnuplot
}

# Get statistics from data
stats data_file using 2 nooutput prefix "TIME"

# Calculate speedup relative to slowest
slowest_time = TIME_max
fastest_time = TIME_min
speedup_max = (fastest_time > 0) ? slowest_time / fastest_time : 1.0

# Count entries
n_entries = TIME_records

# Build title with speedup info
if (strlen(TITLE_SUFFIX) > 0) {
    plot_title = sprintf("Execution Time Comparison (%s)\nMax Speedup: %.2fx", TITLE_SUFFIX, speedup_max)
} else {
    plot_title = sprintf("Execution Time Comparison\nMax Speedup: %.2fx", speedup_max)
}

# --- Enhanced Visual Styling ---
set title plot_title font ",16" enhanced
set ylabel "Elapsed Time (s)" font ",13" offset -1,0
set xlabel "Implementation" font ",13" offset 0,-0.5

# Grid styling
set grid ytics lc rgb "#e0e0e0" lw 1 lt 1
set border 3 lw 1.5 lc rgb "#333333"
set tics nomirror

# Use boxes instead of histograms for better control
set style fill solid 0.85 border lc rgb "#333333"
set style data boxes
set boxwidth 0.4

# Set x range with padding for single entries
set xrange [-0.5:n_entries-0.5]

# Auto-adjust y range with padding for labels
ymax = TIME_max * 1.25
set yrange [0:ymax]

# Style for bars - modern color scheme
set style line 1 lc rgb '#5DA5DA' lt 1 lw 2  # Blue for others
set style line 2 lc rgb '#60BD68' lt 1 lw 2  # Green for best
set style line 3 lc rgb '#F17CB0' lt 1 lw 2  # Pink accent

# Adjust figure width based on number of entries
base_w = 1000
if (n_entries > 8) {
    width_px = base_w + (n_entries - 8) * 60
} else {
    width_px = base_w
}
height_px = 650

# Rotate x labels if many entries
if (n_entries > 6) {
    set xtics rotate by -35 right font ",10" offset 0,-0.3
} else {
    set xtics font ",11"
}

# Legend
set key top right box opaque font ",10"

# PNG output
set terminal pngcairo size width_px,height_px enhanced font 'Arial,12' background rgb '#fafafa'
set output OUTDIR.'/histogram_times.png'

# Plot with conditional coloring (highlight fastest in green)
plot data_file using 0:($2 == fastest_time ? $2 : 1/0):xtic(1) with boxes ls 2 title 'Best', \
     ''        using 0:($2 != fastest_time ? $2 : 1/0) with boxes ls 1 title 'Others', \
     ''        using 0:($2 + ymax*0.02):(sprintf("%.3f s", $2)) with labels center font ",9" tc rgb "#333333" notitle

# SVG output
set terminal svg size width_px,height_px enhanced font 'Arial,12' background rgb '#fafafa'
set output OUTDIR.'/histogram_times.svg'
replot
