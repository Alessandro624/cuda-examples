#!/usr/bin/env gnuplot
#
# Generic GPU Occupancy Plot Generator
# Automatically adapts to any number of implementations
#

reset

# Check if output directory argument is provided
if (!exists("OUTDIR")) OUTDIR = "."

data_file = OUTDIR."/occupancy_data.dat"

# Check if data file exists
if (system("test -f ".data_file) ne "") {
    print "Error: ".data_file." not found"
    exit gnuplot
}

# Get statistics
stats data_file using 2 nooutput prefix "OCC"

# Count entries for x-axis formatting
n_entries = OCC_records
avg_occ = OCC_mean

set title sprintf("GPU SM Occupancy Comparison\nAverage: %.1f%%", avg_occ * 100) font ",16" enhanced
set xlabel "Implementation" font ",13" offset 0,-0.5
set ylabel "Occupancy (%)" font ",13" offset -1,0

# --- Enhanced Visual Styling ---
set border 3 lw 1.5 lc rgb "#333333"
set tics nomirror

set style fill solid 0.80 border lc rgb "#444444"

# Adjust figure width based on number of entries
base_w = 1000
if (n_entries > 8) {
    width_px = base_w + (n_entries - 8) * 70
} else {
    width_px = base_w
}
height_px = 650

# Adjust xtics font/rotation if many entries
if (n_entries > 6) {
    set xtics rotate by -35 right font ",10" offset 0,-0.3
} else {
    set xtics font ",11"
}

# Format y-axis as percentage
set ytics format "%.0f%%" font ",10"
set ytics 10

# Color bars by occupancy level (improved colors)
# Red: < 30%, Orange: 30-50%, Yellow: 50-70%, Green: > 70%
set style line 1 lc rgb '#E74C3C' lt 1 lw 2  # Poor (red)
set style line 2 lc rgb '#E67E22' lt 1 lw 2  # Low-Medium (orange)
set style line 3 lc rgb '#F1C40F' lt 1 lw 2  # Medium (yellow)
set style line 4 lc rgb '#27AE60' lt 1 lw 2  # Good (green)

# Add reference lines for occupancy levels with better styling
set arrow 1 from graph 0, first 30 to graph 1, first 30 nohead dt 4 lc rgb "#E74C3C" lw 1.2
set arrow 2 from graph 0, first 50 to graph 1, first 50 nohead dt 4 lc rgb "#E67E22" lw 1.2
set arrow 3 from graph 0, first 70 to graph 1, first 70 nohead dt 4 lc rgb "#F1C40F" lw 1.2

# Add labels for occupancy zones (positioned on right side)
set label 1 "Poor (<30%)" at graph 0.98, first 15 right font ",9" tc rgb "#E74C3C"
set label 2 "Low (30-50%)" at graph 0.98, first 40 right font ",9" tc rgb "#E67E22"
set label 3 "Medium (50-70%)" at graph 0.98, first 60 right font ",9" tc rgb "#F1C40F"
set label 4 "Good (>70%)" at graph 0.98, first 85 right font ",9" tc rgb "#27AE60"

# Box appearance
set style data boxes
set boxwidth 0.4
set xrange [-0.5:n_entries-0.5]
set grid ytics lc rgb "#e0e0e0" lw 1 lt 1
set yrange [0:105]

# Legend
set key top left box opaque font ",9" samplen 1.5

# PNG output (width adapts to number of entries)
set terminal pngcairo size width_px,height_px enhanced font 'Arial,12' background rgb '#fafafa'
set output OUTDIR.'/occupancy.png'

# Plot with 4-tier coloring (multiply by 100 for percentage display)
plot data_file using 0:($2*100 < 30 ? $2*100 : 1/0):xtic(1) with boxes ls 1 title 'Poor', \
     ''        using 0:($2*100 >= 30 && $2*100 < 50 ? $2*100 : 1/0) with boxes ls 2 title 'Low', \
     ''        using 0:($2*100 >= 50 && $2*100 < 70 ? $2*100 : 1/0) with boxes ls 3 title 'Medium', \
     ''        using 0:($2*100 >= 70 ? $2*100 : 1/0) with boxes ls 4 title 'Good', \
     ''        using 0:($2*100 + 2):(sprintf("%.0f%%", $2*100)) with labels center font ",9" tc rgb "#333333" notitle

# SVG output (same width)
set terminal svg size width_px,height_px enhanced font 'Arial,12' background rgb '#fafafa'
set output OUTDIR.'/occupancy.svg'
replot
