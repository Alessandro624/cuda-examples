#!/usr/bin/env python3
"""
GPU Profiling Metrics Parser
Parses NVIDIA profiler output (nvprof/ncu) and generates roofline data
"""

import os
import re
import csv
import glob
import statistics
import argparse
from pathlib import Path

# Default GPU specs (overridden by gpu_info.log if available)
DEFAULT_SPECS = {
    'bw_dram': 200.0,
    'bw_l2': 2000.0,
    'bw_shared': 2000.0,
    'peak_flops_fp32': 5000.0,
    'peak_flops_fp64': 200.0
}

# Transaction size (bytes per transaction)
TRANSACTION_SIZE = 32.0

def parse_gpu_info(log_file):
    """Parse GPU specifications from gpu_info.log"""
    specs = DEFAULT_SPECS.copy()
    
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found. Using default specs.")
        return specs
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Parse bandwidth values
    patterns = {
        'bw_dram': r'Global.*?read.*?:.*?([0-9\.]+)\s*GB/s',
        'bw_shared': r'Shared.*?read.*?:.*?([0-9\.]+)\s*GB/s',
        'bw_l2': r'(?:Texture|L2).*?read.*?:.*?([0-9\.]+)\s*GB/s',
        'peak_flops_fp32': r'Peak FP32.*?:.*?([0-9\.]+)\s*GFLOP/s',
        'peak_flops_fp64': r'Peak FP64.*?:.*?([0-9\.]+)\s*GFLOP/s',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            specs[key] = float(match.group(1))
    
    print(f"\nGPU Specifications:")
    print(f"  DRAM:    {specs['bw_dram']:.1f} GB/s")
    print(f"  L2:      {specs['bw_l2']:.1f} GB/s")
    print(f"  Shared:  {specs['bw_shared']:.1f} GB/s")
    print(f"  FP32:    {specs['peak_flops_fp32']:.1f} GFLOP/s")
    print(f"  FP64:    {specs['peak_flops_fp64']:.1f} GFLOP/s")
    
    return specs

def get_unit_multiplier(unit_str):
    """Convert time units to seconds"""
    if not unit_str:
        return 1.0
    u = unit_str.lower().strip()
    multipliers = {
        's': 1.0,
        'ms': 1e-3,
        'us': 1e-6,
        'ns': 1e-9
    }
    return multipliers.get(u, 1.0)

def read_csv_with_units(path):
    """Read CSV with optional unit row, handling nvprof multi-line format"""
    rows = []
    if not os.path.exists(path):
        return rows
    
    with open(path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    if not content.strip():
        return rows
    
    # Clean nvprof output: merge lines that are continuations
    # nvprof sometimes splits long kernel names across lines
    lines = []
    buffer = ""
    for line in content.split('\n'):
        stripped = line.strip()
        # Skip nvprof header lines
        if stripped.startswith('==') or not stripped:
            continue
        
        # Count quotes to detect incomplete lines
        buffer += line
        quote_count = buffer.count('"')
        
        if quote_count % 2 == 0:
            # Complete line
            lines.append(buffer.strip())
            buffer = ""
        else:
            # Incomplete line, continue buffering
            buffer += " "
    
    if buffer.strip():
        lines.append(buffer.strip())
    
    if not lines:
        return rows
    
    # Find header line
    header_idx = -1
    for i, line in enumerate(lines[:20]):
        if '"Type"' in line or '"Name"' in line or 'Name' in line or 'Kernel' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return rows
    
    # Parse header - handle quoted CSV properly
    header_line = lines[header_idx]
    keys = parse_csv_line(header_line)
    
    # Check for unit row
    unit_map = {}
    data_start_idx = header_idx + 1
    
    if len(lines) > header_idx + 1:
        next_line = lines[header_idx + 1]
        potential_units = parse_csv_line(next_line)
        
        # Check if this looks like a unit row (contains time units or %)
        if any(u in ['s', 'ms', 'us', 'ns', '%'] for u in potential_units):
            data_start_idx = header_idx + 2
            for i, u in enumerate(potential_units):
                if i < len(keys):
                    unit_map[keys[i]] = get_unit_multiplier(u)
    
    # Read data rows
    for line in lines[data_start_idx:]:
        if not line.strip():
            continue
        
        values = parse_csv_line(line)
        row = {}
        
        for i, val in enumerate(values):
            if i < len(keys):
                key = keys[i]
                # Apply unit conversion
                if key in unit_map and unit_map[key] != 1.0:
                    try:
                        val_clean = re.sub(r'[^0-9\.\-eE]', '', val)
                        if val_clean:
                            row[key] = float(val_clean) * unit_map[key]
                        else:
                            row[key] = val
                    except:
                        row[key] = val
                else:
                    row[key] = val
        
        if row:
            rows.append(row)
    
    return rows

def parse_csv_line(line):
    """Parse a CSV line handling quoted fields"""
    fields = []
    current = ""
    in_quotes = False
    
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            fields.append(current.strip().replace('"', ''))
            current = ""
        else:
            current += char
    
    fields.append(current.strip().replace('"', ''))
    return fields

def extract_kernel_name(full_name, kernel_filter=None):
    """Extract kernel name from full mangled name"""
    if not full_name:
        return None
    
    # Skip memory operations and API calls
    skip_patterns = ['[CUDA memcpy', '[CUDA memset', 'cudaLaunch', 'cudaMalloc', 'cudaFree']
    for skip in skip_patterns:
        if skip in full_name:
            return None
    
    # If filter provided, check if any filter matches
    if kernel_filter:
        for kf in kernel_filter:
            if kf.lower() in full_name.lower():
                return kf
        return None  # No match, skip this kernel
    
    # Extract kernel name from mangled C++ name
    # Pattern 1: "void kernel_name<...>(...)" - template kernels from nvprof
    match = re.search(r'\bvoid\s+(\w+)\s*<', full_name)
    if match:
        return match.group(1)
    
    # Pattern 2: "void kernel_name(...)" - simple kernels with void prefix
    match = re.search(r'\bvoid\s+(\w+)\s*\(', full_name)
    if match:
        return match.group(1)
    
    # Pattern 3: "kernel_name<template>(args)" - template without void
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*<', full_name)
    if match:
        return match.group(1)
    
    # Pattern 4: "kernel_name(args)" - simple kernel
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', full_name)
    if match:
        return match.group(1)
    
    # Pattern 5: Namespace::kernel_name
    match = re.search(r'(\w+)(?:<|::|\()', full_name)
    if match:
        name = match.group(1)
        # Skip if it's just "void"
        if name.lower() != 'void':
            return name
    
    # Fallback: first word that's not void
    words = re.findall(r'\b([a-zA-Z_]\w*)\b', full_name)
    for w in words:
        if w.lower() != 'void':
            return w
    
    return full_name[:50]  # Truncate long names

def clean_version_name(filename):
    """Generate clean version name from filename"""
    base = Path(filename).stem
    # Remove common prefixes
    for prefix in ['sciara_cuda_', 'sciara_', 'cuda_', 'app_']:
        if base.startswith(prefix):
            base = base[len(prefix):]
    
    # Replace underscores/hyphens with spaces and title case
    name = re.sub(r'[_\-]+', ' ', base).strip().title()
    
    # Special cases
    if name.lower() in ['cuda', 'global']:
        return 'Global'
    
    return name

def parse_dataset(base_path, kernel_filter=None):
    """Parse all CSV files for one executable"""
    data = {'kernels': {}}
    
    files = {
        'summary': base_path + '_summary.csv',
        'compute': base_path + '_compute.csv',
        'memory': base_path + '_memory.csv',
        'occupancy': base_path + '_occupancy.csv',
    }
    
    # Check which files exist
    existing = {k: v for k, v in files.items() if os.path.exists(v)}
    if not existing:
        print(f"  Warning: No CSV files found for {os.path.basename(base_path)}")
        return data
    
    # Read all CSV files
    csv_data = {k: read_csv_with_units(v) for k, v in existing.items()}
    
    kernels = {}
    
    def ensure_kernel(k):
        if k not in kernels:
            kernels[k] = {
                'total_flops': 0.0,
                'total_bytes': 0.0,
                'time_total_s': 0.0,
                'invocations': 0,
                'occupancies': []
            }
    
    # Parse summary (time and invocations)
    for row in csv_data.get('summary', []):
        # Handle different column names for kernel name
        name_val = row.get('Name', row.get('Kernel', ''))
        kname = extract_kernel_name(name_val, kernel_filter)
        if not kname:
            continue
        
        ensure_kernel(kname)
        
        # Time - check various field names and handle already-converted values
        time_val = None
        for time_field in ['Time', 'GPU Time', 'Duration', 'Avg']:
            if time_field in row and row[time_field]:
                try:
                    val = row[time_field]
                    if isinstance(val, (int, float)):
                        time_val = float(val)
                    else:
                        # Remove non-numeric chars and parse
                        val_clean = re.sub(r'[^0-9\.\-eE]', '', str(val))
                        if val_clean:
                            time_val = float(val_clean)
                    if time_val is not None and time_val > 0:
                        # Accumulate time for same kernel (multiple invocations)
                        kernels[kname]['time_total_s'] += time_val
                        break
                except Exception as e:
                    pass
        
        # Invocations
        for inv_field in ['Invocations', 'Calls']:
            if inv_field in row and row[inv_field]:
                try:
                    inv_str = str(row[inv_field])
                    inv_val = int(re.sub(r'[^0-9]', '', inv_str) or 1)
                    kernels[kname]['invocations'] += inv_val
                    break
                except:
                    pass
    
    # Parse compute metrics (FLOP counts)
    for row in csv_data.get('compute', []):
        kname = extract_kernel_name(row.get('Kernel', row.get('Name', '')), kernel_filter)
        if not kname:
            continue
        
        ensure_kernel(kname)
        
        metric = row.get('Metric Name', row.get('Metric', ''))
        val_str = str(row.get('Metric Value', row.get('Avg', row.get('Value', '0'))))
        
        try:
            val = float(re.sub(r'[^0-9\.]', '', val_str))
        except:
            val = 0.0
        
        # Accumulate FLOP counts
        if any(x in metric.lower() for x in ['flop', 'fma', 'fadd', 'fmul', 'dadd', 'dmul', 'dfma']):
            kernels[kname]['total_flops'] += val
    
    # Parse memory metrics
    for row in csv_data.get('memory', []):
        kname = extract_kernel_name(row.get('Kernel', row.get('Name', '')), kernel_filter)
        if not kname:
            continue
        
        ensure_kernel(kname)
        
        metric = row.get('Metric Name', row.get('Metric', ''))
        val_str = str(row.get('Metric Value', row.get('Avg', row.get('Value', '0'))))
        
        try:
            val = float(re.sub(r'[^0-9\.]', '', val_str))
        except:
            val = 0.0
        
        # Memory transactions
        if 'transaction' in metric.lower():
            kernels[kname]['total_bytes'] += val * TRANSACTION_SIZE
        elif 'bytes' in metric.lower():
            kernels[kname]['total_bytes'] += val
    
    # Parse occupancy
    for row in csv_data.get('occupancy', []):
        kname = extract_kernel_name(row.get('Kernel', row.get('Name', '')), kernel_filter)
        if not kname:
            continue
        
        ensure_kernel(kname)
        
        val_str = str(row.get('Metric Value', row.get('Avg', row.get('Value', '0'))))
        try:
            val = float(re.sub(r'[^0-9\.]', '', val_str))
            # Convert percentage to fraction if needed
            if val > 1.0:
                val /= 100.0
            kernels[kname]['occupancies'].append(val)
        except:
            pass
    
    data['kernels'] = kernels
    return data

def calculate_roofline_point(flops, bytes_accessed, time_s, name, kernel=""):
    """Calculate GFLOP/s and arithmetic intensity"""
    if time_s <= 0 or flops <= 0:
        print(f"  Warning [{name}{' - ' + kernel if kernel else ''}]: "
              f"Invalid data (time={time_s:.6f}s, flops={flops:.2e})")
        return None
    
    gflops = (flops / time_s) / 1e9
    ai = flops / max(1.0, bytes_accessed)
    
    print(f"  [{name}{' - ' + kernel if kernel else ''}]:")
    print(f"    FLOP:    {flops:.2e}")
    print(f"    Bytes:   {bytes_accessed:.2e}")
    print(f"    Time:    {time_s:.6f} s")
    print(f"    GFLOPS:  {gflops:.4f}")
    print(f"    AI:      {ai:.4f} FLOP/byte")
    
    return gflops, ai

def main():
    parser = argparse.ArgumentParser(description='Parse GPU profiling metrics')
    parser.add_argument('--results-dir', default='.', type=str,
                        help='Directory containing profiling results')
    parser.add_argument('--kernels', type=str,
                        help='Comma-separated list of kernel names to analyze')
    parser.add_argument('--precision', choices=['fp32', 'fp64'], default='fp64',
                        help='Precision for roofline plot')
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    kernel_filter = args.kernels.split(',') if args.kernels else None
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found")
        return 1
    
    # Parse GPU specs
    gpu_info_file = os.path.join(results_dir, 'gpu_info.log')
    specs = parse_gpu_info(gpu_info_file)
    
    # Choose precision
    peak_flops = specs['peak_flops_fp64'] if args.precision == 'fp64' else specs['peak_flops_fp32']
    
    # Write specs for gnuplot
    specs_file = os.path.join(results_dir, 'roofline_specs.gp')
    with open(specs_file, 'w') as f:
        f.write(f"bw_dram = {specs['bw_dram']}\n")
        f.write(f"bw_l2 = {specs['bw_l2']}\n")
        f.write(f"bw_shared = {specs['bw_shared']}\n")
        f.write(f"peak_flops = {peak_flops}\n")
    
    # Find all summary CSV files
    summary_files = glob.glob(os.path.join(results_dir, '*_summary.csv'))
    
    if not summary_files:
        print(f"Warning: No profiling CSV files found in {results_dir}")
        return 1
    
    roofline_data = []
    time_data = []
    occupancy_data = []
    
    print("\n" + "="*70)
    print("PROCESSING PROFILING DATA")
    print("="*70)
    
    for summary_file in summary_files:
        base_path = summary_file.replace('_summary.csv', '')
        exec_name = clean_version_name(base_path)
        
        print(f"\n--- {exec_name} ---")
        
        data = parse_dataset(base_path, kernel_filter)
        kernels = data['kernels']
        
        if not kernels:
            print(f"  No kernels found")
            continue
        
        # Generate data for EACH kernel separately
        for kernel_name, kdata in kernels.items():
            # Create label: "ExecName_KernelName" or just "KernelName" if single exec
            if len(summary_files) > 1:
                label = f"{exec_name}_{kernel_name}"
            else:
                label = kernel_name
            
            kernel_time = kdata['time_total_s']
            kernel_flops = kdata['total_flops']
            kernel_bytes = kdata['total_bytes']
            
            # Calculate roofline point for this kernel
            if kernel_flops > 0 and kernel_time > 0:
                result = calculate_roofline_point(kernel_flops, kernel_bytes, kernel_time, exec_name, kernel_name)
                if result:
                    gflops, ai = result
                    roofline_data.append({
                        'label': label,
                        'ai': ai,
                        'gflops': gflops
                    })
            
            # Time data per kernel
            time_data.append((label, kernel_time))
            
            # Occupancy per kernel
            avg_occupancy = statistics.mean(kdata['occupancies']) if kdata['occupancies'] else 0.0
            occupancy_data.append((label, avg_occupancy))
    
    # Write output files
    with open(os.path.join(results_dir, 'roofline_data.dat'), 'w') as f:
        f.write('# Label AI GFLOPS\n')
        for d in roofline_data:
            f.write(f'"{d["label"]}" {d["ai"]:.6f} {d["gflops"]:.4f}\n')
    
    with open(os.path.join(results_dir, 'time_data.dat'), 'w') as f:
        f.write('# Version Time_s\n')
        for name, time in sorted(time_data, key=lambda x: x[1], reverse=True):
            f.write(f'"{name}" {time:.6f}\n')
    
    with open(os.path.join(results_dir, 'occupancy_data.dat'), 'w') as f:
        f.write('# Version Occupancy\n')
        for name, occ in sorted(occupancy_data, key=lambda x: x[0]):
            f.write(f'"{name}" {occ:.6f}\n')
    
    print("\n" + "="*70)
    print("OUTPUT FILES WRITTEN")
    print("="*70)
    print(f"  {os.path.join(results_dir, 'roofline_data.dat')}")
    print(f"  {os.path.join(results_dir, 'time_data.dat')}")
    print(f"  {os.path.join(results_dir, 'occupancy_data.dat')}")
    print(f"  {specs_file}")
    
    return 0

if __name__ == '__main__':
    exit(main())
