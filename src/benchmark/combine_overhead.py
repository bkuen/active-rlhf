#!/usr/bin/env python3
"""
Combines multiple overhead_<ENV>_aggregated.csv files into a single LaTeX table.
Extracts environment name from filename and adds it as a column.

Usage:
    python combine_overhead.py --input-dir output/ --output output/combined_overhead_table.tex
    python combine_overhead.py --files output/overhead_walker_aggregated.csv output/overhead_hopper_aggregated.csv --output output/combined.tex
"""

import argparse
import os
import pandas as pd
import glob
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Combine multiple overhead aggregated CSV files into a single LaTeX table")
    parser.add_argument("--input-dir", help="Directory containing overhead_*_aggregated.csv files")
    parser.add_argument("--files", nargs="+", help="Specific CSV files to combine")
    parser.add_argument("--output", required=True, help="Output LaTeX file path")
    parser.add_argument("--environments", nargs="+", help="Environment names to use (in order of files)")
    return parser.parse_args()

def extract_environment_from_filename(filename):
    """Extract environment name from filename like 'overhead_walker_aggregated.csv' -> 'Walker'"""
    basename = os.path.basename(filename)
    # Match pattern: overhead_<env>_aggregated.csv
    match = re.search(r'overhead_([^_]+)_aggregated\.csv', basename)
    if match:
        env = match.group(1)
        # Capitalize first letter
        return env.capitalize()
    else:
        # Fallback: use filename without extension
        return os.path.splitext(basename)[0]

def load_and_process_csv(filepath, env_name=None):
    """Load CSV file and add environment column."""
    df = pd.read_csv(filepath)
    
    # Extract environment name if not provided
    if env_name is None:
        env_name = extract_environment_from_filename(filepath)
    
    # Add environment column
    df['Environment'] = env_name
    
    return df

def main():
    args = parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Get list of CSV files
    csv_files = []
    if args.input_dir:
        # Find all overhead_*_aggregated.csv files in directory
        pattern = os.path.join(args.input_dir, "overhead_*_aggregated.csv")
        csv_files = glob.glob(pattern)
        csv_files.sort()  # Sort for consistent ordering
    elif args.files:
        csv_files = args.files
    else:
        print("Error: Must specify either --input-dir or --files")
        return
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load and combine all CSV files
    combined_dfs = []
    for i, csv_file in enumerate(csv_files):
        env_name = None
        if args.environments and i < len(args.environments):
            env_name = args.environments[i]
        
        try:
            df = load_and_process_csv(csv_file, env_name)
            combined_dfs.append(df)
            print(f"Loaded {csv_file} -> {env_name or extract_environment_from_filename(csv_file)}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if not combined_dfs:
        print("No valid CSV files loaded!")
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    
    # Reorder columns to put Environment first
    column_order = ['Environment', 'method', 'wall_clock_h', 'gpu_h', 'cpu_h', 'peak_ram_gb', 'wall_clock_min']
    available_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    print(f"\nCombined data shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Generate LaTeX table
    generate_combined_latex_table(combined_df, args.output)
    
    # Also save as CSV for reference
    csv_output = args.output.replace('.tex', '.csv')
    combined_df.to_csv(csv_output, index=False)
    print(f"📊 Combined data saved to: {csv_output}")

def generate_combined_latex_table(df, output_path):
    """Generate LaTeX table from combined DataFrame."""
    
    latex_table = r"""\begin{table}[h]
  \centering
  \begin{tabular}{lllll}
    \textbf{Method} & \textbf{Wall-clock min} & \textbf{GPU h} & \textbf{CPU h} & \textbf{Peak RAM (GB)} \\
    """
    
    # Custom environment ordering
    env_order = ['Walker', 'Halfcheetah', 'Ant', 'Hopper']
    
    # Get all environments and sort according to custom order
    all_environments = df['Environment'].unique()
    print(f"Available environments: {list(all_environments)}")
    
    environments = []
    
    # Add environments in custom order
    for env in env_order:
        if env in all_environments:
            environments.append(env)
            print(f"Added {env} to order")
    
    # Add any remaining environments not in the custom order
    for env in sorted(all_environments):
        if env not in environments:
            environments.append(env)
            print(f"Added remaining {env} to order")
    
    print(f"Final environment order: {environments}")
    
    # Custom method ordering
    method_order = ['Random', 'VARIQuery', 'DUO', 'Hybrid']
    
    for i, env in enumerate(environments):
        env_data = df[df['Environment'] == env]
        
        # Get Random baseline values for this environment
        random_data = env_data[env_data['method'] == 'Random']
        random_values = {}
        if not random_data.empty:
            random_row = random_data.iloc[0]
            random_values = {
                'wall_clock_h': random_row['wall_clock_h'],
                'gpu_h': random_row['gpu_h'],
                'cpu_h': random_row['cpu_h'],
                'peak_ram_gb': random_row['peak_ram_gb']
            }
        
        # Add environment header row
        if i > 0:  # Add empty row before each environment (except first)
            latex_table += r"\n    "
        
        # Add environment name as header
        latex_table += f"\n    \\emph{{\\underline{{{env}}}}} & & & & \\\\"
        
        # Add methods for this environment in custom order
        for method in method_order:
            method_data = env_data[env_data['method'] == method]
            
            if not method_data.empty:
                row = method_data.iloc[0]
                
                # Calculate percentage increase compared to Random (only for non-Random methods)
                def calc_percentage_increase(current, baseline, metric_name):
                    if method == 'Random':  # Don't show percentage for Random
                        return ""
                    
                    if pd.isna(current) or pd.isna(baseline) or baseline == 0:
                        return ""
                    
                    if metric_name in random_values and pd.notna(random_values[metric_name]):
                        baseline_val = random_values[metric_name]
                        if baseline_val > 0:
                            pct_increase = ((current - baseline_val) / baseline_val) * 100
                            return f" ({pct_increase:+.1f}\\%)"
                    return ""
                
                wall_clock_h = f"{row['wall_clock_h']:.2f}" if pd.notna(row["wall_clock_h"]) else r"\textit{TBD}"
                wall_clock_min = f"{row['wall_clock_h']*60:.0f}" if pd.notna(row["wall_clock_h"]) else r"\textit{TBD}"
                gpu_h = f"{row['gpu_h']:.2f}" if pd.notna(row["gpu_h"]) else r"\textit{TBD}"
                cpu_h = f"{row['cpu_h']:.2f}" if pd.notna(row["cpu_h"]) else r"\textit{TBD}"
                peak_ram = f"{row['peak_ram_gb']:.1f}" if pd.notna(row["peak_ram_gb"]) else r"\textit{TBD}"
                
                # Add percentage increases (only for non-Random methods)
                wall_clock_pct = calc_percentage_increase(row['wall_clock_h'], random_values.get('wall_clock_h'), 'wall_clock_h')
                gpu_pct = calc_percentage_increase(row['gpu_h'], random_values.get('gpu_h'), 'gpu_h')
                cpu_pct = calc_percentage_increase(row['cpu_h'], random_values.get('cpu_h'), 'cpu_h')
                ram_pct = calc_percentage_increase(row['peak_ram_gb'], random_values.get('peak_ram_gb'), 'peak_ram_gb')
                
                # Format wall-clock time as "minutes [pct]"
                wall_clock_str = f"{wall_clock_min}{wall_clock_pct}"
                gpu_str = f"{gpu_h}{gpu_pct}"
                cpu_str = f"{cpu_h}{cpu_pct}"
                ram_str = f"{peak_ram}{ram_pct}"
                
                latex_table += f"\n    \\hspace*{{2mm}}{method:<10} & {wall_clock_str:<20} & {gpu_str:<12} & {cpu_str:<12} & {ram_str:<18} \\\\"
    
    latex_table += r"""
  \end{tabular}
  \caption{Computational cost summary across environments. Wall-clock time is shown in minutes. GPU h and CPU h are utilization-weighted device-hours. Peak RAM is the maximum resident set size observed. Percentage increases relative to Random baseline are shown in parentheses.}
  \label{tab:results:performance-comparisons:computational-overhead}
\end{table}"""
    
    with open(output_path, "w") as f:
        f.write(latex_table)
    
    print(f"📄 Combined LaTeX table saved to: {output_path}")

if __name__ == "__main__":
    main() 