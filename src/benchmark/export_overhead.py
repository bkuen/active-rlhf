#!/usr/bin/env python3
"""
export_overhead.py

A self-contained Python script that exports computational-overhead metrics from W&B runs.

Usage:
    python export_overhead.py --project <entity/project> --output <output_file> [--include PREFIX] [--include PREFIX2] ...

Examples:
    python export_overhead.py --project bkuen-ludwig-maximilianuniversity-of-munich/thesis-benchmark-walker --output output/overhead.csv
    python export_overhead.py --project entity/project --output output.csv --include prefppo_random__ --include prefppo_duo_prio__
"""

import argparse
import os
import pandas as pd
import numpy as np
from wandb import Api
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Export computational overhead metrics from W&B runs")
    parser.add_argument("--project", help="W&B project in format 'entity/project'")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--include", action="append", default=[], 
                       help="Include runs with names containing this substring (can be used multiple times)")
    parser.add_argument("--exclude", action="append", default=[], 
                       help="Exclude runs with names containing this substring (can be used multiple times)")
    parser.add_argument("--method-mapping", nargs=2, action="append", default=[],
                       help="Map run name prefix to method label: --method-mapping PREFIX LABEL")
    return parser.parse_args()

def extract_method_name(run_name, method_mappings):
    """Extract method name from run name using provided mappings."""
    for prefix, label in method_mappings:
        if prefix in run_name:
            return label
    # Fallback: try to extract from common patterns
    if "random" in run_name.lower():
        return "Random"
    elif "variquery" in run_name.lower():
        return "VARIQuery"
    elif "duo" in run_name.lower():
        return "DUO"
    elif "hybrid" in run_name.lower():
        return "Hybrid"
    else:
        return "Unknown"

def main():
    args = parse_args()
    
    # Parse entity/project
    if "/" not in args.project:
        print("Error: Project must be in format 'entity/project'")
        sys.exit(1)
    
    entity, project = args.project.split("/", 1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize W&B API
    api = Api()
    runs = api.runs(f"{entity}/{project}")
    
    print(f"Processing {len(runs)} runs from {entity}/{project}")
    
    rows = []
    processed_count = 0
    
    for run in runs:
        # Apply filters
        if args.include and not any(include in run.name for include in args.include):
            continue
        if args.exclude and any(exclude in run.name for exclude in args.exclude):
            continue
        
        print(f"Processing run: {run.name}")
        
        try:
            # Get system metrics
            system_df = run.history(stream="events", pandas=True)
            if system_df.empty:
                print(f"  ⚠️  No system metrics, skipping")
                continue

            # Sort by runtime, not by wall-clock timestamp
            system_df = system_df.sort_values('_runtime')
            
            # Compute exact time-deltas in seconds
            system_df['dt_s'] = system_df['_runtime'].diff().fillna(0)
            
            # Extract method name
            method_mappings = dict(args.method_mapping) if args.method_mapping else {}
            method = extract_method_name(run.name, method_mappings.items())
            
            # Compute wall-clock hours
            wall_clock_h = system_df["_runtime"].max() / 3600.0
            
            # Compute CPU hours (utilization-based)
            if "system.cpu" in system_df.columns:
                # Fill-forward (then back-fill) CPU util so every dt_s has a value
                system_df['cpu_util'] = (system_df['system.cpu']
                                        .ffill()     # carry last known util forward
                                        .bfill()     # if there's no prior value, carry the next one backward
                                        .fillna(0))  # if it was *always* missing, assume zero
                
                # Integrate busy-seconds, then convert to hours
                system_df['cpu_busy_s'] = (system_df['cpu_util'] / 100) * system_df['dt_s']
                cpu_h = system_df['cpu_busy_s'].sum() / 3600
            else:
                cpu_h = np.nan
            
            # Compute GPU hours (utilization-based using system.gpu.0.gpu)
            if "system.gpu.0.gpu" in system_df.columns:
                # Fill-forward (then back-fill) GPU util so every dt_s has a value
                system_df['gpu_util'] = (system_df['system.gpu.0.gpu']
                                        .ffill()     # carry last known util forward
                                        .bfill()     # if there's no prior value, carry the next one backward
                                        .fillna(0))  # if it was *always* missing, assume zero
                
                # Integrate busy-seconds, then convert to hours
                system_df['gpu_busy_s'] = (system_df['gpu_util'] / 100) * system_df['dt_s']
                gpu_h = system_df['gpu_busy_s'].sum() / 3600
            else:
                gpu_h = np.nan
            
            # Compute peak RAM in GB
            if "system.proc.memory.rssMB" in system_df.columns:
                peak_ram_gb = system_df["system.proc.memory.rssMB"].max() / 1024.0
            else:
                peak_ram_gb = np.nan
            
            # Extract additional metadata
            config = run.config
            env = config.get("env", "unknown")
            seed = config.get("seed", np.nan)
            budget = config.get("budget", np.nan)
            
            rows.append({
                "run_name": run.name,
                "method": method,
                "env": env,
                "seed": seed,
                "budget": budget,
                "wall_clock_h": wall_clock_h,
                "wall_clock_min": wall_clock_h * 60,  # Convert to minutes
                "cpu_h": cpu_h,
                "gpu_h": gpu_h,
                "peak_ram_gb": peak_ram_gb
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {run.name}: {e}")
            continue
    
    if not rows:
        print("No runs found matching the criteria!")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n✅ Processed {processed_count} runs")
    print(f"📁 Saved results to: {args.output}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    if "method" in df.columns:
        method_summary = df.groupby("method").agg({
            "wall_clock_h": ["mean", "std"],
            "cpu_h": ["mean", "std"],
            "gpu_h": ["mean", "std"],
            "peak_ram_gb": ["mean", "std"]
        }).round(3)
        print(method_summary)
    
    # Generate LaTeX table if requested
    if args.output.endswith('.csv'):
        tex_output = args.output.replace('.csv', '_table.tex')
        generate_latex_table(df, tex_output)
        print(f"📄 LaTeX table saved to: {tex_output}")
        
        # Save aggregated DataFrame for later combination
        if "method" in df.columns:
            aggregated_df = df.groupby("method").agg({
                "wall_clock_h": "mean",
                "gpu_h": "mean",
                "cpu_h": "mean", 
                "peak_ram_gb": "mean"
            }).reset_index()
            
            # Add wall_clock_min column
            aggregated_df["wall_clock_min"] = aggregated_df["wall_clock_h"] * 60
            
            # Save aggregated data
            aggregated_output = args.output.replace('.csv', '_aggregated.csv')
            aggregated_df.to_csv(aggregated_output, index=False)
            print(f"📊 Aggregated data saved to: {aggregated_output}")

def generate_latex_table(df, output_path):
    """Generate LaTeX table from the DataFrame."""
    # Group by method and compute means
    if "method" in df.columns:
        summary_df = df.groupby("method").agg({
            "wall_clock_h": "mean",
            "gpu_h": "mean",
            "cpu_h": "mean", 
            "peak_ram_gb": "mean"
        }).reset_index()
        
        latex_table = r"""\begin{table}[h]
  \centering
  \begin{tabular}{lrrrr}
    \toprule
    \textbf{Method} & \textbf{Wall-clock h (min)} & \textbf{GPU h} & \textbf{CPU h} & \textbf{Peak RAM (GB)} \\
    \midrule"""
        
        for _, row in summary_df.iterrows():
            method = row["method"]
            wall_clock_h = f"{row['wall_clock_h']:.2f}" if pd.notna(row["wall_clock_h"]) else r"\textit{TBD}"
            wall_clock_min = f"{row['wall_clock_h']*60:.0f}" if pd.notna(row["wall_clock_h"]) else r"\textit{TBD}"
            gpu_h = f"{row['gpu_h']:.2f}" if pd.notna(row["gpu_h"]) else r"\textit{TBD}"
            cpu_h = f"{row['cpu_h']:.2f}" if pd.notna(row["cpu_h"]) else r"\textit{TBD}"
            peak_ram = f"{row['peak_ram_gb']:.1f}" if pd.notna(row["peak_ram_gb"]) else r"\textit{TBD}"
            
            # Format wall-clock time as "hours (minutes)"
            wall_clock_str = f"{wall_clock_h} ({wall_clock_min})"
            
            latex_table += f"\n    {method:<10} & {wall_clock_str:<15} & {gpu_h:<6} & {cpu_h:<6} & {peak_ram:<14} \\\\"
        
        latex_table += r"""
    \bottomrule
  \end{tabular}
  \caption{Computational cost summary. Wall-clock time is shown in hours, with minutes in parentheses. GPU h and CPU h are utilization-weighted device-hours. Peak RAM is the maximum resident set size observed.}
  \label{tab:results:overhead}
\end{table}"""
        
        with open(output_path, "w") as f:
            f.write(latex_table)

if __name__ == "__main__":
    main()
