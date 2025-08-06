#!/usr/bin/env python3
"""
Generate LaTeX tables for the appendix from the statistical significance of CSV files.
Handles method renaming and consistent ordering.

Usage:
  python generate_significance_latex_table.py \
    --input ./output/statistical_significance_hopper_400.csv \
    --output ./output/table_hopper_400.tex \
    [--caption "Statistical significance results for Hopper-v5"] \
    [--label "tab:hopper_significance"]
"""

import argparse
import pandas as pd
import numpy as np
import re


def extract_method_name(prefix):
    """Extract method name from prefix and return a clean name."""
    # Remove environment prefix
    method = prefix.split('__', 1)[1] if '__' in prefix else prefix
    
    # Method name mapping
    method_map = {
        'prefppo_duo_prio': 'Duo',
        'prefppo_random': 'Random',
        'prefppo_random_v2': 'Random',
        'prefppo_hybrid_prio': 'Hybrid',
        'prefppo_hybrid_v3': 'Hybrid',
        'prefppo_hybrid_prio_u_v6': 'Hybrid',
        'prefppo_variquery': 'VARIQuery',
        'prefppo_variquery_v3': 'VARIQuery',
        'prefppo_variquery_v10': 'VARIQuery',
    }
    
    return method_map.get(method, method)


def format_p_value(p_value):
    """Format p-value for LaTeX table."""
    if pd.isna(p_value) or p_value > 1:
        return "---"
    elif p_value < 0.001:
        return "$p < 0.001$"
    elif p_value < 0.01:
        return f"$p = {p_value:.3f}$"
    elif p_value < 0.05:
        return f"$p = {p_value:.3f}$"
    else:
        return f"$p = {p_value:.3f}$"


def format_effect_size(dz):
    """Format effect size for LaTeX table."""
    if pd.isna(dz):
        return "---"
    elif abs(dz) < 0.2:
        magnitude = "negligible"
    elif abs(dz) < 0.5:
        magnitude = "small"
    elif abs(dz) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return f"${dz:.3f}$ ({magnitude})"


def format_delta(delta_mean, delta_std):
    """Format delta with confidence interval."""
    if pd.isna(delta_mean) or pd.isna(delta_std):
        return "---"
    
    # Calculate 95% CI
    ci_low = delta_mean - 1.96 * delta_std
    ci_high = delta_mean + 1.96 * delta_std
    
    return f"${delta_mean:.1f}$ $[{ci_low:.1f}, {ci_high:.1f}]$"


def translate_metric_type(metric_type):
    """Translate metric type to display name."""
    translations = {
        'endpoint': 'Mean Final Reward',
        'auc': 'AUC'
    }
    return translations.get(metric_type, metric_type.title())


def generate_latex_table(df, caption="Statistical significance results", label="tab:significance"):
    """Generate LaTeX table from DataFrame."""
    
    # Define method order for consistent presentation
    method_order = ['Random', 'VARIQuery', 'Duo', 'Hybrid']
    
    # Extract and rename methods
    df = df.copy()
    df['method_a'] = df['prefix_a'].apply(extract_method_name)
    df['method_b'] = df['prefix_b'].apply(extract_method_name)
    
    # Filter to only include comparisons between methods in our order
    valid_methods = set(method_order)
    df = df[df['method_a'].isin(valid_methods) & df['method_b'].isin(valid_methods)]
    
    # Sort by metric_type, then by method order
    df['method_a_order'] = df['method_a'].apply(lambda x: method_order.index(x) if x in method_order else 999)
    df['method_b_order'] = df['method_b'].apply(lambda x: method_order.index(x) if x in method_order else 999)
    df = df.sort_values(['metric_type', 'method_a_order', 'method_b_order'])
    
    # Generate LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabularx}{\\textwidth}{@{}llXrr@{}}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Method A} & \\textbf{Method B} & \\textbf{p (Holm)} & \\textbf{Effect Size} & \\textbf{$\\Delta$ (95\\% CI)} \\\\")
    latex_lines.append("\\hline")
    
    current_metric = None
    for _, row in df.iterrows():
        metric_type = translate_metric_type(row['metric_type'])
        
        # Add metric type header if it's new
        if current_metric != metric_type:
            if current_metric is not None:
                latex_lines.append("\\hline")
            latex_lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{metric_type}}}}} \\\\")
            latex_lines.append("\\hline")
            current_metric = metric_type
        
        # Format the row
        method_a = row['method_a']
        method_b = row['method_b']
        p_value = format_p_value(row['p_value_holm'])
        effect_size = format_effect_size(row['dz'])
        delta = format_delta(row['delta_mean'], row['sd_delta'])
        
        latex_lines.append(f"{method_a} & {method_b} & {p_value} & {effect_size} & {delta} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabularx}")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from statistical significance CSV")
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output LaTeX file')
    parser.add_argument('--caption', default="Statistical significance results", help='Table caption')
    parser.add_argument('--label', help='Table label (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Read CSV
    df = pd.read_csv(args.input)
    
    # Auto-generate label if not provided
    if args.label is None:
        # Extract environment name from input filename
        import os
        filename = os.path.basename(args.input)
        if 'hopper' in filename.lower():
            env_name = 'hopper'
        elif 'walker' in filename.lower():
            env_name = 'walker'
        elif 'halfcheetah' in filename.lower():
            env_name = 'halfcheetah'
        elif 'ant' in filename.lower():
            env_name = 'ant'
        else:
            env_name = 'unknown'
        args.label = f"tab:appendix:significance-{env_name}"
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df, args.caption, args.label)
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write(latex_table)
    
    print(f"Generated LaTeX table: {args.output}")


if __name__ == '__main__':
    main() 