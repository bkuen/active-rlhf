#!/usr/bin/env python3
"""
generate_learning_curves.py

Reads a CSV of episodic returns and plots the mean cumulative reward over time
for each method, along with a shaded band indicating variability (std or SEM).
Includes optional smoothing, method renaming, and fixed colors per method.

Input CSV should have columns:
  exp_name, seed, charts/episodic_return, global_step

Usage:
    python3 generate_learning_curves.py \
      --input runs.csv \
      --output learning_curves.png \
      [--ci std|sem] [--smooth WINDOW]
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

# Map raw prefixes to display names
METHOD_NAME_MAP = {
    'duo': 'DUO',
    'duo_prio': 'DUO',
    'random': 'Random',
    'hybrid': 'Hybrid',
    'hybrid_v': 'Hybrid',
    'hybrid_prio': 'Hybrid',
    'hybrid_prio_u_v': 'Hybrid',
    'variquery': 'VARIQuery',
    'variquery_v': 'VARIQuery'
}
# Fixed colors for consistency across environments
COLOR_MAP = {
    'duo':              'orange',
    'duo_prio':         'orange',
    'hybrid':           'red',
    'hybrid_v':         'red',
    'hybrid_prio':      'red',
    'hybrid_prio_u_v':  'red',
    'random':           'blue',
    'variquery':        'green',
    'variquery_v':      'green',
}

def tint(color, amount=0.80, bg="#FFFFFF"):
    """
    Blend `color` toward `bg` (white by default) to make a soft fill color.
    amount=0.70 means 70% of bg + 30% of color.
    """
    rgb = np.array(mcolors.to_rgb(color))
    bg_rgb = np.array(mcolors.to_rgb(bg))
    return mcolors.to_hex((1 - amount) * rgb + amount * bg_rgb)

def extract_method(exp_name: str) -> str:
    """Extract and normalize method name from exp_name."""
    prefix = exp_name.rsplit('__', 2)[0]
    m = re.search(r'prefppo_([A-Za-z_]+)', prefix)
    if m:
        code = m.group(1).lower()
        return code
    return prefix


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves (mean ± variability) for RL runs."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input CSV file")
    parser.add_argument("-o", "--output", default=None,
                        help="Path to save the plot (e.g. .png). If omitted, shows interactively.")
    parser.add_argument("--ci", choices=["std", "sem"], default="std",
                        help="Type of variability band: 'std' or 'sem'.")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Window size for moving-average smoothing (in steps). Default=1 (no smoothing).")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    df['method'] = df['exp_name'].astype(str).apply(extract_method)

    steps = sorted(df['global_step'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, group in df.groupby('method'):
        pivot = group.pivot(index='global_step', columns='seed', values='charts/episodic_return')
        pivot = pivot.reindex(steps)

        if args.smooth > 1:
            mean = pivot.mean(axis=1).rolling(window=args.smooth, center=True, min_periods=1).mean()
            band = pivot.std(axis=1).rolling(window=args.smooth, center=True, min_periods=1).mean() if args.ci=='std' \
                   else pivot.sem(axis=1).rolling(window=args.smooth, center=True, min_periods=1).mean()
        else:
            mean = pivot.mean(axis=1)
            band = pivot.std(axis=1) if args.ci=='std' else pivot.sem(axis=1)

        print("Method:", method)

        line_color = COLOR_MAP.get(method, None)
        fill_color = tint(line_color or "#1f77b4", amount=0.70)  # lightened fill


        ax.fill_between(steps, mean - band, mean + band,
                        color=fill_color,
                        edgecolor=None,
                        alpha=0.45,
                        zorder=1)

        display_name = METHOD_NAME_MAP.get(method, method)
        ax.plot(steps, mean, label=display_name, color=line_color, zorder=3)

    ax.set_xlabel('Global Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Learning Curves')
    ax.legend(title='Method')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
