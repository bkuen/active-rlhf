#!/usr/bin/env python3
"""
Fetch the top runs (by count or percent) from WandB by final episodic return (optionally filtering by sweep),
compute recommended hyperparameter values for all sweep variables (based on median for numeric or mode for categorical),
and output a config, along with histograms of important knobs for only the sweep variables.

Usage:
    python select_hparams_with_args.py \
        --entity bkuen-ludwig-maximilianuniversity-of-munich \
        --project bench-halfcheetah \
        --sweep-config sweep.yaml \
        [--sweep-id SWEEP_ID] \
        [--top-k 20] \
        [--top-pct 0.1] \
        [--var-thresh 1e-6] \
        [--output-dir output]

Note: If both --top-k and --top-pct are provided, --top-pct takes precedence.
"""
import argparse
import os
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from collections import Counter

def load_sweep_vars(sweep_path):
    """
    Load sweep configuration and return list of parameter names without a fixed 'value' field.
    Also return categorical variables (those with 'values' field).
    """
    with open(sweep_path, 'r') as f:
        config = yaml.safe_load(f)
    params = config.get('parameters', {})
    sweep_vars = [k for k,v in params.items() if 'value' not in v]
    categorical_vars = [k for k,v in params.items() if 'value' not in v and 'values' in v]
    return sweep_vars, categorical_vars


def main():
    parser = argparse.ArgumentParser(
        description="Select hyperparameters from WandB runs and plot histograms for important knobs (sweep vars only)."
    )
    parser.add_argument("--entity", type=str, required=True, help="W&B entity/username")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--sweep-config", type=str, required=True, help="Path to W&B sweep YAML configuration")
    parser.add_argument("--sweep-id", type=str, default=None, help="W&B sweep ID to filter runs")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top runs to select")
    parser.add_argument("--top-pct", type=float, default=None, help="Fraction of total runs to select (e.g., 0.1 for top 10%)")
    parser.add_argument("--var-thresh", type=float, default=1e-6, help="Variance threshold for marking knobs as important")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save histogram PNGs")
    args = parser.parse_args()

    # prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # load sweep variable names
    sweep_vars, categorical_vars = load_sweep_vars(args.sweep_config)
    print(f"Sweep variables: {sweep_vars}")
    print(f"Categorical variables: {categorical_vars}")

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"

    # fetch run metadata only
    filters = {"sweep": args.sweep_id} if args.sweep_id else None
    all_runs = api.runs(path, filters=filters) if filters else api.runs(path)
    print(f"Fetched metadata for {len(all_runs)} runs from {path}.")

    # collect summary returns
    runs_with_return = [(run, float(run.summary.get("charts/episodic_return")))
                        for run in all_runs if run.summary.get("charts/episodic_return") is not None]
    if not runs_with_return:
        raise RuntimeError("No runs with summary 'charts/episodic_return'.")

    # sort and determine selection count
    runs_with_return.sort(key=lambda x: x[1], reverse=True)
    if args.top_pct:
        if not (0 < args.top_pct <= 1): raise ValueError("--top-pct must be in (0,1]")
        k = max(1, int(len(runs_with_return) * args.top_pct))
        print(f"Selecting top {args.top_pct*100:.1f}% => {k} runs")
    else:
        k = args.top_k
        print(f"Selecting top {k} runs")
    selected_runs = [r for r,_ in runs_with_return[:k]]

    # gather configs & final returns
    records = []
    for run in selected_runs:
        history = run.scan_history(keys=["charts/episodic_return"])
        returns = [h["charts/episodic_return"] for h in history if h.get("charts/episodic_return") is not None]
        final_ret = returns[-1] if returns else float(run.summary["charts/episodic_return"])
        cfg = run.config
        entry = {k: cfg.get(k) for k in sweep_vars}
        entry['final_return'] = final_ret
        records.append(entry)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No config records found.")

    # numeric variables median/variance
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['final_return'], errors='ignore')
    medians = numeric_df.median()
    variances = numeric_df.var()

    # identify important numeric knobs
    important_numeric = [hp for hp in medians.index if variances[hp] > args.var_thresh]

    # plot histograms only for important numeric sweep vars
    for hp in important_numeric:
        plt.figure()
        plt.hist(df[hp], bins=min(10, len(df)), edgecolor='k')
        plt.axvline(medians[hp], color='r', linestyle='--')
        plt.title(f"{hp} (var={variances[hp]:.3g})")
        out = os.path.join(args.output_dir, f"hist_{hp}.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved histogram for sweep var {hp}: {out}")

    # build recommended config
    recommended = {}
    # numeric: median
    for hp in medians.index:
        recommended[hp] = float(medians[hp])
    # categorical: mode
    for hp in categorical_vars:
        mode = Counter(df[hp]).most_common(1)[0][0]
        recommended[hp] = mode

    # output
    print("\n===== RECOMMENDED HYPERPARAMETERS =====")
    for k,v in recommended.items():
        print(f"{k}: {v}")
    print("======================================\n")

if __name__ == '__main__':
    main()
