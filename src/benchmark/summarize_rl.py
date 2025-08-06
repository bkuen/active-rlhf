#!/usr/bin/env python3
"""
summarize_rl.py

Compute endpoint statistics (mean, median, min, max, IQR, std, SEM, 95% CI, CV)
for each experiment prefix given a CSV with columns:
  exp_name, seed, charts/episodic_return, global_step

Endpoint (per-seed) is configurable: last, tail_mean (K), ema (span),
best (cummax), or mean over the last X steps.

Usage:
  python3 summarize_rl.py -i runs.csv -o summary.csv \
    --endpoint tail_mean --tail-k 10
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats

def extract_prefix(s: str) -> str:
    return s.rsplit("__", 2)[0]

def per_seed_endpoints(df: pd.DataFrame, endpoint: str,
                       tail_k: int, ema_span: int, last_steps: int) -> pd.DataFrame:
    # rename for convenience
    df = df.rename(columns={"charts/episodic_return": "return"}).copy()
    df["prefix"] = df["exp_name"].astype(str).apply(extract_prefix)
    df = df.sort_values(["prefix", "seed", "global_step"])

    endpoints = []
    for (p, seed), g in df.groupby(["prefix", "seed"]):
        if endpoint == "last":
            val = g["return"].iloc[-1]
        elif endpoint == "tail_mean":
            val = g["return"].tail(tail_k).mean()
        elif endpoint == "ema":
            val = g["return"].ewm(span=ema_span, adjust=False).mean().iloc[-1]
        elif endpoint == "best":
            val = g["return"].cummax().iloc[-1]
        elif endpoint == "last_steps":
            max_step = g["global_step"].max()
            win = g[g["global_step"] >= max_step - last_steps]
            val = win["return"].mean()
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        endpoints.append({"prefix": p, "seed": seed, "endpoint": val})
    return pd.DataFrame(endpoints)

def compute_summary_from_endpoints(ep: pd.DataFrame) -> pd.DataFrame:
    # n seeds
    counts = ep.groupby("prefix")["endpoint"].count().rename("n_seeds")
    # basic stats
    agg = ep.groupby("prefix")["endpoint"].agg(
        mean="mean", median="median", min="min", max="max", std="std"
    )

    # IQR
    q1 = ep.groupby("prefix")["endpoint"].quantile(0.25)
    q3 = ep.groupby("prefix")["endpoint"].quantile(0.75)
    agg["IQR"] = (q3 - q1)

    # SEM, CI
    agg["sem"] = agg["std"] / np.sqrt(counts)
    dof = counts - 1
    t95 = stats.t.ppf(0.975, df=dof)
    # t95 = t95.where(dof > 0, np.nan)  # avoid invalid for 1 seed
    ci_half = agg["sem"] * t95
    agg["ci_lower"] = agg["mean"] - ci_half
    agg["ci_upper"] = agg["mean"] + ci_half

    # Coefficient of variation (non-negative)
    agg["CV"] = agg["std"] / agg["mean"].abs()

    summary = agg.merge(counts, left_index=True, right_index=True).reset_index()
    summary = summary.rename(columns={"index": "prefix", "prefix": "exp_name_prefix"})
    # Arrange columns
    cols = ["exp_name_prefix", "n_seeds", "mean", "median",
            "min", "max", "IQR", "std", "sem", "ci_lower", "ci_upper", "CV"]
    return summary[cols]

def main():
    p = argparse.ArgumentParser(
        description="Summarize final episodic returns per experiment prefix"
    )
    p.add_argument("-i", "--input", required=True, help="Path to input CSV")
    p.add_argument("-o", "--output", required=True, help="Path to output summary CSV")
    p.add_argument("--endpoint", choices=["last","tail_mean","ema","best","last_steps"],
                   default="tail_mean",
                   help="Per-seed endpoint definition (default: tail_mean).")
    p.add_argument("--tail-k", type=int, default=10,
                   help="K for tail_mean (default: 10).")
    p.add_argument("--ema-span", type=int, default=10,
                   help="EMA span for ema endpoint (default: 10).")
    p.add_argument("--last-steps", type=int, default=1_000_000,
                   help="Window in env steps for last_steps endpoint (default: 1e6).")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    ep = per_seed_endpoints(df, args.endpoint, args.tail_k, args.ema_span, args.last_steps)
    summary = compute_summary_from_endpoints(ep)
    summary.to_csv(args.output, index=False)
    print(f"Wrote summary to {args.output} using endpoint={args.endpoint}")

if __name__ == "__main__":
    main()
