#!/usr/bin/env python3
"""
pairwise_wilcoxon_both.py

Compute pairwise Wilcoxon signed-rank tests for RL experiment metrics and also report
paired Δ (mean/median), 95% CIs (t and bootstrap), and paired effect sizes (dz, Hedges' gz),
for BOTH endpoint and AUC in one run, grouped by environment.

Input CSV columns:
  exp_name, seed, <metric_column>, global_step

exp_name format (example):
  <env>__<method>__...   (we only need that env is the first token before '__')

Outputs a single long-form CSV with columns:
  env, metric_type (endpoint|auc), metric, prefix_a, prefix_b, n,
  W_statistic, p_value, p_value_holm, reject_null,
  delta_mean, delta_median, sd_delta,
  ci_t_low, ci_t_high, ci_boot_low, ci_boot_high, dz, gz

Usage:
  python pairwise_wilcoxon_both.py \
    --input experiments.csv \
    --metric charts/episodic_return \
    [--tail-k 10] [--endpoint-mode tail_mean|ema|last] \
    [--ci both|t|bootstrap] [--n-boot 10000] \
    [--alpha 0.05] \
    [--split-by-env] \
    --output results_all.csv
"""
import argparse
import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import wilcoxon
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None


# ------------------------ helpers ------------------------

def extract_prefix_and_env(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['prefix'] = df['exp_name'].str.rsplit('__', n=2).str[0]
    df['env'] = df['prefix'].str.split('__').str[0]
    return df


def compute_endpoint(df: pd.DataFrame, metric: str, tail_k=10, mode="tail_mean") -> pd.DataFrame:
    """Endpoint summary per (prefix, seed). Returns columns: env, prefix, seed, value, metric_type, metric"""
    df = df.sort_values('global_step')

    def endpoint(g):
        if mode == "tail_mean":
            return g[metric].tail(tail_k).mean()
        elif mode == "ema":
            return g[metric].ewm(span=10, adjust=False).mean().iloc[-1]
        elif mode == "last":
            return g[metric].iloc[-1]
        raise ValueError(f"Unknown endpoint mode: {mode}")

    # Use a different approach to avoid the FutureWarning
    results = []
    for (prefix, seed), group in df.groupby(['prefix', 'seed']):
        value = endpoint(group)
        results.append({'prefix': prefix, 'seed': seed, 'value': value})
    
    vals = pd.DataFrame(results)
    vals = vals.merge(df[['prefix','env']].drop_duplicates(), on='prefix', how='left')
    vals['metric_type'] = 'endpoint'
    vals['metric'] = metric
    return vals[['env','prefix','seed','value','metric_type','metric']]


def compute_auc(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """AUC per (prefix, seed). Returns columns: env, prefix, seed, value, metric_type, metric"""
    recs = []
    for (prefix, seed), grp in df.groupby(['prefix','seed']):
        g = grp.sort_values('global_step')
        x = g['global_step'].to_numpy()
        y = g[metric].to_numpy()
        if len(x) < 2:
            auc_val = np.nan
        else:
            # Normalize x to [0, 1] range to get reasonable AUC values
            x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
            auc_val = np.trapezoid(y, x_norm)
        recs.append({'prefix': prefix, 'seed': seed, 'value': float(auc_val)})
    vals = pd.DataFrame(recs)
    vals = vals.merge(df[['prefix','env']].drop_duplicates(), on='prefix', how='left')
    vals['metric_type'] = 'auc'
    vals['metric'] = metric
    return vals[['env','prefix','seed','value','metric_type','metric']]


def paired_stats(x: np.ndarray, y: np.ndarray, alpha=0.05, n_boot=10000, ci="both", random_state=0):
    d = x - y
    n = d.size
    mean_d = float(np.mean(d))
    median_d = float(np.median(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0

    # t-based (z≈1.96 for 95%)
    if n > 1 and sd_d > 0:
        z = 1.96
        half = z * sd_d / np.sqrt(n)
        ci_t = (mean_d - half, mean_d + half)
    else:
        ci_t = (mean_d, mean_d)

    # bootstrap percentile CI
    if ci in ("both","bootstrap") and n > 0:
        rng = np.random.default_rng(random_state)
        idx = np.arange(n)
        boots = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            samp = rng.choice(idx, size=n, replace=True)
            boots[b] = np.mean(d[samp])
        ci_boot = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    else:
        ci_boot = (np.nan, np.nan)

    # paired effect sizes
    if sd_d > 0:
        dz = mean_d / sd_d
        J = 1.0 - 3.0/(4*n - 9) if n > 2 else 1.0
        gz = dz * J
    else:
        dz = 0.0 if mean_d == 0 else float('inf')
        gz = dz

    return dict(n=int(n),
                delta_mean=mean_d, delta_median=median_d, sd_delta=sd_d,
                ci_t_low=float(ci_t[0]), ci_t_high=float(ci_t[1]),
                ci_boot_low=float(ci_boot[0]) if not np.isnan(ci_boot[0]) else np.nan,
                ci_boot_high=float(ci_boot[1]) if not np.isnan(ci_boot[1]) else np.nan,
                dz=dz, gz=gz)


def pairwise_block(values: pd.DataFrame, alpha=0.05, n_boot=10000, ci="both"):
    """Run pairwise Wilcoxon and paired Δ within a (env, metric_type, metric) block."""
    prefixes = sorted(values['prefix'].unique())
    results = []

    for i, a in enumerate(prefixes):
        for b in prefixes[i+1:]:
            pivot = values[values['prefix'].isin([a, b])]
            pivot = pivot.pivot(index='seed', columns='prefix', values='value').dropna()
            if pivot.shape[0] < 1:
                continue
            x = pivot[a].to_numpy()
            y = pivot[b].to_numpy()

            try:
                stat, p = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
            except ValueError:
                stat, p = np.nan, np.nan

            stats = paired_stats(x, y, alpha=alpha, n_boot=n_boot, ci=ci, random_state=0)

            results.append({
                'prefix_a': a,
                'prefix_b': b,
                'W_statistic': stat,
                'p_value': p,
                **stats
            })

    return pd.DataFrame(results)


def holm_within_groups(df: pd.DataFrame, alpha=0.05, group_cols=None):
    """Apply Holm within specified groups (e.g., env, metric_type)."""
    if group_cols is None:
        group_cols = []
    if df.empty or multipletests is None:
        df = df.copy()
        df['p_value_holm'] = np.nan
        df['reject_null'] = False
        return df

    parts = []
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.copy()
        rej, p_corr, _, _ = multipletests(g['p_value'].fillna(1.0), alpha=alpha, method='holm')
        g['p_value_holm'] = p_corr
        g['reject_null'] = rej
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else df.copy()
    return out


# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Pairwise Wilcoxon + Δ & CIs for BOTH endpoint and AUC in one file")
    ap.add_argument('--input', '-i', required=True, help='Path to input CSV')
    ap.add_argument('--metric', '-m', default='charts/episodic_return', help='Metric column to use')
    ap.add_argument('--tail-k', type=int, default=10, help='Tail-K for endpoint tail_mean')
    ap.add_argument('--endpoint-mode', choices=['tail_mean','ema','last'], default='tail_mean', help='Endpoint mode')
    ap.add_argument('--ci', choices=['both','t','bootstrap'], default='both', help='Which CI(s) to compute')
    ap.add_argument('--n-boot', type=int, default=10000, help='Bootstrap resamples')
    ap.add_argument('--alpha', type=float, default=0.05, help='Significance level for Holm')
    ap.add_argument('--split-by-env', action='store_true', help='Also write one CSV per environment')
    ap.add_argument('--output', '-o', required=True, help='Path to output CSV (long-form)')
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    raw = extract_prefix_and_env(raw)

    # Compute BOTH endpoint and AUC blocks
    endpoint_vals = compute_endpoint(raw, args.metric, tail_k=args.tail_k, mode=args.endpoint_mode)
    auc_vals = compute_auc(raw, args.metric)

    all_vals = pd.concat([endpoint_vals, auc_vals], ignore_index=True)

    # Run pairwise inside each (env, metric_type, metric)
    blocks = []
    for (env, metric_type, metric), block in all_vals.groupby(['env','metric_type','metric']):
        block_res = pairwise_block(block, alpha=args.alpha, n_boot=args.n_boot, ci=args.ci)
        block_res.insert(0, 'metric', metric)
        block_res.insert(0, 'metric_type', metric_type)
        block_res.insert(0, 'env', env)
        blocks.append(block_res)

    res = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=[
        'env','metric_type','metric','prefix_a','prefix_b','n','W_statistic','p_value',
        'delta_mean','delta_median','sd_delta','ci_t_low','ci_t_high','ci_boot_low','ci_boot_high','dz','gz'
    ])

    # Holm within env × metric_type
    res = holm_within_groups(res, alpha=args.alpha, group_cols=['env','metric_type'])

    # Order columns
    col_order = [
        'env','metric_type','metric','prefix_a','prefix_b','n',
        'W_statistic','p_value','p_value_holm','reject_null',
        'delta_mean','delta_median','sd_delta',
        'ci_t_low','ci_t_high','ci_boot_low','ci_boot_high','dz','gz'
    ]
    cols = [c for c in col_order if c in res.columns] + [c for c in res.columns if c not in col_order]
    res = res[cols].sort_values(['env','metric_type','p_value'], ignore_index=True)

    # Write long-form CSV
    res.to_csv(args.output, index=False)
    print(f"Wrote combined results to {args.output}")

    # Optional split by env
    if args.split_by_env and not res.empty:
        for env, g in res.groupby('env'):
            out_env = args.output.replace('.csv', f'__{env}.csv')
            g.to_csv(out_env, index=False)
            print(f"Wrote {env} to {out_env}")


if __name__ == '__main__':
    main()
