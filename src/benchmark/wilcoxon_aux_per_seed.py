#!/usr/bin/env python3
"""
wilcoxon_auc_by_prefix.py

Parse exp_name strings of the form:
    <experiment-prefix>__<seed>__<timestamp>
into `prefix` and `seed`, compute AUC per run, then run
pairwise Wilcoxon tests on the per-seed AUCs.
"""

import re
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# 1) User inputs — adjust as needed
CSV_PATH      = "output/exported_runs2.csv"   # your CSV file
EXP_NAME_COL  = "exp_name"
GLOBAL_STEP   = "global_step"
VALUE_COL     = "charts/episodic_return"

# 2) Load data
df = pd.read_csv(CSV_PATH)

# 3) Parse out `prefix` and `seed` from each exp_name
#    expecting strings like "Walker2d-v4__pref_ppo_hybrid_v2__4__1750514347"
pat = re.compile(r"(?P<prefix>.+)__(?P<seed>\d+)__(?P<ts>\d+)$")
parsed = df[EXP_NAME_COL].str.extract(pat)
if parsed.isnull().any().any():
    raise ValueError("Some exp_name rows did not match the expected pattern")
df["prefix"] = parsed["prefix"]
df["seed"]   = parsed["seed"].astype(int)

# 4) Compute AUC per (prefix, seed) via trapezoidal rule
df_sorted = df.sort_values(GLOBAL_STEP)
auc_df = (
    df_sorted
      .groupby(["prefix", "seed"])[[GLOBAL_STEP, VALUE_COL]]
      .apply(lambda g: np.trapz(g[VALUE_COL], x=g[GLOBAL_STEP]))
      .reset_index(name="AUC")
)

# 5) Pivot so each row is a seed, each column an experiment prefix
wide_auc = auc_df.pivot(index="seed", columns="prefix", values="AUC")

print("Per-seed AUC table (NaN = missing run):\n")
print(wide_auc, "\n")

# 6) Prepare pairwise comparisons
prefixes = wide_auc.columns.tolist()
pairs = [
    (prefixes[i], prefixes[j])
    for i in range(len(prefixes)) for j in range(i+1, len(prefixes))
]

# 7) Run Wilcoxon signed-rank tests per pair
print("Pairwise Wilcoxon signed-rank tests on per-seed AUCs:\n")
for a, b in pairs:
    pair = wide_auc[[a, b]].dropna()
    n = len(pair)
    print(f"→ {a} vs {b}: {n} seeds in common", end="")
    if n < 1:
        print("  → skipping (no overlapping seeds)\n")
        continue

    diffs = pair[a] - pair[b]
    if (diffs == 0).all():
        print("  → all differences zero (p ≈ 1.0)\n")
        continue

    stat, p = wilcoxon(pair[a], pair[b], alternative="two-sided", zero_method="wilcox")
    sig = "yes" if p < 0.05 else "no"
    print(f"  → stat = {stat:.1f}, p = {p:.3e}, significant? {sig}\n")