"""
wilcoxon_pairwise.py

Run pair-wise Wilcoxon signed-rank tests on three learning-curve
columns (episodic return) that share the same `global_step` index.
This version handles a “long” CSV with columns:
  exp_name, seed, charts/episodic_return, global_step

Adjust CSV_PATH and COLUMN_MAP to your own file & experiment names.
"""

import pandas as pd
from scipy.stats import wilcoxon

# ------------------------------------------------------------------
# 1) User inputs
# ------------------------------------------------------------------
CSV_PATH = "output/exported_runs2.csv"   # path to your CSV
# map human-readable label → exact exp_name value in the CSV
COLUMN_MAP = {
    "Hybrid PPO (H)":     "Walker2d-v4__pref_ppo_hybrid_v2__4__1750514347",
    "VariQuery PPO (V)":  "Walker2d-v4__pref_ppo_variquery_v3__1__1750507449",
    "Random PPO (R)":     "Walker2d-v4__pref_ppo_random__4__1750424766",
}
GLOBAL_STEP_COL = "global_step"          # x-axis column

# ------------------------------------------------------------------
# 2) Load & prepare data
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# keep only the three experiments we're interested in
df = df[df["exp_name"].isin(COLUMN_MAP.values())]

# compute mean episodic return over seeds, at each global_step
mean_df = (
    df
    .groupby(["exp_name", GLOBAL_STEP_COL])["charts/episodic_return"]
    .mean()
    .reset_index()
)

# pivot into wide form: rows = global_step, cols = exp_name values
wide_df = mean_df.pivot(
    index=GLOBAL_STEP_COL,
    columns="exp_name",
    values="charts/episodic_return"
)

# rename the full exp_name columns to our human labels
wide_df = wide_df.rename(columns={v: k for k, v in COLUMN_MAP.items()})

# drop any steps where data is missing for one of the experiments
df_sub = wide_df.dropna().reset_index()

# ------------------------------------------------------------------
# 3) Pair-wise Wilcoxon signed-rank tests
# ------------------------------------------------------------------
pairs = [
    ("Hybrid PPO (H)", "VariQuery PPO (V)"),
    ("Hybrid PPO (H)", "Random PPO (R)"),
    ("VariQuery PPO (V)", "Random PPO (R)"),
]

print(f"Wilcoxon signed-rank tests (paired on `{GLOBAL_STEP_COL}`)\n")
for a, b in pairs:
    stat, p = wilcoxon(
        df_sub[a],
        df_sub[b],
        alternative="two-sided",
        zero_method="wilcox"   # ignores zero differences
    )
    print(f"{a:<22} vs {b:<22}  statistic = {stat:>8.1f},  p-value = {p:.3e}, significant(at alpha=0.05) = {p < 0.05}")

# ------------------------------------------------------------------
# 4) (Optionally) save results to disk
# ------------------------------------------------------------------
# import json, pathlib
# out = {
#     f"{a} vs {b}": {"statistic": float(stat), "p": float(p)}
#     for (a,b), (stat,p) in zip(pairs, [
#         wilcoxon(df_sub[a], df_sub[b], alternative="two-sided", zero_method="wilcox")
#         for a, b in pairs
#     ])
# }
# pathlib.Path("wilcoxon_results.json").write_text(json.dumps(out, indent=2))