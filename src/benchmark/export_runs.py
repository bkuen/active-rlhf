import pandas as pd
import wandb

api = wandb.Api()
# Project is specified by <entity/project-name>
runs = api.runs("bkuen-ludwig-maximilianuniversity-of-munich/final-benchmark-pusher")

# Accumulate all rows in a list of dicts
rows = []
# exp_names = ["pref_ppo_random"]
exp_names = ["pref_ppo_random_v1", "pref_ppo_variquery_v1", "pref_ppo_duo_v1"]

for run in runs:
    # Check if the run name contains any of the specified experiment names
    if not any(exp_name in run.name for exp_name in exp_names):
        continue

    history = run.scan_history()
    print(f"Processing run: {run.name} with seed {run.config.get('seed')}")
    for entry in history:
        if "charts/episodic_return" in entry and "global_step" in entry:
            if entry["charts/episodic_return"] is not None and entry["global_step"] is not None:
                rows.append({
                    "exp_name": run.name,
                    "seed": run.config.get("seed"),
                    "charts/episodic_return": entry["charts/episodic_return"],
                    "global_step": entry["global_step"]
                })

print("Saving data to CSV...")

# Create a DataFrame from the list
df = pd.DataFrame(rows, columns=["exp_name", "seed", "charts/episodic_return", "global_step"])

# Save to CSV
df.to_csv("output/runs_pusher.csv", index=False)
