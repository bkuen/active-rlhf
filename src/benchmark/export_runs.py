import pandas as pd
import wandb

api = wandb.Api()
# Project is specified by <entity/project-name>
runs = api.runs("bkuen-ludwig-maximilianuniversity-of-munich/thesis-benchmark-walker")

# Accumulate all rows in a list of dicts
rows = []
# exp_names = ["pref_ppo_random"]

# HalfCheetah
#exp_names = ["prefppo_random__", "prefppo_variquery__", "prefppo_duo_prio_v2__", "prefppo_hybrid_prio_u_v6__"]
# Hopper
#exp_names = ["prefppo_random__", "prefppo_variquery_v3__", "prefppo_duo_prio__", "prefppo_hybrid_v3__"]
# Ant
#exp_names = ["prefppo_random__", "prefppo_variquery_v10__", "prefppo_duo_prio__", "prefppo_hybrid_prio__"]
# Walker2d
exp_names = ["prefppo_random_v2__", "prefppo_variquery_v3__", "prefppo_duo_prio__", "prefppo_hybrid_prio__"]

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
df.to_csv("output/runs_walker_400.csv", index=False)
