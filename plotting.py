#!/usr/bin/env python
# aggregate_wandb.py
import os
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------  user settings  -------------
ENTITY   = "markus-kv1-ntnu"
PROJECT  = "hippo-double-integrator-aggregate"
METRICS  = [
    "step_count(average)", "reward", "neg_cost",
    "eval step_count(average)", "eval neg_cost(average)",
    "eval reward(average)",
    "bellman_violation_std", "bellman_violation_mean",
    "bellman_violation_max",
]
MAX_STEPS = 127
# -----------------------------------------

api  = wandb.Api()                        # requires WANDB_API_KEY in env
runs = api.runs(f"{ENTITY}/{PROJECT}",
                filters={"state": "finished"})   # only completed runs

dfs  = []
for run in runs:
    hist = run.history(keys=METRICS, samples=MAX_STEPS, pandas=True)  # full DF :contentReference[oaicite:0]{index=0}
    hist["seed"] = run.config.get("seed", -1)
    dfs.append(hist)

df_all = pd.concat(dfs, ignore_index=True)

def plot_metric(df, metric, x_axis="_step"):
    fig = plt.figure(figsize=(8, 4))
    fig.patch.set_facecolor('white')
    ax = sns.lineplot(
        data=df, x=x_axis, y=metric,
        estimator="mean", errorbar="ci", linewidth=2)
    ax.set_title(f"{metric}  (mean ±95 % CI over seeds)")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(metric)
    ax.grid(True)
    plt.savefig(f"plots/{metric}.png", dpi=300)

# plot_metric(df_all, "reward")                 # training reward
# plot_metric(df_all, "eval reward(average)")   # eval reward