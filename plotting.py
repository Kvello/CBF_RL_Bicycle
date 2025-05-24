#!/usr/bin/env python
# aggregate_wandb.py
import os
import wandb
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil


mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
# (optional) add any LaTeX packages you need, e.g. amsmath
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amsfonts}'

ENTITY   = "markus-kv1-ntnu"
PROJECT  = "hippo-double-integrator-aggregate"
BASELINE_PROJECT = "ppo-penalty-double-integrator-aggregate"
env_name = "double_integrator"
METRICS  = [
    "step_count(average)", "reward", "neg_cost",
    "eval step_count(average)", "eval neg_cost(average)",
    "eval reward(average)"
]
metric_to_latex = {
    "step_count(average)": r"$\mathbb{E}[T]$",
    "reward": r"$\mathbb{E}[r]$",
    "neg_cost": r"$-\mathbb{E}[c]$",
    "eval step_count(average)": r"$\mathbb{E}\bigl[T^\text{eval}\bigr]$",
    "eval neg_cost(average)": r"$-\mathbb{E}\bigl[c^\text{eval}\bigr]$",
    "eval reward(average)": r"$\mathbb{E}\bigl[r^\text{eval}\bigr]$"
}
MAX_STEPS = 127
# -----------------------------------------

api  = wandb.Api()                        # requires WANDB_API_KEY in env
runs = api.runs(f"{ENTITY}/{PROJECT}",
                filters={"state": "finished"})   # only completed runs
baseline_runs = api.runs(f"{ENTITY}/{BASELINE_PROJECT}",
                        filters={"state": "finished"})  # baseline runs

dfs  = []
for run in runs:
    hist = run.history(keys=METRICS, samples=MAX_STEPS, pandas=True)  # full DF :contentReference[oaicite:0]{index=0}
    hist["seed"] = run.config.get("seed", -1)
    dfs.append(hist)

df_all = pd.concat(dfs, ignore_index=True)
dfs_baseline = []
for run in baseline_runs:
    hist = run.history(keys=METRICS, samples=MAX_STEPS, pandas=True)
    hist["seed"] = run.config.get("seed", -1)
    hist["constraint_penalty"] = run.config["algorithm"]["constraint_penalty"]
    dfs_baseline.append(hist)
df_baseline = pd.concat(dfs_baseline, ignore_index=True)
penalty_values = df_baseline["constraint_penalty"].unique()

num_plots = len(METRICS)
num_cols = 3
num_rows = ceil(num_plots / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6), sharex=True)
axes = axes.flatten()
legends = ["HiPPO"] + [r"$\nu = $" + str(penalty) for penalty in penalty_values]
for ax, metric in zip(axes, METRICS):
    # baseline “HiPPO” curve
    sns.lineplot(
        data=df_all, x="_step", y=metric,
        estimator="mean", errorbar=("pi",95), linewidth=2,
        label=r"\text{HiPPO}",  # now also LaTeX
        ax=ax,
    )

    # each penalty curve
    for penalty in penalty_values:
        sns.lineplot(
            data= df_baseline[df_baseline.constraint_penalty == penalty],
            x     = "_step",
            y     = metric,
            estimator="mean",
            errorbar =("pi",95),
            linewidth  = 2,
            label = rf"$\nu = {penalty}$",  # raw‐string latex
            ax    = ax,
        )

    ax.grid(True)
    ax.set_title(rf"${metric_to_latex.get(metric, metric)}$")
    ax.set_xlabel("Optimization Steps")
    ax.legend()

for ax in axes[num_plots:]:
    ax.remove()
plt.tight_layout()
plt.savefig(f"plots/{env_name}_aggregate_metrics.pdf")
if env_name == "double_integrator":
    bellman_metrics = [
        "bellman_violation_mean",
        "bellman_violation_max",
        "bellman_violation_std",
    ]
    bellman_metrics_to_latex = {
        "bellman_violation_mean": r"$\mathbb{E}[\text{Bellman Violation}]$",
        "bellman_violation_max": r"$\max(\text{Bellman Violation})$",
        "bellman_violation_std": r"$\sigma(\text{Bellman Violation})$"
    }
    dfs = []
    for run in runs:
        hist = run.history(keys=bellman_metrics, samples=MAX_STEPS, pandas=True)
        hist["seed"] = run.config.get("seed", -1)
        dfs.append(hist)
    df_all = pd.concat(dfs, ignore_index=True)
    fig, axes = plt.subplots(1, len(bellman_metrics), figsize=(12, 4), sharex=True)
    for ax, metric in zip(axes, bellman_metrics):
        sns.lineplot(
            data=df_all, x="_step", y=metric,
            estimator="mean", errorbar=("pi",95), linewidth=2,
            ax=ax,
        )
        ax.set_title(rf"${bellman_metrics_to_latex.get(metric, metric)}$")
        ax.set_xlabel("Optimization Steps")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{env_name}_bellman_metrics.pdf")
plt.close('all')  # Close all figures to free memory
        
    
