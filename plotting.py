#!/usr/bin/env python
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
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
env_name = "double-integrator"
PROJECT  = "hippo-"+env_name+"-aggregate"
BASELINE_PROJECT = "ppo-penalty-"+env_name+"-aggregate"
METRICS  = [
    "step_count(average)",
    "eval step_count(average)",
    "neg_cost",
    "eval neg_cost(average)",
    "reward",
    "eval reward(average)"
]
metric_to_latex = {
    "step_count(average)": r"$\hat{\mathbb{E}}[T]$",
    "reward": r"$\hat{\mathbb{E}}[r]$",
    "neg_cost": r"$-\hat{\mathbb{E}}[c]$",
    "eval step_count(average)": r"$\hat{\mathbb{E}}\bigl[T_\text{eval}\bigr]$",
    "eval neg_cost(average)": r"$-\hat{\mathbb{E}}\bigl[c_\text{eval}\bigr]$",
    "eval reward(average)": r"$\hat{\mathbb{E}}\bigl[r_\text{eval}\bigr]$"
}
MAX_STEPS = None
zoomed_metrics = []
if env_name ==  "double-integrator":
    MAX_STEPS = 127
    zoomed_metrics = [
        "eval neg_cost(average)",
        "neg_cost",
    ]
elif env_name == "cartpole":
    MAX_STEPS = 255
elif env_name == "quadcopter":
    MAX_STEPS = 2048
assert MAX_STEPS is not None, "MAX_STEPS should not be none"
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
num_cols = 2
num_rows = ceil(num_plots / num_cols)

styles = [('-', 'o'), ('--', 's'), ('-.', '^'), (':', 'd')]
penalties = sorted(penalty_values)
colors = sns.color_palette(n_colors=len(penalties) + 1)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9), sharex=True)
axes = axes.flatten()
# zoomed_metrics = []
errorbar = ("pi",50)
for ax, metric in zip(axes, METRICS):
    # baseline “HiPPO” curve
    sns.lineplot(
        data=df_all, x="_step", y=metric,
        estimator="mean", errorbar=errorbar, linewidth=2,
        label=r"\text{HiPPO}",  # now also LaTeX
        ax=ax,
        alpha=0.9,
        linestyle="-", markers="o",
        legend=False,
        color=colors[0]
    )
    if metric in zoomed_metrics:
        axins = zoomed_inset_axes(ax,
            zoom=5.0,
            loc='center right')
        sns.lineplot(
            data=df_all, x="_step", y=metric,
            estimator="mean", errorbar=errorbar, linewidth=2,
            legend=False,
            ax=axins,
            alpha=0.9,
            linestyle="-", markers="o",
            color=colors[0]
        )

    # each penalty curve
    for i, penalty in enumerate(penalties):
        ls, mk = styles[i%len(styles)]
        sns.lineplot(
            data= df_baseline[df_baseline.constraint_penalty == penalty],
            x     = "_step",
            y     = metric,
            estimator="mean",
            errorbar =errorbar,
            linewidth  = 2,
            label = rf"$\nu = {penalty}$",  # raw‐string latex
            legend = False,
            ax    = ax,
            alpha=0.7,
            color=colors[i+1],marker=mk,linestyle=ls,
            markevery=20
        )
        if metric in zoomed_metrics:
            sns.lineplot(
                    data= df_baseline[df_baseline.constraint_penalty == penalty],
                    x     = "_step",
                    y     = metric,
                    estimator="mean",
                    errorbar =errorbar,
                    linewidth  = 2,
                    ax    = axins,
                    legend = False,
                    alpha=0.7,
                    color=colors[i+1],marker=mk,linestyle=ls,
                    markevery=20
            )
     
    if metric in zoomed_metrics:
        axins.set_xlim(int(MAX_STEPS*0.9), MAX_STEPS)
        if metric in ["neg_cost","eval neg_cost(average)"]:
            axins.set_ylim(-5e-3, 0.0)
        else:
            max_T = df_all[metric].max()
            axins.set_ylim(int(max_T*0.85),max_T)
        axins.set_ylabel("")
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
    ax.set_ylabel("")
    ax.yaxis.labelpad = 0
    ax.grid(True)
    ax.set_title(rf"${metric_to_latex.get(metric, metric)}$")
    ax.set_xlabel("Optimization Steps")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,loc="lower right", bbox_to_anchor=(0.97, 0.1))

for ax in axes[num_plots:]:
    ax.remove()
plt.tight_layout()
plt.savefig(f"plots/{env_name}_aggregate_metrics.pdf")
if env_name == "double-integrator":
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
            estimator="mean", errorbar=errorbar, linewidth=2,
            ax=ax,
        )
        ax.set_title(rf"${bellman_metrics_to_latex.get(metric, metric)}$")
        ax.set_xlabel("Optimization Steps")
        ax.set_ylabel("")
        ax.yaxis.labelpad = 0
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{env_name}_bellman_metrics.pdf")
plt.close('all')  # Close all figures to free memory
        
    
