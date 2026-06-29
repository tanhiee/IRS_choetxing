"""
visualize.py
------------
Generates 4 publication-ready charts from training and evaluation data.

Charts produced (saved to results/):
  1. training_curve.png      – Episode reward over timesteps (training progress)
  2. ttr_comparison.png      – Box plot: Time-to-Recovery (PPO vs Static Playbook)
  3. damage_impact.png       – Bar chart: avg max compromised hosts per agent
  4. attack_success_rate.png – % of episodes where ≥3 hosts compromised at end

Usage:
  python visualize.py
  (Run AFTER train.py and evaluate.py have completed.)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import RESULTS_DIR, TRAINING_REWARDS_CSV

matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

PALETTE = {
    "PPO":           "#4C72B0",
    "StaticPlaybook":"#DD8452",
    "accent":        "#55A868",
    "bg":            "#F8F9FA",
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_training_csv() -> pd.DataFrame | None:
    if not os.path.exists(TRAINING_REWARDS_CSV):
        print(f"  [WARN] Training reward log not found: {TRAINING_REWARDS_CSV}")
        print("         Run train.py first.")
        return None
    return pd.read_csv(TRAINING_REWARDS_CSV)


def load_eval_csv() -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Evaluation results not found: {path}")
        print("         Run evaluate.py first.")
        return None
    return pd.read_csv(path)


def savefig(fig, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved → {path}")


# ── Chart 1: Training Curve ───────────────────────────────────────────────────

def plot_training_curve(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Raw episode rewards as faint scatter
    ax.scatter(df["timestep"], df["reward"], alpha=0.15, s=8,
               color=PALETTE["PPO"], label="_nolegend_")

    # Rolling mean (window = 10% of episodes, min 5)
    window = max(5, len(df) // 10)
    rolled = df["reward"].rolling(window, min_periods=1).mean()
    ax.plot(df["timestep"], rolled, color=PALETTE["PPO"],
            linewidth=2.2, label=f"Rolling mean (w={window})")

    ax.axhline(rolled.iloc[-1], color=PALETTE["accent"],
               linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Final mean: {rolled.iloc[-1]:.1f}")

    ax.set_xlabel("Training Timestep")
    ax.set_ylabel("Episode Reward")
    ax.set_title("PPO Agent – Training Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.35)

    savefig(fig, "training_curve.png")


# ── Chart 2: Time-to-Recovery Box Plot ────────────────────────────────────────

def plot_ttr_comparison(df: pd.DataFrame):
    ppo = df[df["agent"] == "PPO"]["time_to_recovery"].values
    bl  = df[df["agent"] == "StaticPlaybook"]["time_to_recovery"].values

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bp = ax.boxplot(
        [ppo, bl],
        tick_labels=["PPO Agent", "Static Playbook"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    colors = [PALETTE["PPO"], PALETTE["StaticPlaybook"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Annotate medians
    for i, data in enumerate([ppo, bl], start=1):
        med = np.median(data)
        ax.text(i, med + 0.5, f"{med:.0f}", ha="center", va="bottom",
                fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("Steps to Full Recovery (lower ✓ better)")
    ax.set_title("Time to Recovery – PPO vs Static Playbook",
                 fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.35)

    savefig(fig, "ttr_comparison.png")


# ── Chart 3: Damage Impact Bar Chart ─────────────────────────────────────────

def plot_damage_impact(df: pd.DataFrame):
    agents = df["agent"].unique()
    means  = [df[df["agent"] == a]["max_compromised"].mean() for a in agents]
    stds   = [df[df["agent"] == a]["max_compromised"].std()  for a in agents]
    colors = [PALETTE.get(a, "#999999") for a in agents]
    labels = ["PPO Agent", "Static Playbook"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.88, width=0.45,
                  error_kw=dict(linewidth=1.5, ecolor="#555555"))

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{m:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Avg Peak Compromised Hosts (lower ✓ better)")
    ax.set_ylim(0, max(means) * 1.35 + 0.5)
    ax.set_title("Damage Impact – Max Simultaneously Compromised Hosts",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.35)

    savefig(fig, "damage_impact.png")


# ── Chart 4: Attack Success Rate ─────────────────────────────────────────────

def plot_attack_success_rate(df: pd.DataFrame, threshold: int = 3):
    """
    ASR = % of episodes where the maximum number of simultaneously compromised
    hosts was >= threshold at any point during the episode.
    (Attacker 'succeeded' in spreading to at least 3 out of 6 hosts.)
    """
    agents = df["agent"].unique()
    asrs   = []
    for a in agents:
        sub  = df[df["agent"] == a]
        asr  = (sub["max_compromised"] >= threshold).mean() * 100
        asrs.append(asr)

    colors = [PALETTE.get(a, "#999999") for a in agents]
    labels = ["PPO Agent", "Static Playbook"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.bar(labels, asrs, color=colors, alpha=0.88, width=0.45)
    for bar, v in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel(f"Attack Success Rate (≥{threshold} hosts compromised) [%]")
    ax.set_title(f"Attack Success Rate – Attacker Reached ≥{threshold}/{len(df['agent'].unique())*0 + 6} Hosts",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axhline(50, color="#888", linewidth=1.0, linestyle="--", alpha=0.5, label="50% reference")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.35)

    savefig(fig, "attack_success_rate.png")


# ── Overview Dashboard (all 4 in one PNG) ────────────────────────────────────

def plot_dashboard(train_df, eval_df):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axs:
        ax.set_facecolor(PALETTE["bg"])

    # ── Panel 1: Training Curve ─────────────────────────────────────
    if train_df is not None:
        window = max(5, len(train_df) // 10)
        rolled = train_df["reward"].rolling(window, min_periods=1).mean()
        axs[0].scatter(train_df["timestep"], train_df["reward"],
                       alpha=0.12, s=5, color=PALETTE["PPO"])
        axs[0].plot(train_df["timestep"], rolled, color=PALETTE["PPO"], linewidth=2)
        axs[0].set_title("Training Curve", fontweight="bold")
        axs[0].set_xlabel("Timestep"); axs[0].set_ylabel("Episode Reward")
        axs[0].grid(axis="y", alpha=0.3)

    # ── Panel 2: TTR Box Plot ───────────────────────────────────────
    if eval_df is not None:
        ppo = eval_df[eval_df["agent"] == "PPO"]["time_to_recovery"].values
        bl  = eval_df[eval_df["agent"] == "StaticPlaybook"]["time_to_recovery"].values
        bp  = axs[1].boxplot([ppo, bl], tick_labels=["PPO", "Static"],
                             patch_artist=True, widths=0.4,
                             medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], [PALETTE["PPO"], PALETTE["StaticPlaybook"]]):
            patch.set_facecolor(color); patch.set_alpha(0.85)
        axs[1].set_title("Time to Recovery (steps)", fontweight="bold")
        axs[1].set_ylabel("Steps"); axs[1].grid(axis="y", alpha=0.3)

    # ── Panel 3: Damage Impact ──────────────────────────────────────
    if eval_df is not None:
        agents = ["PPO", "StaticPlaybook"]
        means  = [eval_df[eval_df["agent"] == a]["max_compromised"].mean() for a in agents]
        cols   = [PALETTE["PPO"], PALETTE["StaticPlaybook"]]
        bars   = axs[2].bar(["PPO", "Static"], means, color=cols, alpha=0.88, width=0.4)
        for bar, m in zip(bars, means):
            axs[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{m:.2f}", ha="center", fontweight="bold")
        axs[2].set_title("Avg. Peak Compromised Hosts", fontweight="bold")
        axs[2].set_ylabel("Hosts"); axs[2].grid(axis="y", alpha=0.3)

    # ── Panel 4: ASR ────────────────────────────────────────────────
    if eval_df is not None:
        threshold = 3
        agents  = ["PPO", "StaticPlaybook"]
        asrs    = [(eval_df[eval_df["agent"] == a]["max_compromised"] >= threshold).mean() * 100
                   for a in agents]
        cols    = [PALETTE["PPO"], PALETTE["StaticPlaybook"]]
        bars    = axs[3].bar(["PPO", "Static"], asrs, color=cols, alpha=0.88, width=0.4)
        for bar, v in zip(bars, asrs):
            axs[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", fontweight="bold")
        axs[3].set_ylim(0, 115)
        axs[3].set_title(f"Attack Success Rate (≥{threshold} hosts)", fontweight="bold")
        axs[3].set_ylabel("ASR (%)"); axs[3].grid(axis="y", alpha=0.3)

    # Shared legend
    handles = [
        mpatches.Patch(color=PALETTE["PPO"],           label="PPO Agent"),
        mpatches.Patch(color=PALETTE["StaticPlaybook"], label="Static Playbook"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("IRS-RL: Automated Incident Response via Reinforcement Learning\n"
                 "Performance Dashboard", fontsize=16, fontweight="bold", y=1.01)

    savefig(fig, "dashboard.png")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  IRS-RL Visualization")
    print("=" * 70)

    train_df = load_training_csv()
    eval_df  = load_eval_csv()

    if train_df is not None:
        print("\n[1/5] Training curve …")
        plot_training_curve(train_df)

    if eval_df is not None:
        print("[2/5] Time-to-Recovery box plot …")
        plot_ttr_comparison(eval_df)
        print("[3/5] Damage impact bar chart …")
        plot_damage_impact(eval_df)
        print("[4/5] Attack success rate chart …")
        plot_attack_success_rate(eval_df)

    if train_df is not None or eval_df is not None:
        print("[5/5] Combined dashboard …")
        plot_dashboard(train_df, eval_df)

    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
