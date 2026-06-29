"""
plot_training_curve_v2.py
─────────────────────────
Vẽ lại Hình 3 (Training Convergence Curve) từ dữ liệu log thực tế
của 5 seed RecurrentPPO.

Khắc phục mâu thuẫn giữa Hình 3 (~200–220) và Bảng I (3875.42) bằng cách:
  1. Parse log training thực tế từ RPPOseed0-4.txt
  2. Vẽ đường training reward đúng thang đo
  3. Thêm đường ngang evaluation reward (3875.42) để so sánh trực quan
  4. Thêm chú thích giải thích sự khác biệt training vs evaluation reward

Output: results/training_curve_v2.png  (thay Hình 3 cũ trong bài)
"""

import re
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

os.makedirs("results", exist_ok=True)

# ── 1. Parse training log từ RPPOseed0-4.txt ──────────────────────────────────
# Format: Step  XXX,XXX | Ep YYY | Mean reward (last 50): ZZZ.ZZ

SEED_COLORS = {
    0: "#4C72B0",   # blue
    1: "#DD8452",   # orange
    2: "#55A868",   # green
    3: "#C44E52",   # red
    4: "#8172B2",   # purple
}

# Data điểm log từ RPPOseed0-4.txt (parse thủ công từ log)
# Seed 0: chỉ có điểm cuối vì log bị gộp lại 1 dòng
# Seed 3 và 4: có nhiều điểm log mỗi 10k steps
SEED_DATA = {
    # (timestep, mean_reward_last50) pairs
    0: [(500_000, 1968.97)],  # chỉ có điểm cuối từ log
    1: [(500_000, 2831.44)],  # chỉ có điểm cuối
    2: [(500_000, 2790.78)],  # chỉ có điểm cuối
    3: [
        (60_000,  -5775), (70_000,  -1196), (80_000,  -1237), (90_000,  -1888),
        (100_000,  -673), (110_000,  -653), (120_000, -1335), (130_000,   738),
        (140_000,   792), (150_000,  1745), (160_000,  1719), (170_000,  1709),
        (180_000,   845), (190_000,  1273), (200_000,  1713), (210_000,  2320),
        (220_000, -7700), (230_000,  2727), (240_000,  2870), (250_000,  2376),
        (260_000,  2456), (270_000,  2402), (280_000,  1882), (290_000,  2898),
        (300_000,  2302), (310_000,   170), (320_000,  2150), (330_000,  2184),
        (340_000,  2460), (350_000,   835), (360_000,   954), (370_000,  2313),
        (380_000,  3116), (390_000,  3001), (400_000,  2930), (410_000, -1319),
        (420_000,  2114), (430_000,  3077), (440_000,  3405), (450_000,  3222),
        (460_000,  3307), (470_000,  3303), (480_000,  3468), (490_000,  3402),
        (500_000,  3229),
    ],
    4: [
        (10_000,  -9574), (20_000,  -1166), (30_000,  -3278), (40_000,  -1613),
        (50_000,  -1378), (60_000,  -2546), (70_000,  -6592), (80_000,  -2988),
        (90_000,  -3955), (100_000, -1004), (110_000, -1328), (120_000, -3550),
        (130_000,   401), (140_000,  1184), (150_000,  1990), (160_000,  1579),
        (170_000,  2146), (180_000,  1259), (190_000,  2074), (200_000,  2122),
        (210_000,  2114), (220_000, -6890), (230_000,  2295), (240_000,  2469),
        (250_000,  2219), (260_000,  2529), (270_000,  2708), (280_000,  2749),
        (290_000,  2892), (300_000,  3239), (310_000,  2578), (320_000,  2667),
        (330_000,  2505), (340_000,  2643), (350_000,  2866), (360_000,  2776),
        (370_000,  2336), (380_000,  2449), (390_000,  1853), (400_000,  3190),
        (410_000,  3055), (420_000,  2931), (430_000,  2956), (440_000,  3052),
        (450_000,  3072), (460_000,  3622), (470_000,  3256), (480_000,  3140),
        (490_000,  3331), (500_000,  3137),
    ],
}

# Evaluation reward (Bảng I) — post-training deterministic policy, 250 episodes
EVAL_REWARD_MEAN = 3875.42
EVAL_REWARD_STD  =  516.07
PLAYBOOK_REWARD  = 4603.41

# ── 2. Vẽ hình ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

# Vẽ từng seed có dữ liệu đầy đủ (seed 3 & 4)
for seed in [3, 4]:
    pts = SEED_DATA[seed]
    steps = [p[0] for p in pts]
    rewards = [p[1] for p in pts]
    ax.plot(steps, rewards, color=SEED_COLORS[seed], alpha=0.45,
            linewidth=1.0, linestyle="-", zorder=2)

# Tổng hợp envelope từ seed 3 và 4 (có đủ data points)
s3 = {p[0]: p[1] for p in SEED_DATA[3]}
s4 = {p[0]: p[1] for p in SEED_DATA[4]}
common_steps = sorted(set(s3.keys()) & set(s4.keys()))
mean_curve = [(t, np.mean([s3[t], s4[t]])) for t in common_steps]
steps_m = [p[0] for p in mean_curve]
rewards_m = [p[1] for p in mean_curve]

# Rolling mean để làm mượt
window = 7
padded = rewards_m[:window//2] + rewards_m + rewards_m[-(window//2):]
rolled = [np.mean(padded[i:i+window]) for i in range(len(rewards_m))]

ax.plot(steps_m, rolled, color="#2C5F8A", linewidth=2.5, zorder=4,
        label="Rolling mean reward (Seeds 3 & 4, w=7)")

# Điểm cuối của seed 0, 1, 2 (chỉ có giá trị cuối)
final_vals = [SEED_DATA[s][-1][1] for s in [0, 1, 2]]
final_mean = np.mean(final_vals)
for s in [0, 1, 2]:
    ax.scatter(500_000, SEED_DATA[s][-1][1], color=SEED_COLORS[s],
               marker="D", s=80, zorder=5, label=f"Seed {s} final (only logged once)")

# ── Đường evaluation reward ────────────────────────────────────────────────────
ax.axhline(EVAL_REWARD_MEAN, color="#E84B4B", linewidth=2.2,
           linestyle="--", zorder=3,
           label=f"Post-training Evaluation Reward: {EVAL_REWARD_MEAN:.1f} ± {EVAL_REWARD_STD:.1f}")
ax.fill_between([0, 500_000],
                EVAL_REWARD_MEAN - EVAL_REWARD_STD,
                EVAL_REWARD_MEAN + EVAL_REWARD_STD,
                color="#E84B4B", alpha=0.10, zorder=1)

ax.axhline(PLAYBOOK_REWARD, color="#F5A623", linewidth=1.8,
           linestyle=":", zorder=3,
           label=f"Static Playbook Reward: {PLAYBOOK_REWARD:.1f}")

# ── Zero line ─────────────────────────────────────────────────────────────────
ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-", alpha=0.5, zorder=1)

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_xlabel("Training Timestep", fontsize=12)
ax.set_ylabel("Episode Cumulative Reward\n(Mean of last 50 training episodes)", fontsize=11)
ax.set_title("Fig. 3: RecurrentPPO (LSTM) – Training Convergence Curve\n(5 Seeds, 500k Timesteps)",
             fontsize=13, fontweight="bold", pad=12)

ax.set_xlim(0, 510_000)
ax.set_ylim(-12_000, 5_500)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
ax.grid(axis="y", alpha=0.30, linestyle="--")

# ── Annotation chú thích phân biệt training vs evaluation ────────────────────
ax.annotate(
    "Training reward fluctuates due to\nexploration (ent_coef=0.02);\nevaluation uses deterministic policy",
    xy=(500_000, EVAL_REWARD_MEAN),
    xytext=(330_000, 4_800),
    fontsize=8.5,
    color="#C0392B",
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E84B4B", alpha=0.85),
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color="#2C5F8A", linewidth=2.5, label="Training rolling mean (Seeds 3 & 4)"),
    Line2D([0], [0], color=SEED_COLORS[0], linewidth=0, marker="D", markersize=7,
           label=f"Seed 0 final: {SEED_DATA[0][-1][1]:.0f}"),
    Line2D([0], [0], color=SEED_COLORS[1], linewidth=0, marker="D", markersize=7,
           label=f"Seed 1 final: {SEED_DATA[1][-1][1]:.0f}"),
    Line2D([0], [0], color=SEED_COLORS[2], linewidth=0, marker="D", markersize=7,
           label=f"Seed 2 final: {SEED_DATA[2][-1][1]:.0f}"),
    Line2D([0], [0], color="#E84B4B", linewidth=2.2, linestyle="--",
           label=f"Eval reward (post-training): {EVAL_REWARD_MEAN:.1f} ± {EVAL_REWARD_STD:.1f}"),
    mpatches.Patch(facecolor="#E84B4B", alpha=0.15, label=f"Eval ± 1 SD"),
    Line2D([0], [0], color="#F5A623", linewidth=1.8, linestyle=":",
           label=f"Static Playbook: {PLAYBOOK_REWARD:.1f}"),
]
ax.legend(handles=legend_handles, frameon=True, framealpha=0.9,
          fontsize=8.5, loc="lower right", ncol=1)

# ── Caption footnote ─────────────────────────────────────────────────────────
fig.text(
    0.5, -0.05,
    "Note: Training reward (y-axis) = mean of last 50 consecutive training episodes with stochastic exploration policy.\n"
    "Evaluation reward (red dashed line, Table I) = mean over 250 post-training episodes with deterministic (greedy) policy.\n"
    "The gap between training and evaluation values is expected due to policy stochasticity during training.",
    ha="center", va="top", fontsize=8, color="#555555", style="italic",
    wrap=True
)

plt.tight_layout()
out_path = "results/training_curve_v2.png"
fig.savefig(out_path, bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"[OK] Saved: {out_path}")
print(f"\nKey numbers for paper:")
print(f"  Training final mean (Seeds 0-4): {np.mean([SEED_DATA[s][-1][1] for s in range(5)]):.1f}")
print(f"  Evaluation reward (Table I):     {EVAL_REWARD_MEAN:.2f} ± {EVAL_REWARD_STD:.2f}")
print(f"  Ratio (eval/training_mean):      {EVAL_REWARD_MEAN / np.mean([SEED_DATA[s][-1][1] for s in range(5)]):.2f}x")
print(f"\nFigure saved to: {out_path}")
