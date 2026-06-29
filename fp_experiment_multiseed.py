"""
fp_experiment_multiseed.py
--------------------------
Run Table III (OOD/Fragility test) over 5 independent seeds
instead of only 1 seed as in the original version.

Khắc phục: Bảng III dùng 1 seed — mâu thuẫn với chuẩn 5-seed đã tuyên bố.

Output:
  - In bảng kết quả Mean ± SD ra console
  - Lưu results/table3_multiseed.csv
  - Vẽ results/fig_ood_fragility.png (biểu đồ so sánh OOD)

Usage:
  python fp_experiment_multiseed.py
"""

import os
import sys
try:
    import numpy._core.numeric as _numeric
except ImportError:
    import numpy.core.numeric as _numeric
    sys.modules['numpy._core.numeric'] = _numeric

# Patch BitGenerators for NumPy 1.x vs 2.x compatibility
try:
    import numpy.random._pickle as p
    import numpy.random._pcg64 as _pcg64
    import numpy.random._mt19937 as _mt19937
    import numpy.random._philox as _philox
    import numpy.random._sfc64 as _sfc64
    p.BitGenerators[_pcg64.PCG64] = _pcg64.PCG64
    p.BitGenerators[_pcg64.PCG64DXSM] = _pcg64.PCG64DXSM
    p.BitGenerators[_mt19937.MT19937] = _mt19937.MT19937
    p.BitGenerators[_philox.Philox] = _philox.Philox
    p.BitGenerators[_sfc64.SFC64] = _sfc64.SFC64
except Exception:
    pass
import io
import csv
import json
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Fix Windows cp1252 encoding
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.makedirs("results", exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("[ERROR] sb3_contrib not installed. Run: pip install sb3-contrib")
    sys.exit(1)

import config
import wrapper
from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

# ── Config ────────────────────────────────────────────────────────────────────
FP_RATES   = [0.05, 0.15, 0.30, 0.45]
SEEDS      = [0, 1, 2, 3, 4]
N_EPISODES = 50   # episodes per seed per FP rate (= 250 total)

# ── Episode runner ────────────────────────────────────────────────────────────
def run_episodes(model, agent_type, env, n_episodes, seed):
    """Run n_episodes evaluation episodes; return list of metric dicts."""
    results = []
    np.random.seed(seed)
    random.seed(seed)
    obs, _ = env.reset(seed=seed)

    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()

        lstm_state = None
        ep_start   = np.ones((1,), dtype=bool)

        if agent_type == "playbook":
            model.reset()

        total_reward   = 0.0
        total_wasted   = 0
        total_invest   = 0
        ttr            = config.MAX_STEPS

        for step in range(1, config.MAX_STEPS + 1):
            if agent_type == "rppo":
                action, lstm_state = model.predict(
                    obs[np.newaxis, :],
                    state=lstm_state,
                    episode_start=ep_start,
                    deterministic=True,
                )
                act = action[0]
            else:  # playbook
                action, _ = model.predict(
                    obs[np.newaxis, :],
                    episode_start=ep_start,
                    deterministic=True,
                )
                act = action

            ep_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(act)

            total_reward += reward
            total_wasted += info["wasted"]
            total_invest += info["investigated"]

            if info["true_clean"] == len(config.HOSTS) and ttr == config.MAX_STEPS:
                ttr = step

            if terminated or truncated:
                break

        results.append({
            "reward":       total_reward,
            "ttr":          ttr,
            "wasted":       total_wasted,
            "investigations": total_invest,
        })

    return results

# ── Main experiment ───────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 80)
    print("  Table III - OOD Fragility Test (5 Seeds x 50 Episodes = 250 per condition)")
    print("=" * 80)

    # Collect: table_data[fp_rate][agent] = {reward:[], ttr:[], wasted:[], invest:[]}
    table_data = {}

    for fp in FP_RATES:
        table_data[fp] = {"rppo": {"reward":[],"ttr":[],"wasted":[],"invest":[]},
                          "playbook": {"reward":[],"ttr":[],"wasted":[],"invest":[]}}

        # Override FP rate globally
        wrapper.FALSE_POSITIVE_RATE = fp
        config.FALSE_POSITIVE_RATE  = fp

        print(f"\n== FP Rate = {fp*100:.0f}% ==============================")

        for seed in SEEDS:
            print(f"   Seed {seed}: ", end="", flush=True)

            # Load RPPO model for this seed
            rppo_path = f"results/models/rppo_seed{seed}_ablationnone.zip"
            if not os.path.exists(rppo_path):
                # Fallback
                rppo_path = "rppo_irs_final.zip"
                if not os.path.exists(rppo_path):
                    print(f"[SKIP — model not found: {rppo_path}]")
                    continue
            
            env_rppo = SchoolIRSEnv()
            env_pb   = SchoolIRSEnv()
            custom_objects = {
                "action_space": env_rppo.action_space,
                "observation_space": env_rppo.observation_space
            }
            rppo_model = RecurrentPPO.load(rppo_path, custom_objects=custom_objects)
            pb_model   = StaticPlaybookAgent()
            # Run RPPO
            rppo_res = run_episodes(rppo_model, "rppo", env_rppo, N_EPISODES, seed)
            for r in rppo_res:
                table_data[fp]["rppo"]["reward"].append(r["reward"])
                table_data[fp]["rppo"]["ttr"].append(r["ttr"])
                table_data[fp]["rppo"]["wasted"].append(r["wasted"])
                table_data[fp]["rppo"]["invest"].append(r["investigations"])

            # Run Playbook
            pb_res = run_episodes(pb_model, "playbook", env_pb, N_EPISODES, seed)
            for r in pb_res:
                table_data[fp]["playbook"]["reward"].append(r["reward"])
                table_data[fp]["playbook"]["ttr"].append(r["ttr"])
                table_data[fp]["playbook"]["wasted"].append(r["wasted"])
                table_data[fp]["playbook"]["invest"].append(r["investigations"])

            print(f"RPPO_mean={np.mean([r['reward'] for r in rppo_res]):.1f}  "
                  f"PB_mean={np.mean([r['reward'] for r in pb_res]):.1f}")

    # ── Print Table III ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  TABLE III (Updated -- 5 Seeds, 250 Episodes per Condition)")
    print("=" * 100)
    header = f"{'FP Rate':^8} | {'Agent':^14} | {'Reward (Mean+-SD)':^20} | {'TTR (Mean+-SD)':^18} | {'Wasted (Mean+-SD)':^18} | {'Investigate (Mean+-SD)':^22}"
    print(header)
    print("-" * 100)

    csv_rows = []
    for fp in FP_RATES:
        for agent_key, agent_label in [("rppo","RecurrentPPO"), ("playbook","Static Playbook")]:
            d = table_data[fp][agent_key]
            if len(d["reward"]) == 0:
                continue
            r_m, r_s  = np.mean(d["reward"]),  np.std(d["reward"])
            t_m, t_s  = np.mean(d["ttr"]),     np.std(d["ttr"])
            w_m, w_s  = np.mean(d["wasted"]),  np.std(d["wasted"])
            i_m, i_s  = np.mean(d["invest"]),  np.std(d["invest"])

            row_str = (
                f"{fp*100:6.0f}%   | {agent_label:^14} | "
                f"{r_m:8.2f}+-{r_s:7.2f}  | "
                f"{t_m:6.2f}+-{t_s:5.2f}  | "
                f"{w_m:6.2f}+-{w_s:5.2f}  | "
                f"{i_m:8.2f}+-{i_s:7.2f}"
            )
            print(row_str)
            csv_rows.append({
                "fp_rate": fp, "agent": agent_label,
                "reward_mean": round(r_m,2), "reward_sd": round(r_s,2),
                "ttr_mean": round(t_m,2), "ttr_sd": round(t_s,2),
                "wasted_mean": round(w_m,2), "wasted_sd": round(w_s,2),
                "investigate_mean": round(i_m,2), "investigate_sd": round(i_s,2),
                "n_episodes": len(d["reward"]),
            })
        print("-" * 100)

    print("\nFootnote: Results aggregated over 5 independent random seeds (50 episodes/seed).")
    print("Total episodes per condition: 250. See Section V.A for seed configuration.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = "results/table3_multiseed.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[OK] CSV saved: {csv_path}")

    # ── Plot Figure ───────────────────────────────────────────────────────────
    if csv_rows:
        plot_ood_figure(table_data)

    # Save raw data as JSON for reproducibility
    json_path = "results/table3_raw.json"
    serializable = {}
    for fp, agents in table_data.items():
        serializable[str(fp)] = {}
        for agent, metrics in agents.items():
            serializable[str(fp)][agent] = {k: [round(x,4) for x in v] for k,v in metrics.items()}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OK] Raw data saved: {json_path}")


def plot_ood_figure(table_data):
    """Vẽ biểu đồ OOD fragility comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#F8F9FA")
    for ax in axes:
        ax.set_facecolor("#F8F9FA")

    fp_vals  = [fp * 100 for fp in FP_RATES]
    RPPO_COL = "#4C72B0"
    PB_COL   = "#DD8452"

    for ax, metric, ylabel, title in [
        (axes[0], "reward",  "Mean Cumulative Reward (±SD)",   "Reward vs. FP Rate"),
        (axes[1], "wasted",  "Mean Wasted Restores (±SD)",     "Wasted Restores vs. FP Rate"),
    ]:
        rppo_means = [np.mean(table_data[fp]["rppo"][metric])    for fp in FP_RATES]
        rppo_stds  = [np.std(table_data[fp]["rppo"][metric])     for fp in FP_RATES]
        pb_means   = [np.mean(table_data[fp]["playbook"][metric]) for fp in FP_RATES]
        pb_stds    = [np.std(table_data[fp]["playbook"][metric])  for fp in FP_RATES]

        ax.errorbar(fp_vals, rppo_means, yerr=rppo_stds, marker="o", color=RPPO_COL,
                    linewidth=2.0, capsize=5, label="RecurrentPPO (LSTM)", zorder=3)
        ax.errorbar(fp_vals, pb_means,   yerr=pb_stds,   marker="s", color=PB_COL,
                    linewidth=2.0, capsize=5, label="Static Playbook",     zorder=3)

        ax.axvline(15, color="#888888", linestyle="--", linewidth=1.0,
                   alpha=0.7, label="Training FP rate (15%)")
        ax.set_xlabel("SIEM False Positive Rate (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(fp_vals)
        ax.set_xticklabels([f"{v:.0f}%" for v in fp_vals])
        ax.legend(frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Fig. X: OOD Robustness — Performance Under Varying SIEM False Positive Rates\n"
        "(5 Seeds × 50 Episodes = 250 Episodes per Condition)",
        fontsize=12, fontweight="bold", y=1.02
    )

    fig.text(
        0.5, -0.03,
        "Note: Error bars show ±1 SD across 250 evaluation episodes (5 seeds × 50 episodes/seed).\n"
        "Vertical dashed line marks training FP rate (15%). Values beyond this line represent OOD conditions.",
        ha="center", fontsize=8.5, color="#555555", style="italic"
    )

    plt.tight_layout()
    out_path = "results/fig_ood_fragility.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"[OK] OOD figure saved: {out_path}")


if __name__ == "__main__":
    main()
