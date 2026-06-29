import csv
import numpy as np
from scipy.stats import mannwhitneyu

SEEDS = [0, 1, 2, 3, 4]
metrics = ["reward", "ttr", "max_compromised", "wasted", "restored", "ended_clean", "investigate_ratio"]

# Helper functions
def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
    return (mean1 - mean2) / pooled_se if pooled_se > 0 else 0.0

def compute_cliffs_delta(group1, group2):
    n1, n2 = len(group1), len(group2)
    greater = 0
    less = 0
    for x in group1:
        for y in group2:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (n1 * n2)

def compute_95_ci(data):
    n = len(data)
    if n < 2:
        return 0.0
    sd = np.std(data, ddof=1)
    margin = 2.776 * (sd / np.sqrt(n))
    return margin

def main():
    # Load raw episode results
    all_episodes = []
    csv_path = "results/evaluation_all_episodes.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_episodes.append({
                "agent": row["agent"],
                "seed": int(row["seed"]),
                "episode": int(row["episode"]),
                "reward": float(row["reward"]),
                "ttr": float(row["ttr"]),
                "max_compromised": float(row["max_compromised"]),
                "ended_clean": float(row["ended_clean"]),
                "wasted": float(row["wasted"]),
                "restored": float(row["restored"]),
                "investigate_ratio": float(row["investigate_ratio"])
            })

    # Compute seed-level stats
    seed_means = {}
    for agent in ["rppo", "mlpppo", "playbook"]:
        seed_means[agent] = {m: [] for m in metrics}
        agent_eps = [e for e in all_episodes if e["agent"] == agent]
        for seed in SEEDS:
            seed_eps = [e for e in agent_eps if e["seed"] == seed]
            for m in metrics:
                mean_val = np.mean([e[m] for e in seed_eps])
                seed_means[agent][m].append(mean_val)

    # Print Table I summary
    print("=" * 100)
    print("  SEED-LEVEL STATISTICS (n=5 seeds)")
    print("=" * 100)
    print(f"{'Metric':<25} | {'RecurrentPPO (LSTM)':^22} | {'MLP-PPO (Memoryless)':^22} | {'Static Playbook (Rule)':^22}")
    print("-" * 100)
    
    for m in metrics:
        def format_metric(agent_key):
            vals = seed_means[agent_key][m]
            mean = np.mean(vals)
            sd = np.std(vals, ddof=1)
            ci = compute_95_ci(vals)
            if m == "ended_clean":
                return f"{mean*100:.2f}% +- {sd*100:.2f}% [CI: {ci*100:.2f}%]"
            return f"{mean:.2f} +- {sd:.2f} [CI: {ci:.2f}]"
        print(f"{m:<25} | {format_metric('rppo'):^22} | {format_metric('mlpppo'):^22} | {format_metric('playbook'):^22}")
    print("=" * 100)

    # Statistical significance tests at seed level
    print("\n" + "=" * 100)
    print("  STATISTICAL SIGNIFICANCE TESTS AT SEED LEVEL (n=5)")
    print("=" * 100)
    comparisons = [
        ("rppo", "mlpppo", "RecurrentPPO vs MLP-PPO"),
        ("rppo", "playbook", "RecurrentPPO vs Static Playbook"),
    ]
    
    for m in ["reward", "ttr", "ended_clean"]:
        print(f"\n--- Metric: {m.upper()} ---")
        for a1, a2, label in comparisons:
            g1 = seed_means[a1][m]
            g2 = seed_means[a2][m]
            stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
            d = compute_cohens_d(g1, g2)
            delta = compute_cliffs_delta(g1, g2)
            print(f"  {label:<32} | p-value = {p_val:.4f} | Cohen's d = {d:.3f} | Cliff's delta = {delta:.3f}")
    print("=" * 100)

if __name__ == "__main__":
    main()
