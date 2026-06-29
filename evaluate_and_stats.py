import sys
import os
import csv
import numpy as np
import random
from scipy.stats import mannwhitneyu

# --- NumPy Compatibility Layer ---
try:
    import numpy._core.numeric as _numeric
except ImportError:
    import numpy.core.numeric as _numeric
    sys.modules['numpy._core.numeric'] = _numeric

# --- Patch BitGenerators ---
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

from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    pass

from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent
import config

SEEDS = [0, 1, 2, 3, 4]
N_EPISODES = 50

# --- Helper Functions ---
def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
    return (mean1 - mean2) / pooled_se if pooled_se > 0 else 0.0

def compute_cliffs_delta(group1, group2):
    """
    Compute Cliff's Delta effect size.
    d = (sum_{i, j} [x_i > y_j] - [x_i < y_j]) / (n1 * n2)
    """
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
    """
    Compute 95% Confidence Interval for a sample of size 5 using Student's t-distribution.
    t_critical for df=4 at alpha=0.05 (two-tailed) is 2.776.
    CI = Mean +- 2.776 * (SD / sqrt(5))
    """
    n = len(data)
    if n < 2:
        return 0.0
    mean = np.mean(data)
    sd = np.std(data, ddof=1)
    margin = 2.776 * (sd / np.sqrt(n))
    return margin

# --- Evaluation loop ---
def evaluate_agent(agent_type, env):
    all_episodes = []
    
    for seed in SEEDS:
        print(f"Evaluating {agent_type} - Seed {seed}...")
        np.random.seed(seed)
        random.seed(seed)
        
        # Load model/agent
        if agent_type == "rppo":
            model_path = f"results/models/rppo_seed{seed}_ablationnone.zip"
            custom_objects = {"action_space": env.action_space, "observation_space": env.observation_space}
            model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
        elif agent_type == "mlpppo":
            model_path = f"results/models/mlpppo_seed{seed}_ablationnone.zip"
            custom_objects = {"action_space": env.action_space, "observation_space": env.observation_space}
            model = PPO.load(model_path, custom_objects=custom_objects)
        elif agent_type == "playbook":
            model = StaticPlaybookAgent()
            
        obs, _ = env.reset(seed=seed)
        
        for ep in range(1, N_EPISODES + 1):
            if ep > 1:
                obs, _ = env.reset()
                
            if agent_type == "playbook":
                model.reset()
                
            lstm_state = None
            ep_start = np.ones((1,), dtype=bool)
            
            total_reward = 0.0
            max_comp = 0
            ttr = config.MAX_STEPS
            ended_clean = False
            total_wasted = 0
            total_restored = 0
            total_invested = 0
            step = 0
            
            while True:
                if agent_type == "rppo":
                    action, lstm_state = model.predict(
                        obs[np.newaxis, :],
                        state=lstm_state,
                        episode_start=ep_start,
                        deterministic=True,
                    )
                    act = action[0]
                elif agent_type == "mlpppo":
                    action, _ = model.predict(
                        obs[np.newaxis, :],
                        deterministic=True,
                    )
                    act = action[0]
                elif agent_type == "playbook":
                    action, _ = model.predict(
                        obs[np.newaxis, :],
                        episode_start=ep_start,
                        deterministic=True,
                    )
                    act = action
                    
                ep_start = np.zeros((1,), dtype=bool)
                obs, reward, terminated, truncated, info = env.step(act)
                
                total_reward += reward
                step += 1
                total_wasted += info["wasted"]
                total_restored += info["restored"]
                total_invested += info["investigated"]
                max_comp = max(max_comp, info["true_compromised"])
                
                if info["true_clean"] == len(config.HOSTS) and ttr == config.MAX_STEPS:
                    ttr = step
                    
                if terminated or truncated:
                    ended_clean = (info["true_compromised"] == 0)
                    break
                    
            all_episodes.append({
                "agent": agent_type,
                "seed": seed,
                "episode": ep,
                "reward": total_reward,
                "ttr": ttr,
                "max_compromised": max_comp,
                "ended_clean": int(ended_clean),
                "wasted": total_wasted,
                "restored": total_restored,
                "investigated": total_invested,
                "investigate_ratio": total_invested / (step * 6)
            })
            
    return all_episodes

def main():
    env = SchoolIRSEnv(use_belief=False)
    os.makedirs("results", exist_ok=True)
    
    # 1. Run all evaluations
    rppo_episodes = evaluate_agent("rppo", env)
    mlpppo_episodes = evaluate_agent("mlpppo", env)
    playbook_episodes = evaluate_agent("playbook", env)
    
    all_episodes = rppo_episodes + mlpppo_episodes + playbook_episodes
    
    # Save raw episode results
    csv_path = "results/evaluation_all_episodes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_episodes[0].keys())
        writer.writeheader()
        writer.writerows(all_episodes)
    print(f"Saved raw episode results to {csv_path}")
    
    # 2. Compute seed-level stats
    # For each agent, group by seed, compute mean for each seed. This yields 5 seed means per agent.
    metrics = ["reward", "ttr", "max_compromised", "wasted", "restored", "ended_clean", "investigate_ratio"]
    seed_means = {}
    
    for agent in ["rppo", "mlpppo", "playbook"]:
        seed_means[agent] = {m: [] for m in metrics}
        agent_eps = [e for e in all_episodes if e["agent"] == agent]
        
        for seed in SEEDS:
            seed_eps = [e for e in agent_eps if e["seed"] == seed]
            for m in metrics:
                mean_val = np.mean([e[m] for e in seed_eps])
                seed_means[agent][m].append(mean_val)
                
    # 3. Print updated Table I (with 95% Confidence Interval based on n=5 seed means)
    print("\n" + "=" * 100)
    print("  UPDATED TABLE I: SEED-LEVEL STATISTICS (n=5 seeds)")
    print("=" * 100)
    print(f"{'Chỉ số':<28} | {'RecurrentPPO (LSTM)':^24} | {'MLP-PPO (Không bộ nhớ)':^24} | {'Static Playbook (Luật)':^24}")
    print("-" * 100)
    
    metric_labels = {
        "reward": "Phần thưởng tích lũy",
        "ttr": "Thời gian Phục hồi (TTR)",
        "max_compromised": "Số máy nhiễm tối đa",
        "wasted": "Số lần Restore lỗi (Wasted)",
        "restored": "Khôi phục thành công",
        "ended_clean": "Tập kết thúc sạch (%)",
        "investigate_ratio": "Tỉ lệ điều tra mỗi bước"
    }
    
    for m in metrics:
        def format_metric(agent_key):
            vals = seed_means[agent_key][m]
            mean = np.mean(vals)
            sd = np.std(vals, ddof=1)
            ci = compute_95_ci(vals)
            if m == "ended_clean":
                # Convert to percent
                return f"{mean*100:.2f}% ± {sd*100:.2f}% [CI: {ci*100:.2f}%]"
            return f"{mean:.2f} ± {sd:.2f} [CI: {ci:.2f}]"
            
        print(f"{metric_labels[m]:<28} | {format_metric('rppo'):^24} | {format_metric('mlpppo'):^24} | {format_metric('playbook'):^24}")
    print("=" * 100)
    
    # 4. Statistical tests at the seed level (n=5 per group)
    print("\n" + "=" * 100)
    print("  STATISTICAL SIGNIFICANCE TESTS AT SEED LEVEL (n=5)")
    print("=" * 100)
    
    comparisons = [
        ("rppo", "mlpppo", "RecurrentPPO vs MLP-PPO"),
        ("rppo", "playbook", "RecurrentPPO vs Static Playbook"),
    ]
    
    for m in ["reward", "ttr", "ended_clean"]:
        print(f"\n--- Chỉ số: {m.upper()} ---")
        for a1, a2, label in comparisons:
            g1 = seed_means[a1][m]
            g2 = seed_means[a2][m]
            
            # Mann-Whitney U
            stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
            # Cohen's d
            d = compute_cohens_d(g1, g2)
            # Cliff's delta
            delta = compute_cliffs_delta(g1, g2)
            
            print(f"  {label:<32} | p-value = {p_val:.4f} | Cohen's d = {d:.3f} | Cliff's delta = {delta:.3f}")
            
    print("=" * 100)
    
    # Save seed-level statistics for python scripts/paper updating
    summary_path = "results/table1_multiseed_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(seed_means, f, indent=2)
    print(f"Saved seed-level means to {summary_path}")

if __name__ == "__main__":
    main()
