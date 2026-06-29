import argparse
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
import time
import numpy as np

# stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    pass

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    print("[WARNING] scipy is not installed. Statistical significance checks will be skipped.")
    mannwhitneyu = None

from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent
from config import HOSTS, MAX_STEPS, RESULTS_DIR

# -- Statistical Helpers -------------------------------------------------------
def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0
    
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
    if pooled_se > 0:
        return (mean1 - mean2) / pooled_se
    return 0.0

def run_stats_test(group1, group2):
    if len(group1) == 0 or len(group2) == 0:
        return 1.0, 0.0, "N/A"
    
    cohen_d = compute_cohens_d(group1, group2)
    
    if mannwhitneyu is not None:
        try:
            stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
        except Exception:
            p_val = 1.0
    else:
        p_val = 1.0
        
    if p_val > 0.05:
        sig_label = "not statistically significant"
    else:
        sig_label = f"sig (p={p_val:.4f})"
        
    return p_val, cohen_d, sig_label

# -- Episode Runners -----------------------------------------------------------
def run_playbook_evaluation(env, n_episodes, seeds):
    agent = StaticPlaybookAgent()
    metrics = {
        "reward": [],
        "ttr": [],
        "max_compromised": [],
        "ended_clean": [],
        "wasted": [],
        "restored": [],
        "investigates": [],
        "investigate_ratio": []
    }
    
    for seed in seeds:
        set_random_seed(seed)
        obs, _ = env.reset(seed=seed)
        
        for ep in range(1, n_episodes + 1):
            if ep > 1:
                obs, _ = env.reset()
            agent.reset()
            
            total_reward = 0.0
            max_comp = 0
            ttr = MAX_STEPS
            ended_clean = False
            total_wasted = 0
            total_restored = 0
            total_invested = 0
            step = 0
            ep_start = np.ones((1,), dtype=bool)
            
            while True:
                action, _ = agent.predict(
                    obs[np.newaxis, :],
                    episode_start=ep_start,
                    deterministic=True,
                )
                ep_start = np.zeros((1,), dtype=bool)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step += 1
                total_wasted += info["wasted"]
                total_restored += info["restored"]
                total_invested += info["investigated"]
                max_comp = max(max_comp, info["true_compromised"])
                
                if info["true_clean"] == len(HOSTS) and ttr == MAX_STEPS:
                    ttr = step
                    
                if terminated or truncated:
                    ended_clean = (info["true_compromised"] == 0)
                    break
                    
            metrics["reward"].append(total_reward)
            metrics["ttr"].append(ttr)
            metrics["max_compromised"].append(max_comp)
            metrics["ended_clean"].append(int(ended_clean))
            metrics["wasted"].append(total_wasted)
            metrics["restored"].append(total_restored)
            metrics["investigates"].append(total_invested)
            metrics["investigate_ratio"].append(total_invested / (step * 6))
            
    return metrics

def run_sb3_evaluation(agent_type, env, n_episodes, seeds, ablation="none"):
    metrics = {
        "reward": [],
        "ttr": [],
        "max_compromised": [],
        "ended_clean": [],
        "wasted": [],
        "restored": [],
        "investigates": [],
        "investigate_ratio": []
    }
    
    for seed in seeds:
        # Construct model path
        model_name = f"{agent_type}_seed{seed}_ablation{ablation}"
        model_path = f"results/models/{model_name}.zip"
        
        if not os.path.exists(model_path):
            # Fallback to seed 0 or default model
            if agent_type == "rppo":
                if os.path.exists("results/models/rppo_seed0_ablationnone.zip"):
                    model_path = "results/models/rppo_seed0_ablationnone.zip"
                elif os.path.exists("rppo_irs_final.zip"):
                    model_path = "rppo_irs_final.zip"
            elif agent_type == "mlpppo":
                if os.path.exists("results/models/mlpppo_seed0_ablationnone.zip"):
                    model_path = "results/models/mlpppo_seed0_ablationnone.zip"
            
            if not os.path.exists(model_path):
                print(f"  [Skip] Model not found: {model_path} (Ablation: {ablation})")
                continue
        
        print(f"  Evaluating {agent_type.upper()} seed {seed} using '{model_path}' ...")
        
        custom_objects = {
            "action_space": env.action_space,
            "observation_space": env.observation_space
        }
        if agent_type == "rppo":
            model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
        else:
            model = PPO.load(model_path, custom_objects=custom_objects)
            
        set_random_seed(seed)
        obs, _ = env.reset(seed=seed)
        
        for ep in range(1, n_episodes + 1):
            if ep > 1:
                obs, _ = env.reset()
                
            lstm_state = None
            ep_start = np.ones((1,), dtype=bool)
            
            total_reward = 0.0
            max_comp = 0
            ttr = MAX_STEPS
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
                    ep_start = np.zeros((1,), dtype=bool)
                    action_step = action[0]
                else:
                    action, _ = model.predict(
                        obs[np.newaxis, :],
                        deterministic=True,
                    )
                    action_step = action[0]
                    
                obs, reward, terminated, truncated, info = env.step(action_step)
                
                total_reward += reward
                step += 1
                total_wasted += info["wasted"]
                total_restored += info["restored"]
                total_invested += info["investigated"]
                max_comp = max(max_comp, info["true_compromised"])
                
                if info["true_clean"] == len(HOSTS) and ttr == MAX_STEPS:
                    ttr = step
                    
                if terminated or truncated:
                    ended_clean = (info["true_compromised"] == 0)
                    break
                    
            metrics["reward"].append(total_reward)
            metrics["ttr"].append(ttr)
            metrics["max_compromised"].append(max_comp)
            metrics["ended_clean"].append(int(ended_clean))
            metrics["wasted"].append(total_wasted)
            metrics["restored"].append(total_restored)
            metrics["investigates"].append(total_invested)
            metrics["investigate_ratio"].append(total_invested / (step * 6))
            
    return metrics

# -- Results Printing Helpers --------------------------------------------------
def print_comparison_results(rppo, mlp, playbook):
    keys = ["reward", "ttr", "max_compromised", "wasted", "restored", "investigates", "investigate_ratio", "ended_clean"]
    labels = {
        "reward": "Mean Cumulative Reward",
        "ttr": "Time to Recovery (steps)",
        "max_compromised": "Max Compromised Hosts",
        "wasted": "Mean Wasted Restores",
        "restored": "Successful Restores",
        "investigates": "Total Investigates",
        "investigate_ratio": "Investigation/step Ratio",
        "ended_clean": "Episodes Ended Clean (%)"
    }
    
    print("\n" + "=" * 100)
    print(f"  {'Metric':<30} | {'RecurrentPPO':^18} | {'MLP-PPO':^18} | {'Static Playbook':^18}")
    print("=" * 100)
    
    for k in keys:
        rppo_vals = rppo.get(k, [])
        mlp_vals = mlp.get(k, [])
        pb_vals = playbook.get(k, [])
        
        def format_metric(vals, is_pct=False):
            if len(vals) == 0:
                return "N/A"
            mean = np.mean(vals)
            std = np.std(vals)
            if is_pct:
                return f"{mean*100:.1f}% +- {std*100:.1f}%"
            return f"{mean:.2f} +- {std:.2f}"
            
        is_pct = (k == "ended_clean")
        print(f"  {labels[k]:<30} | {format_metric(rppo_vals, is_pct):^18} | "
              f"{format_metric(mlp_vals, is_pct):^18} | {format_metric(pb_vals, is_pct):^18}")
        
    print("=" * 100)
    print("\n" + "=" * 100)
    print("  STATISTICAL SIGNIFICANCE COMPARISONS")
    print("=" * 100)
    print(f"  {'Metric':<25} | {'RPPO vs MLP-PPO':^22} | {'RPPO vs Playbook':^22} | {'MLP-PPO vs Playbook':^22}")
    print("-" * 100)
    
    for k in keys:
        rppo_vals = rppo.get(k, [])
        mlp_vals = mlp.get(k, [])
        pb_vals = playbook.get(k, [])
        
        _, d1, sig1 = run_stats_test(rppo_vals, mlp_vals)
        _, d2, sig2 = run_stats_test(rppo_vals, pb_vals)
        _, d3, sig3 = run_stats_test(mlp_vals, pb_vals)
        
        def format_comp(d, sig):
            if sig == "N/A":
                return "N/A"
            if "not statistically significant" in sig:
                return "Not Sig"
            return f"Sig (d={d:.2f})"
            
        print(f"  {labels[k]:<25} | {format_comp(d1, sig1):^22} | {format_comp(d2, sig2):^22} | {format_comp(d3, sig3):^22}")
        
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description="Multi-seed Evaluation and Statistical Reporting")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per seed to evaluate")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seed list")
    parser.add_argument("--ablation", type=str, choices=["none", "A", "B", "C", "D"], default="none",
                        help="Run ablation evaluation instead of comparative evaluation")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    env = SchoolIRSEnv()
    
    print("\n" + "=" * 70)
    print(f"  IRS-RL Comparative Evaluation Pipeline")
    print(f"  Episodes per seed : {args.episodes}")
    print(f"  Seeds evaluated   : {seeds}")
    print(f"  Ablation Mode     : {args.ablation}")
    print("=" * 70)
    
    if args.ablation == "none":
        print("\n[1/3] Evaluating Static Playbook baseline ...")
        playbook_metrics = run_playbook_evaluation(env, args.episodes, seeds)
        
        print("\n[2/3] Evaluating RecurrentPPO models ...")
        rppo_metrics = run_sb3_evaluation("rppo", env, args.episodes, seeds)
        
        print("\n[3/3] Evaluating MLP-PPO baseline models ...")
        mlp_metrics = run_sb3_evaluation("mlpppo", env, args.episodes, seeds)
        
        print_comparison_results(rppo_metrics, mlp_metrics, playbook_metrics)
    else:
        # Run Ablation metric reporting
        print(f"\nEvaluating Ablation Variant {args.ablation} ...")
        rppo_metrics = run_sb3_evaluation("rppo", env, args.episodes, seeds, ablation=args.ablation)
        
        keys = ["reward", "wasted", "ended_clean", "ttr", "investigate_ratio"]
        print("\n" + "=" * 80)
        print(f"  Ablation Variant {args.ablation} Results Summary")
        print("=" * 80)
        for k in keys:
            vals = rppo_metrics.get(k, [])
            if len(vals) == 0:
                print(f"  {k:<20}: N/A")
                continue
            mean = np.mean(vals)
            std = np.std(vals)
            if k == "ended_clean":
                print(f"  {k:<20}: {mean*100:.2f}% +- {std*100:.2f}%")
            else:
                print(f"  {k:<20}: {mean:.2f} +- {std:.2f}")
        print("=" * 80)

if __name__ == "__main__":
    main()
