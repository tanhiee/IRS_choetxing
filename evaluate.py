import os
import csv
import sys
import time
 
import numpy as np
 
from CybORG import CybORG
from stable_baselines3 import PPO
 
from school_env import SchoolScenarioGenerator
from wrapper import SchoolIRSWrapper
from baseline_agent import StaticPlaybookAgent
from config import (
    MODEL_PATH,
    BENCHMARK_EPISODES,
    MAX_STEPS,
    RESULTS_DIR,
    HOSTS,
)
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def make_env():
    sg = SchoolScenarioGenerator()
    return SchoolIRSWrapper(CybORG(scenario_generator=sg))
 
 
def run_episodes(agent, env, n_episodes: int, agent_name: str) -> list[dict]:
    """
    Run agent for n_episodes and return a list of per-episode metric dicts.
 
    Uses info["true_compromised"] and info["true_clean"] for all evaluation
    metrics to accurately capture Red's spread behavior.
    """
    results = []
 
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward    = 0.0
        max_compromised = 0
        ttr             = MAX_STEPS  # default: never fully recovered
        ended_clean     = False
        total_wasted    = 0
        total_restored  = 0
        step            = 0
 
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
 
            total_reward    += reward
            step            += 1
            total_wasted    += info["wasted"]
            total_restored  += info["restored"]
 
            # ── Use TRUE counts for all metrics ──────────────────────────────
            true_comp  = info["true_compromised"]
            true_clean = info["true_clean"]
 
            # Peak compromise: track the highest true compromised count seen
            max_compromised = max(max_compromised, true_comp)
 
            # TTR: first step where ALL hosts are truly clean simultaneously
            if true_clean == len(HOSTS) and ttr == MAX_STEPS:
                ttr = step
 
            if terminated or truncated:
                # Episode truly ended clean only if ground truth is all-clean
                ended_clean = (true_comp == 0)
                break
 
        results.append({
            "agent":            agent_name,
            "episode":          ep,
            "total_reward":     round(total_reward, 2),
            "steps":            step,
            "time_to_recovery": ttr,
            "max_compromised":  max_compromised,
            "ended_clean":      int(ended_clean),
            "total_wasted":     total_wasted,
            "total_restored":   total_restored,
        })
 
        pct = ep / n_episodes * 100
        sys.stdout.write(
            f"\r  [{agent_name}] Episode {ep:>3}/{n_episodes}  ({pct:.0f}%)"
        )
        sys.stdout.flush()
 
    print()  # newline after progress bar
    return results
 
 
def print_summary(results: list[dict], agent_name: str):
    rewards   = [r["total_reward"]     for r in results]
    ttrs      = [r["time_to_recovery"] for r in results]
    comps     = [r["max_compromised"]  for r in results]
    cleans    = [r["ended_clean"]      for r in results]
    wasteds   = [r["total_wasted"]     for r in results]
    restores  = [r["total_restored"]   for r in results]
 
    print(f"\n  ── {agent_name} ──")
    print(f"  {'Metric':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*75}")
    print(f"  {'Total Reward':<35} {np.mean(rewards):>10.2f} {np.std(rewards):>10.2f} "
          f"{np.min(rewards):>10.2f} {np.max(rewards):>10.2f}")
    print(f"  {'Time to Recovery (steps)':<35} {np.mean(ttrs):>10.1f} {np.std(ttrs):>10.1f} "
          f"{np.min(ttrs):>10} {np.max(ttrs):>10}")
    print(f"  {'Max Compromised Hosts (true)':<35} {np.mean(comps):>10.2f} {np.std(comps):>10.2f} "
          f"{np.min(comps):>10} {np.max(comps):>10}")
    print(f"  {'Episodes Ended Clean (%)':<35} {np.mean(cleans)*100:>10.1f}%")
    print(f"  {'Total Wasted Restores / Episode':<35} {np.mean(wasteds):>10.2f} {np.std(wasteds):>10.2f} "
          f"{np.min(wasteds):>10} {np.max(wasteds):>10}")
    print(f"  {'Total Successful Restores / Ep':<35} {np.mean(restores):>10.2f} {np.std(restores):>10.2f} "
          f"{np.min(restores):>10} {np.max(restores):>10}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    print("\n" + "=" * 70)
    print(f"  IRS-RL Evaluation  —  {BENCHMARK_EPISODES} episodes per agent")
    print("=" * 70)
 
    all_results = []
 
    # ── PPO Agent ─────────────────────────────────────────────────────────────
    print(f"\n[1/2] Loading PPO model from '{MODEL_PATH}.zip' …")
    ppo_model = PPO.load(MODEL_PATH)
    ppo_env   = make_env()
 
    print("  Running PPO agent …")
    start = time.perf_counter()
    ppo_results = run_episodes(ppo_model, ppo_env, BENCHMARK_EPISODES, "PPO")
    elapsed_ppo = time.perf_counter() - start
 
    print(f"  Completed in {elapsed_ppo:.1f}s")
    print_summary(ppo_results, "PPO")
    all_results.extend(ppo_results)
 
    # ── Static Playbook ───────────────────────────────────────────────────────
    print(f"\n[2/2] Running Static Playbook baseline …")
    baseline     = StaticPlaybookAgent()
    baseline_env = make_env()
 
    start = time.perf_counter()
    bl_results = run_episodes(baseline, baseline_env, BENCHMARK_EPISODES, "StaticPlaybook")
    elapsed_bl = time.perf_counter() - start
 
    print(f"  Completed in {elapsed_bl:.1f}s")
    print_summary(bl_results, "StaticPlaybook")
    all_results.extend(bl_results)
 
    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
 
    print(f"\n  Results saved → {csv_path}")
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
 
 
if __name__ == "__main__":
    evaluate()