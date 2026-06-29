import os
import csv
import sys
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    pass

from adversarial_env import AdversarialSchoolEnv
from baseline_agent import StaticPlaybookAgent
from gnn_model import GNNFeatureExtractor
import config

def convert_to_18d(obs):
    """
    Slices the 24D observation (4 channels per host) to 18D (3 channels per host)
    by dropping the belief channel, ensuring compatibility with legacy models.
    """
    # If batched
    if len(obs.shape) == 2:
        obs_reshaped = obs.reshape(-1, 6, 4)
        obs_sliced = obs_reshaped[:, :, :3]
        return obs_sliced.reshape(-1, 18)
    else:
        obs_reshaped = obs.reshape(6, 4)
        obs_sliced = obs_reshaped[:, :3]
        return obs_sliced.flatten()

def convert_to_6d(obs, env):
    """
    Converts a flat 24D observation to the legacy 6D space (0=clean, 1=alert, 2=compromised)
    expected by the original MLP-PPO model.
    """
    obs_6d = np.zeros(6, dtype=np.float32)
    obs_flat = obs.flatten()
    for i, host in enumerate(config.HOSTS):
        base = i * 4
        alert = obs_flat[base + 0]
        if host in env.compromised:
            obs_6d[i] = 2.0 # Compromised
        elif alert > 0.0:
            obs_6d[i] = 1.0 # Infected / Alert
        else:
            obs_6d[i] = 0.0 # Clean
    return obs_6d

def convert_to_multidiscrete_action(scalar_action):
    """
    Converts a legacy scalar action (0=Do Nothing, 1-6=Restore host)
    to the new MultiDiscrete space ([0,1,2] for each of the 6 hosts).
    """
    actions = np.zeros(6, dtype=np.int64)
    if scalar_action > 0:
        actions[scalar_action - 1] = 2 # Restore
    return actions

def run_evaluation(defender_agent, defender_type, env, n_episodes, attacker_agent=None):
    """
    Evaluates a defender against a rule-based or RL attacker.
    Handles LSTM state for RecurrentPPO.
    """
    results = []
    
    # Configure environmental attacker
    env.opponent_policy = attacker_agent
    
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        lstm_state = None
        ep_start = np.ones((1,), dtype=bool)
        
        total_reward = 0.0
        max_compromised = 0
        ttr = config.MAX_STEPS
        ended_clean = False
        total_wasted = 0
        total_restored = 0
        total_invested = 0
        step = 0
        
        while True:
            # Format observation for defender
            if defender_type == "gnn_belief_rppo":
                # Expects full 24D observation
                action, lstm_state = defender_agent.predict(
                    obs[np.newaxis, :],
                    state=lstm_state,
                    episode_start=ep_start,
                    deterministic=True
                )
                act = action[0]
            elif defender_type == "legacy_rppo":
                # Expects 18D observation
                obs_18d = convert_to_18d(obs)
                action, lstm_state = defender_agent.predict(
                    obs_18d[np.newaxis, :],
                    state=lstm_state,
                    episode_start=ep_start,
                    deterministic=True
                )
                act = action[0]
            elif defender_type == "legacy_mlp_ppo":
                # Expects 6D observation
                obs_6d = convert_to_6d(obs, env)
                action, _ = defender_agent.predict(
                    obs_6d[np.newaxis, :],
                    deterministic=True
                )
                act = convert_to_multidiscrete_action(action[0])
            elif defender_type == "playbook":
                # Playbook expects 18D observation
                obs_18d = convert_to_18d(obs)
                action, _ = defender_agent.predict(
                    obs_18d,
                    episode_start=ep_start,
                    deterministic=True
                )
                act = action
                
            ep_start = np.zeros((1,), dtype=bool)
            obs, reward, term, trunc, info = env.step(act)
            
            total_reward += reward
            step += 1
            total_wasted += info["wasted"]
            total_restored += info["restored"]
            total_invested += info["investigated"]
            
            true_comp = info["true_compromised"]
            max_compromised = max(max_compromised, true_comp)
            
            if info["true_clean"] == len(config.HOSTS) and ttr == config.MAX_STEPS:
                ttr = step
                
            if term or trunc:
                ended_clean = (true_comp == 0)
                break
                
        results.append({
            "reward": total_reward,
            "ttr": ttr,
            "max_compromised": max_compromised,
            "ended_clean": int(ended_clean),
            "wasted": total_wasted,
            "restored": total_restored,
            "investigated": total_invested
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Graph-Belief Adversarial RL Incident Response Defender")
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes per agent")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    env = AdversarialSchoolEnv(mode="defender")
    
    # ── Load Defender Models ──────────────────────────────────────────────────
    print("\n[Loading] Loading defender agents...")
    
    # 1. GNN-Belief Recurrent PPO (Ours)
    gnn_belief_path = "results/models/rppo_defender_adv.zip"
    if os.path.exists(gnn_belief_path):
        custom_objects = {
            "action_space": env.action_space,
            "observation_space": env.observation_space,
        }
        gnn_belief_model = RecurrentPPO.load(gnn_belief_path, custom_objects=custom_objects)
        print("  [OK] Loaded GNN-Belief Recurrent PPO Defender")
    else:
        gnn_belief_model = None
        print("  [FAIL] GNN-Belief Recurrent PPO Defender not found")
        
    # 2. Legacy Recurrent PPO
    legacy_rppo_path = "rppo_irs_final.zip"
    if os.path.exists(legacy_rppo_path):
        legacy_rppo_model = RecurrentPPO.load(legacy_rppo_path)
        print("  [OK] Loaded Legacy Recurrent PPO Defender")
    else:
        legacy_rppo_model = None
        print("  [FAIL] Legacy Recurrent PPO Defender not found")
        
    # 3. Legacy MLP PPO
    legacy_mlp_path = "ppo_school_irs_final.zip"
    if os.path.exists(legacy_mlp_path):
        legacy_mlp_model = PPO.load(legacy_mlp_path)
        print("  [OK] Loaded Legacy MLP PPO Defender")
    else:
        legacy_mlp_model = None
        print("  [FAIL] Legacy MLP PPO Defender not found")
        
    # 4. Static Playbook
    playbook_agent = StaticPlaybookAgent()
    print("  [OK] Loaded Static Playbook Agent")
    
    # ── Load Attacker Model ────────────────────────────────────────────────────
    attacker_path = "results/models/ppo_attacker_adv.zip"
    if os.path.exists(attacker_path):
        attacker_model = PPO.load(attacker_path)
        print("  [OK] Loaded Trained RL Attacker")
    else:
        attacker_model = None
        print("  [FAIL] Trained RL Attacker not found (will evaluate only against rule-based attacker)")
        
    defenders = []
    if gnn_belief_model:
        defenders.append((gnn_belief_model, "gnn_belief_rppo", "GNN-Belief Recurrent PPO (Ours)"))
    if legacy_rppo_model:
        defenders.append((legacy_rppo_model, "legacy_rppo", "Legacy Recurrent PPO"))
    if legacy_mlp_model:
        defenders.append((legacy_mlp_model, "legacy_mlp_ppo", "Legacy MLP PPO"))
    defenders.append((playbook_agent, "playbook", "Static Playbook"))
    
    attackers = [
        (None, "Rule-based Attacker"),
    ]
    if attacker_model:
        attackers.append((attacker_model, "RL Attacker (Adversarial)"))
        
    print("\n" + "=" * 80)
    print(f"  BENCHMARK EVALUATION ({args.episodes} episodes per configuration)")
    print("=" * 80)
    
    all_summary = []
    
    for att_agent, att_name in attackers:
        print(f"\n>>>> Evaluating against: {att_name} <<<<")
        print("-" * 80)
        
        for def_agent, def_type, def_name in defenders:
            print(f"  Running {def_name}...")
            t0 = time.perf_counter()
            results = run_evaluation(def_agent, def_type, env, args.episodes, attacker_agent=att_agent)
            dt = time.perf_counter() - t0
            
            rewards = [r["reward"] for r in results]
            ttrs = [r["ttr"] for r in results]
            comps = [r["max_compromised"] for r in results]
            cleans = [r["ended_clean"] for r in results]
            wasteds = [r["wasted"] for r in results]
            restores = [r["restored"] for r in results]
            
            mean_r = np.mean(rewards)
            mean_ttr = np.mean(ttrs)
            mean_comp = np.mean(comps)
            clean_pct = np.mean(cleans) * 100
            mean_w = np.mean(wasteds)
            mean_res = np.mean(restores)
            
            print(f"    Mean Reward      : {mean_r:^8.2f}")
            print(f"    TTR (steps)      : {mean_ttr:^8.1f}")
            print(f"    Peak Compromised : {mean_comp:^8.2f}")
            print(f"    End Clean (%)    : {clean_pct:^8.1f}%")
            print(f"    Wasted Restores  : {mean_w:^8.2f}")
            print(f"    Elapsed Time     : {dt:.1f}s")
            print("-" * 50)
            
            all_summary.append({
                "attacker": att_name,
                "defender": def_name,
                "reward": round(mean_r, 2),
                "ttr": round(mean_ttr, 2),
                "peak_compromised": round(mean_comp, 2),
                "end_clean_pct": round(clean_pct, 2),
                "wasted_restores": round(mean_w, 2),
                "successful_restores": round(mean_res, 2),
            })
            
    # Save benchmark results to CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/adversarial_evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary)
    print(f"\n[OK] Saved evaluation results to: {csv_path}")

if __name__ == "__main__":
    main()
