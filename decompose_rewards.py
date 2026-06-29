import os
import sys
import numpy as np
from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent
from config import MODEL_PATH_RPPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("[Error] sb3-contrib is required.")
    sys.exit(1)

def run_decomposition(n_episodes=50):
    env = SchoolIRSEnv()
    
    # Components to track
    components = [
        "reward_step_penalty",
        "reward_clean_host",
        "reward_compromise_penalty",
        "reward_downtime_penalty",
        "reward_restore_bonus",
        "reward_wasted_restore",
        "reward_investigate_success",
        "reward_invalid_action",
        "reward_investigate_cost",
        "reward_do_nothing_bonus",
    ]
    
    results = {
        "RecurrentPPO": {c: [] for c in components},
        "StaticPlaybook": {c: [] for c in components}
    }
    
    # 1. Run RecurrentPPO
    model = RecurrentPPO.load(MODEL_PATH_RPPO)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        lstm_state = None
        ep_start = np.ones((1,), dtype=bool)
        ep_sums = {c: 0.0 for c in components}
        
        while True:
            action, lstm_state = model.predict(
                obs[np.newaxis, :],
                state=lstm_state,
                episode_start=ep_start,
                deterministic=True,
            )
            ep_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action[0])
            
            for c in components:
                ep_sums[c] += info.get(c, 0.0)
                
            if terminated or truncated:
                break
        
        for c in components:
            results["RecurrentPPO"][c].append(ep_sums[c])
            
    # 2. Run StaticPlaybook
    playbook = StaticPlaybookAgent()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        playbook.reset()
        ep_start = np.ones((1,), dtype=bool)
        ep_sums = {c: 0.0 for c in components}
        
        while True:
            action, _ = playbook.predict(
                obs[np.newaxis, :],
                episode_start=ep_start,
                deterministic=True,
            )
            ep_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            
            for c in components:
                ep_sums[c] += info.get(c, 0.0)
                
            if terminated or truncated:
                break
        
        for c in components:
            results["StaticPlaybook"][c].append(ep_sums[c])
            
    # Print Results
    print("\n" + "=" * 80)
    print("  REWARD DECOMPOSITION SUMMARY (Average per Episode over 50 Episodes)")
    print("=" * 80)
    print(f"  {'Reward Component':<30} | {'RecurrentPPO':>15} | {'StaticPlaybook':>15} | {'Difference':>15}")
    print("-" * 80)
    
    for c in components:
        mean_rppo = np.mean(results["RecurrentPPO"][c])
        mean_pb = np.mean(results["StaticPlaybook"][c])
        diff = mean_rppo - mean_pb
        print(f"  {c:<30} | {mean_rppo:>15.2f} | {mean_pb:>15.2f} | {diff:>15.2f}")
    
    print("=" * 80)

if __name__ == "__main__":
    run_decomposition()
