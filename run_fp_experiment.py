import os
import sys
import numpy as np
import random
from sb3_contrib import RecurrentPPO

# Import project files
import config
import wrapper
from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent

def run_evaluation(model, agent_type, env, n_episodes):
    results = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        lstm_state = None
        ep_start = np.ones((1,), dtype=bool)
        
        if agent_type == "playbook":
            model.reset()
            
        total_reward = 0.0
        step = 0
        wasted_restores = 0
        investigations = 0
        ttr = 200
        
        while True:
            if agent_type == "rppo":
                action, lstm_state = model.predict(
                    obs[np.newaxis, :],
                    state=lstm_state,
                    episode_start=ep_start,
                    deterministic=True
                )
                act = action[0]
            else:
                action, _ = model.predict(
                    obs[np.newaxis, :],
                    episode_start=ep_start,
                    deterministic=True
                )
                act = action
                
            ep_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(act)
            
            total_reward += reward
            step += 1
            wasted_restores += info["wasted"]
            investigations += info["investigated"]
            
            if info["true_clean"] == len(config.HOSTS) and ttr == 200:
                ttr = step
                
            if terminated or truncated:
                break
                
        results.append({
            "reward": total_reward,
            "ttr": ttr,
            "wasted": wasted_restores,
            "investigations": investigations
        })
        
    return results

def main():
    print("Loading model...")
    rppo = RecurrentPPO.load("rppo_irs_final")
    playbook = StaticPlaybookAgent()
    
    fp_rates = [0.05, 0.15, 0.30, 0.45]
    n_episodes = 50
    
    print(f"{'FP Rate':<10} | {'Agent':<15} | {'Mean Reward':<12} | {'Mean TTR':<10} | {'Mean Wasted':<12} | {'Mean Investigate':<16}")
    print("-" * 80)
    
    for fp in fp_rates:
        # Override FP rate in wrapper
        wrapper.FALSE_POSITIVE_RATE = fp
        config.FALSE_POSITIVE_RATE = fp
        
        # Fresh envs
        env_rppo = SchoolIRSEnv()
        env_pb = SchoolIRSEnv()
        
        # Seed for reproducibility
        env_rppo.reset(seed=42)
        env_pb.reset(seed=42)
        random.seed(42)
        np.random.seed(42)
        
        # Run RPPO
        rppo_res = run_evaluation(rppo, "rppo", env_rppo, n_episodes)
        r_rewards = [x["reward"] for x in rppo_res]
        r_ttrs = [x["ttr"] for x in rppo_res]
        r_wasted = [x["wasted"] for x in rppo_res]
        r_invest = [x["investigations"] for x in rppo_res]
        
        # Seed for playbook
        env_pb.reset(seed=42)
        random.seed(42)
        np.random.seed(42)
        
        # Run Playbook
        pb_res = run_evaluation(playbook, "playbook", env_pb, n_episodes)
        p_rewards = [x["reward"] for x in pb_res]
        p_ttrs = [x["ttr"] for x in pb_res]
        p_wasted = [x["wasted"] for x in pb_res]
        p_invest = [x["investigations"] for x in pb_res]
        
        print(f"{fp:<10.2f} | {'RecurrentPPO':<15} | {np.mean(r_rewards):<12.2f} | {np.mean(r_ttrs):<10.2f} | {np.mean(r_wasted):<12.2f} | {np.mean(r_invest):<16.2f}")
        print(f"{fp:<10.2f} | {'StaticPlaybook':<15} | {np.mean(p_rewards):<12.2f} | {np.mean(p_ttrs):<10.2f} | {np.mean(p_wasted):<12.2f} | {np.mean(p_invest):<16.2f}")
        print("-" * 80)

if __name__ == "__main__":
    main()
