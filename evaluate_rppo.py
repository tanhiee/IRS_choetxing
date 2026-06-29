"""
evaluate_rppo.py  —  RecurrentPPO vs Static Playbook Evaluation
================================================================

Key difference from evaluate.py (legacy PPO):
  RecurrentPPO maintains an LSTM hidden state across timesteps within
  each episode.  You MUST pass `state` and `episode_start` to .predict()
  at every step, and reset them at episode boundaries — otherwise the LSTM
  cannot leverage temporal context and will perform poorly.

This script handles that correctly.

Usage
-----
    python evaluate_rppo.py
    python evaluate_rppo.py --model path/to/custom_model
    python evaluate_rppo.py --episodes 100
"""

import argparse
import csv
import os
import sys
try:
    import numpy._core.numeric as _numeric
    import numpy._core as _core
except ImportError:
    import numpy.core.numeric as _numeric
    import numpy.core as _core
    sys.modules['numpy._core.numeric'] = _numeric
    sys.modules['numpy._core'] = _core
import time

import numpy as np

from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent
from config import (
    MODEL_PATH_RPPO,
    BENCHMARK_EPISODES,
    MAX_STEPS,
    RESULTS_DIR,
    HOSTS,
)

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:
    raise ImportError(
        "sb3-contrib is required.\n  pip install sb3-contrib"
    ) from exc


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_rppo_episodes(model, env, n_episodes: int, agent_name: str) -> list[dict]:
    """
    Evaluate a RecurrentPPO model for n_episodes with correct LSTM state handling.

    RecurrentPPO passes the LSTM hidden state (h, c) between steps.
    At each episode start we set episode_start=True so the policy zeros
    its hidden state.  Within an episode episode_start=False, allowing
    the LSTM to accumulate evidence across steps.

    The LSTM's temporal memory is what allows it to exploit the
    Investigate → observe → Restore pattern across two timesteps.
    """
    results = []

    for ep in range(1, n_episodes + 1):
        obs, _   = env.reset()
        # LSTM initial state — None signals the policy to use zero init
        lstm_state  = None
        # episode_start tells the LSTM to zero h,c at the episode boundary
        ep_start    = np.ones((1,), dtype=bool)

        total_reward    = 0.0
        max_compromised = 0
        ttr             = MAX_STEPS
        ended_clean     = False
        total_wasted    = 0
        total_restored  = 0
        total_invested  = 0
        step            = 0

        while True:
            # Pass lstm_state so the LSTM can carry its history forward.
            # The obs needs a batch dimension for the LSTM policy.
            action, lstm_state = model.predict(
                obs[np.newaxis, :],
                state=lstm_state,
                episode_start=ep_start,
                deterministic=True,
            )
            ep_start = np.zeros((1,), dtype=bool)   # only True on first step

            obs, reward, terminated, truncated, info = env.step(action[0])

            total_reward   += reward
            step           += 1
            total_wasted   += info["wasted"]
            total_restored += info["restored"]
            total_invested += info["investigated"]

            true_comp = info["true_compromised"]
            max_compromised = max(max_compromised, true_comp)

            if info["true_clean"] == len(HOSTS) and ttr == MAX_STEPS:
                ttr = step

            if terminated or truncated:
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
            "total_invested":   total_invested,
        })

        sys.stdout.write(
            f"\r  [{agent_name}] Episode {ep:>3}/{n_episodes}  ({ep/n_episodes*100:.0f}%)"
        )
        sys.stdout.flush()

    print()
    return results


def run_playbook_episodes(agent: StaticPlaybookAgent, env, n_episodes: int, agent_name: str) -> list[dict]:
    """
    Evaluate the rule-based StaticPlaybookAgent for n_episodes.
    No LSTM state management needed; but agent.reset() is called per episode.
    """
    results = []

    for ep in range(1, n_episodes + 1):
        obs, _   = env.reset()
        agent.reset()

        total_reward    = 0.0
        max_compromised = 0
        ttr             = MAX_STEPS
        ended_clean     = False
        total_wasted    = 0
        total_restored  = 0
        total_invested  = 0
        step            = 0

        ep_start = np.ones((1,), dtype=bool)

        while True:
            action, _ = agent.predict(
                obs[np.newaxis, :],
                episode_start=ep_start,
                deterministic=True,
            )
            ep_start = np.zeros((1,), dtype=bool)

            # StaticPlaybookAgent returns shape (N,) directly — no batch dim —
            # so use `action` directly (not action[0] which would be a scalar).
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward   += reward
            step           += 1
            total_wasted   += info["wasted"]
            total_restored += info["restored"]
            total_invested += info["investigated"]

            true_comp = info["true_compromised"]
            max_compromised = max(max_compromised, true_comp)

            if info["true_clean"] == len(HOSTS) and ttr == MAX_STEPS:
                ttr = step

            if terminated or truncated:
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
            "total_invested":   total_invested,
        })

        sys.stdout.write(
            f"\r  [{agent_name}] Episode {ep:>3}/{n_episodes}  ({ep/n_episodes*100:.0f}%)"
        )
        sys.stdout.flush()

    print()
    return results


# ── Summary Printer ────────────────────────────────────────────────────────────

def print_summary(results: list[dict], agent_name: str):
    def col(key):
        return [r[key] for r in results]

    rewards   = col("total_reward")
    ttrs      = col("time_to_recovery")
    comps     = col("max_compromised")
    cleans    = col("ended_clean")
    wasteds   = col("total_wasted")
    restores  = col("total_restored")
    invested  = col("total_invested")

    w = 35
    print(f"\n  == {agent_name} ==")
    print(f"  {'Metric':<{w}} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*75}")
    for label, data in [
        ("Total Reward",             rewards),
        ("Time to Recovery (steps)", ttrs),
        ("Max Compromised (true)",   comps),
        ("Total Wasted Restores",    wasteds),
        ("Total Successful Restores",restores),
        ("Total Investigates",       invested),
    ]:
        print(
            f"  {label:<{w}} {np.mean(data):>10.2f} {np.std(data):>10.2f} "
            f"{np.min(data):>10.1f} {np.max(data):>10.1f}"
        )
    print(f"  {'Episodes Ended Clean (%)':<{w}} {np.mean(cleans)*100:>10.1f}%")


# -- Main ----------------------------------------------------------------------

def evaluate(model_path: str, n_episodes: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  IRS-RL Evaluation (RecurrentPPO)  -  {n_episodes} episodes/agent")
    print("=" * 70)

    all_results = []

    # -- RecurrentPPO Agent ----------------------------------------------------
    model_file = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
    print(f"\n[1/2] Loading RecurrentPPO from '{model_file}' ...")
    rppo_env = SchoolIRSEnv()
    custom_objects = {
        "action_space": rppo_env.action_space,
        "observation_space": rppo_env.observation_space
    }
    model   = RecurrentPPO.load(model_path, custom_objects=custom_objects)

    print("  Running RecurrentPPO agent ...")
    t0 = time.perf_counter()
    rppo_results = run_rppo_episodes(model, rppo_env, n_episodes, "RecurrentPPO")
    print(f"  Completed in {time.perf_counter()-t0:.1f}s")
    print_summary(rppo_results, "RecurrentPPO")
    all_results.extend(rppo_results)

    # -- Static Playbook Baseline -----------------------------------------------
    print(f"\n[2/2] Running Static Playbook baseline ...")
    playbook     = StaticPlaybookAgent()
    playbook_env = SchoolIRSEnv()

    t0 = time.perf_counter()
    bl_results = run_playbook_episodes(playbook, playbook_env, n_episodes, "StaticPlaybook")
    print(f"  Completed in {time.perf_counter()-t0:.1f}s")
    print_summary(bl_results, "StaticPlaybook")
    all_results.extend(bl_results)

    # -- Save CSV ---------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "evaluation_rppo.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Results saved -> {csv_path}")
    except PermissionError:
        fallback_path = os.path.join(RESULTS_DIR, f"evaluation_rppo_{int(time.time())}.csv")
        with open(fallback_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  [WARNING] original file locked. Results saved to fallback -> {fallback_path}")
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the RecurrentPPO IRS agent vs Static Playbook."
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH_RPPO,
        help=f"Path to the RecurrentPPO model (default: {MODEL_PATH_RPPO})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=BENCHMARK_EPISODES,
        help=f"Episodes per agent (default: {BENCHMARK_EPISODES})",
    )
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
