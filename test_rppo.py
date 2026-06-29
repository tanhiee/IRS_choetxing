"""
test_rppo.py  —  Step-by-step RPPO model inspector
====================================================
Loads a trained RecurrentPPO model and runs it for one (or more) episodes,
printing each timestep's full state: observation, LSTM decision, action taken,
and reward received.

This is the diagnostic tool — use it to understand WHAT the agent has learned.

Usage
-----
    # Test default model (rppo_irs_final.zip), 1 episode:
    python test_rppo.py

    # Test best checkpoint, 3 episodes, 30-step limit:
    python test_rppo.py --model logs/best_rppo/best_model --episodes 3 --steps 30

    # Test and pause 0.5s between steps (easier to read):
    python test_rppo.py --delay 0.5
"""

import argparse
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
from config import HOSTS, MODEL_PATH_RPPO, MAX_STEPS

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("Run: pip install sb3-contrib")

# ── Action label map ──────────────────────────────────────────────────────────
ACTION_LABELS = {0: "DoNothing   ", 1: "Investigate ", 2: "RESTORE *   "}
OBS_CHANNELS  = 3   # must match wrapper.py

# ── ANSI colour helpers (Windows-safe fallback) ───────────────────────────────
try:
    import os
    os.system("")   # enable ANSI on Windows terminal
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except Exception:
    RED = YELLOW = GREEN = CYAN = GREY = BOLD = RESET = ""


def host_status_str(env: SchoolIRSEnv, host: str) -> str:
    """Return a colour-coded status string for a host."""
    if host in env.compromised:
        age = env.compromise_age.get(host, 1)
        return f"{RED}COMPROMISED(age={age}){RESET}"
    elif host in env.downtime:
        ticks = env.downtime[host]
        return f"{YELLOW}DOWNTIME({ticks}){RESET}"
    else:
        return f"{GREEN}clean{RESET}"


def decode_obs(obs: np.ndarray) -> list[dict]:
    """Decode the flat obs vector into per-host dicts."""
    decoded = []
    for i, host in enumerate(HOSTS):
        base = i * OBS_CHANNELS
        decoded.append({
            "host":        host,
            "alert":       int(obs[base + 0]),
            "inv_prev":    int(obs[base + 1]),
            "downtime_f":  round(float(obs[base + 2]), 2),
        })
    return decoded


def run_episode(
    model:     "RecurrentPPO",
    env:       SchoolIRSEnv,
    ep_num:    int,
    max_steps: int,
    delay:     float,
):
    obs, _ = env.reset()

    # LSTM state — None = zero-initialise at episode start
    lstm_state  = None
    ep_start    = np.ones((1,), dtype=bool)

    total_reward   = 0.0
    action_counts  = {0: 0, 1: 0, 2: 0}

    print(f"\n{'='*72}")
    print(f"  {BOLD}Episode {ep_num}{RESET}  -  max {max_steps} steps")
    print(f"{'='*72}")
    print(f"  {'Step':<5} {'Host':<14} {'Alert':>5} {'InvPrev':>7} {'DT':>4}"
          f"  {'Action':<15} {'True State':<22} {'Reward':>8}")
    print(f"  {'-'*72}")

    for step in range(1, max_steps + 1):
        # ── LSTM-aware prediction ───────────────────────────────────────────
        # The hidden state (h, c) carries memory from previous steps.
        # episode_start=True zeros the state at episode boundary.
        action_batch, lstm_state = model.predict(
            obs[np.newaxis, :],
            state=lstm_state,
            episode_start=ep_start,
            deterministic=True,
        )
        ep_start = np.zeros((1,), dtype=bool)
        actions  = action_batch[0]   # strip batch dim → shape (N,)

        # ── Decode obs BEFORE stepping (shows what agent saw) ───────────────
        decoded = decode_obs(obs)

        # ── Step the environment ────────────────────────────────────────────
        obs_next, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        # ── Print per-host rows ─────────────────────────────────────────────
        first_row = True
        for i, host in enumerate(HOSTS):
            act_id   = int(actions[i])
            act_str  = ACTION_LABELS[act_id]
            d        = decoded[i]
            true_st  = host_status_str(env, host)

            action_counts[act_id] += 1

            # Colour the action label
            if act_id == 2:
                act_display = f"{RED}{act_str}{RESET}"
            elif act_id == 1:
                act_display = f"{CYAN}{act_str}{RESET}"
            else:
                act_display = f"{GREY}{act_str}{RESET}"

            step_col = f"{step:<5}" if first_row else f"{'':5}"
            first_row = False

            print(
                f"  {step_col} {host:<14} {d['alert']:>5} {d['inv_prev']:>7} "
                f"{d['downtime_f']:>4}  {act_display:<24} {true_st:<30}"
                + (f"  {reward:>+8.2f}" if i == 0 else "")
            )

        print(f"  {'':5} {'-'*67}  Step total: {reward:>+.2f}")

        if delay > 0:
            time.sleep(delay)

        obs = obs_next
        if terminated or truncated:
            break

    # ── Episode summary ─────────────────────────────────────────────────────
    print(f"\n  {'-'*72}")
    print(f"  {BOLD}Episode {ep_num} Summary{RESET}")
    print(f"  {'-'*30}")
    print(f"  Total steps        : {step}")
    print(f"  Total reward       : {total_reward:>+.2f}")
    print(f"  Hosts compromised  : {info['true_compromised']}")
    print(f"  Hosts in downtime  : {info['hosts_in_downtime']}")
    print(f"  Successful restores: {info['restored']}")
    print(f"  Wasted restores    : {info['wasted']}")
    print(f"  Actions breakdown  :")
    total_actions = sum(action_counts.values())
    for act_id, label in [(0,"DoNothing"),(1,"Investigate"),(2,"Restore")]:
        n = action_counts[act_id]
        pct = n / total_actions * 100 if total_actions else 0
        bar = "█" * int(pct / 3)
        print(f"    {label:<15} {n:>4}  ({pct:5.1f}%)  {bar}")
    print(f"  {'-'*72}")

    return total_reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step-by-step RPPO model inspector."
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH_RPPO,
        help=f"Path to RecurrentPPO .zip (default: {MODEL_PATH_RPPO})",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Max steps per episode to display (default: 50, full=200)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to pause between steps, e.g. 0.3 (default: 0)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full 200-step episodes (overrides --steps)",
    )
    args = parser.parse_args()
    max_steps = MAX_STEPS if args.full else args.steps
    model_path = args.model
    env = SchoolIRSEnv()
    custom_objects = {
        "action_space": env.action_space,
        "observation_space": env.observation_space
    }
    model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
    print(f"  Policy: {model.policy.__class__.__name__}")
    print(f"  LSTM hidden size: {model.policy.lstm_actor.hidden_size}")
    rewards = []

    for ep in range(1, args.episodes + 1):
        r = run_episode(model, env, ep, max_steps, args.delay)
        rewards.append(r)

    if args.episodes > 1:
        print(f"\n{'='*72}")
        print(f"  {BOLD}Overall ({args.episodes} episodes){RESET}")
        print(f"  Mean reward : {np.mean(rewards):>+.2f}")
        print(f"  Std         : {np.std(rewards):>+.2f}")
        print(f"  Min / Max   : {np.min(rewards):>+.2f} / {np.max(rewards):>+.2f}")
        print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
