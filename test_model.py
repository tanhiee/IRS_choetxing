"""
test_model.py – Enhanced & Translated to English
-------------------------------------------------
Runs a single 30-step episode with the trained PPO model and prints
a clean, colour-coded step-by-step network status table.
"""

import time
from CybORG import CybORG
from stable_baselines3 import PPO

from school_env import SchoolScenarioGenerator
from wrapper import SchoolIRSWrapper
from config import MODEL_PATH, HOSTS

# ANSI colour helpers (no external deps)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

STATE_LABELS = {
    0: (GREEN,  "Clean        "),
    1: (YELLOW, "Infected     "),
    2: (RED,    "Compromised  "),
}

def colour_status(val: int) -> str:
    col, label = STATE_LABELS[val]
    return f"{col}{label}{RESET}"


def run_test(n_steps: int = 30, delay: float = 0.4):
    # ── Environment ──────────────────────────────────────────────────────────
    sg         = SchoolScenarioGenerator()
    cyborg_env = CybORG(scenario_generator=sg)
    env        = SchoolIRSWrapper(cyborg_env)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}Loading model: {MODEL_PATH}.zip …{RESET}")
    model = PPO.load(MODEL_PATH)

    print("\n" + "=" * 80)
    print(f"{BOLD}{'IRS-RL: PPO AGENT EVALUATION  (Single Episode)':^80}{RESET}")
    print("=" * 80)
    print(f"  Testing for {n_steps} steps against FiniteStateRedAgent\n")

    obs, info = env.reset()
    total_reward = 0.0

    try:
        for step in range(1, n_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward

            action_label = (
                f"Restore {HOSTS[int(action) - 1]}" if int(action) > 0
                else "Do Nothing (Sleep)"
            )

            print(f"\n{BOLD}[ Step {step:>3} ]{RESET}  Blue action: {action_label}")
            print(f"  {'Host':<15} Status")
            print(f"  {'-'*32}")
            for i, host in enumerate(HOSTS):
                status = colour_status(int(obs[i]))
                print(f"  {host:<15} {status}")

            summary = (
                f"  Clean: {info['clean']}  |  "
                f"Infected: {info['infected']}  |  "
                f"Compromised: {info['compromised']}"
            )
            print(summary)
            print(f"  Step reward: {reward:+.2f}   |   Cumulative: {total_reward:+.2f}")

            time.sleep(delay)

            if terminated or truncated:
                print(f"\n{'>>> Episode ended <<<':^80}")
                break

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")

    print("\n" + "=" * 80)
    print(f"{BOLD}  TOTAL REWARD: {total_reward:.2f}{RESET}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_test()