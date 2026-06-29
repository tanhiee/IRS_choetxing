"""
train.py – Enhanced
---------------------
Improvements over the original:
  - All hyperparameters sourced from config.py
  - EvalCallback: saves the best model checkpoint every EVAL_FREQ steps
  - RewardLoggerCallback: writes per-episode cumulative rewards to CSV
  - --quick flag: runs only QUICK_TIMESTEPS for fast smoke-testing
  - Graceful KeyboardInterrupt handling
"""

import argparse
import csv
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_checker import check_env

from CybORG import CybORG
from school_env import SchoolScenarioGenerator
from wrapper import SchoolIRSWrapper
from config import (
    PPO_PARAMS,
    TOTAL_TIMESTEPS,
    QUICK_TIMESTEPS,
    EVAL_FREQ,
    EVAL_EPISODES,
    MODEL_PATH,
    TRAINING_REWARDS_CSV,
    RESULTS_DIR,
)


# ── Custom Callback: log per-episode reward to CSV ────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """Records the cumulative episode reward every time an episode ends."""

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path     = csv_path
        self._episode_reward = 0.0
        self._episode_num    = 0
        self._file   = None
        self._writer = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file   = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestep", "episode", "reward"])

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        self._episode_reward += reward
        if done:
            self._episode_num += 1
            self._writer.writerow(
                [self.num_timesteps, self._episode_num, self._episode_reward]
            )
            self._file.flush()
            self._episode_reward = 0.0
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env():
    sg         = SchoolScenarioGenerator()
    cyborg_env = CybORG(scenario_generator=sg)
    return SchoolIRSWrapper(cyborg_env)


# ── Main Training Function ────────────────────────────────────────────────────

def train(quick: bool = False):
    total_steps = QUICK_TIMESTEPS if quick else TOTAL_TIMESTEPS

    print("\n" + "=" * 70)
    print(f"  IRS-RL Training  –  {'QUICK MODE' if quick else 'FULL MODE'}")
    print(f"  Timesteps : {total_steps:,}")
    print("=" * 70)

    # ── Environment ──────────────────────────────────────────────────────────
    env = make_env()

    print("\n[1/4] Checking environment compatibility …")
    check_env(env, warn=True)
    print("      ✓ Environment OK\n")

    # ── Eval env (separate instance, not reset during training) ──────────────
    eval_env = make_env()

    # ── Callbacks ────────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval_results/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    reward_logger = RewardLoggerCallback(csv_path=TRAINING_REWARDS_CSV)

    callbacks = CallbackList([eval_callback, reward_logger])

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/4] Initialising PPO model …")
    model = PPO(
        policy          = PPO_PARAMS["policy"],
        env             = env,
        verbose         = PPO_PARAMS["verbose"],
        tensorboard_log = PPO_PARAMS["tensorboard_log"],
        learning_rate   = PPO_PARAMS["learning_rate"],
        n_steps         = PPO_PARAMS["n_steps"],
        batch_size      = PPO_PARAMS["batch_size"],
        n_epochs        = PPO_PARAMS["n_epochs"],
        gamma           = PPO_PARAMS["gamma"],
        gae_lambda      = PPO_PARAMS["gae_lambda"],
        clip_range      = PPO_PARAMS["clip_range"],
        ent_coef        = PPO_PARAMS["ent_coef"],
        seed            = PPO_PARAMS["seed"],
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"[3/4] Training for {total_steps:,} timesteps …\n")
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted – saving current model …")

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n[4/4] Model saved → {MODEL_PATH}.zip")
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the IRS-RL PPO agent.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Run a short smoke-test ({QUICK_TIMESTEPS:,} steps instead of {TOTAL_TIMESTEPS:,})",
    )
    args = parser.parse_args()
    train(quick=args.quick)