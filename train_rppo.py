import argparse
import csv
import os
import sys

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env

# ── sb3-contrib: RecurrentPPO ─────────────────────────────────────────────────
try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:
    raise ImportError(
        "sb3-contrib is required for RecurrentPPO.\n"
        "Install it with:  pip install sb3-contrib"
    ) from exc

from wrapper import SchoolIRSEnv
from config import (
    RPPO_PARAMS,
    TOTAL_TIMESTEPS,
    QUICK_TIMESTEPS,
    EVAL_FREQ,
    EVAL_EPISODES,
    MODEL_PATH_RPPO,
    RPPO_REWARDS_CSV,
    RESULTS_DIR,
)


# ── Callback: per-episode reward logger ───────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Appends (timestep, episode, cumulative_reward) to a CSV file after each
    episode ends.  Compatible with RecurrentPPO's non-vectorised rollout loop.
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path        = csv_path
        self._ep_reward      = 0.0
        self._ep_num         = 0
        self._file           = None
        self._writer         = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file   = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestep", "episode", "reward"])

    def _on_step(self) -> bool:
        # `rewards` and `dones` are 1-D arrays of length n_envs (=1 here)
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self._ep_num += 1
            self._writer.writerow([self.num_timesteps, self._ep_num, round(self._ep_reward, 4)])
            self._file.flush()
            self._ep_reward = 0.0
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


# ── Callback: console progress ─────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    """Prints a one-line progress update every `log_interval` episodes."""

    def __init__(self, log_interval: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._ep_num      = 0
        self._ep_reward   = 0.0
        self._ep_rewards  = []

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self._ep_num    += 1
            self._ep_rewards.append(self._ep_reward)
            self._ep_reward  = 0.0

            if self._ep_num % self.log_interval == 0:
                recent = self._ep_rewards[-self.log_interval:]
                mean_r = np.mean(recent)
                sys.stdout.write(
                    f"\r  [RPPO] Step {self.num_timesteps:>8,d} | "
                    f"Ep {self._ep_num:>5,d} | "
                    f"Mean reward (last {self.log_interval}): {mean_r:>8.2f}"
                )
                sys.stdout.flush()
        return True


# ── Evaluation helper (inline, no CybORG needed) ──────────────────────────────

def evaluate_rppo(model: "RecurrentPPO", n_episodes: int = 10) -> float:
    """
    Run `model` for `n_episodes` and return the mean episode reward.

    Correctly manages the LSTM hidden state across steps within each episode
    and resets it at episode boundaries.
    """
    env = SchoolIRSEnv()
    total_rewards = []

    for _ in range(n_episodes):
        obs, _       = env.reset()
        # LSTM state: tuple of (h, c) each shaped (n_lstm_layers, 1, hidden_size)
        lstm_state   = None
        ep_start     = np.ones((1,), dtype=bool)   # True = start of episode
        ep_reward    = 0.0
        done         = False

        while not done:
            # Pass lstm_state so the policy can update its recurrent memory.
            # episode_starts signals the LSTM to zero its hidden state.
            action, lstm_state = model.predict(
                obs[np.newaxis, :],        # add batch dim expected by LSTM policy
                state=lstm_state,
                episode_start=ep_start,
                deterministic=True,
            )
            ep_start = np.zeros((1,), dtype=bool)  # only True on first step

            obs, reward, terminated, truncated, _ = env.step(action[0])
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)

    return float(np.mean(total_rewards))


# ── Eval Callback (wraps inline evaluator) ────────────────────────────────────

class RecurrentEvalCallback(BaseCallback):
    """
    Periodically evaluate the model and save the best checkpoint.

    This replaces stable-baselines3's built-in EvalCallback which does not
    handle the LSTM state correctly for RecurrentPPO.
    """

    def __init__(
        self,
        eval_freq:    int = 10_000,
        n_episodes:   int = 10,
        save_path:    str = "./logs/best_rppo/",
        model_name:   str = "best_model",
        verbose:      int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq  = eval_freq
        self.n_episodes = n_episodes
        self.save_path  = save_path
        self.model_name = model_name
        self._best_mean = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            mean_r = evaluate_rppo(self.model, self.n_episodes)
            if self.verbose:
                print(
                    f"\n  [Eval @ {self.num_timesteps:,}d steps] "
                    f"Mean reward = {mean_r:.2f}"
                    + (" ← NEW BEST" if mean_r > self._best_mean else "")
                )
            if mean_r > self._best_mean:
                self._best_mean = mean_r
                os.makedirs(self.save_path, exist_ok=True)
                self.model.save(os.path.join(self.save_path, self.model_name))
        return True


# ── Main Training Function ────────────────────────────────────────────────────

def train(quick: bool = False, resume: bool = False):
    total_steps = QUICK_TIMESTEPS if quick else TOTAL_TIMESTEPS

    print("\n" + "=" * 70)
    print(f"  IRS-RL  RecurrentPPO Training  -  {'QUICK' if quick else 'FULL'} MODE")
    print(f"  Timesteps    : {total_steps:,}")
    print(f"  LSTM hidden  : {RPPO_PARAMS['policy_kwargs']['lstm_hidden_size']}")
    print(f"  LSTM layers  : {RPPO_PARAMS['policy_kwargs']['n_lstm_layers']}")
    print("=" * 70)

    # -- Environment -----------------------------------------------------------
    print("\n[1/4] Building + checking environment ...")
    env = SchoolIRSEnv()
    check_env(env, warn=True)
    print("      [OK] Environment OK")

    # -- Callbacks -------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)

    reward_logger  = RewardLoggerCallback(csv_path=RPPO_REWARDS_CSV)
    progress_cb    = ProgressCallback(log_interval=50)
    eval_cb        = RecurrentEvalCallback(
        eval_freq=EVAL_FREQ,
        n_episodes=EVAL_EPISODES,
        save_path="./logs/best_rppo/",
    )
    callbacks = CallbackList([reward_logger, progress_cb, eval_cb])

    # -- Model -----------------------------------------------------------------
    print("\n[2/4] Initialising RecurrentPPO ...")

    if resume and os.path.exists(f"{MODEL_PATH_RPPO}.zip"):
        print(f"  Resuming from checkpoint: {MODEL_PATH_RPPO}.zip")
        model = RecurrentPPO.load(MODEL_PATH_RPPO, env=env)
    else:
        model = RecurrentPPO(
            policy          = RPPO_PARAMS["policy"],
            env             = env,
            verbose         = RPPO_PARAMS["verbose"],
            tensorboard_log = RPPO_PARAMS["tensorboard_log"],
            learning_rate   = RPPO_PARAMS["learning_rate"],
            n_steps         = RPPO_PARAMS["n_steps"],
            batch_size      = RPPO_PARAMS["batch_size"],
            n_epochs        = RPPO_PARAMS["n_epochs"],
            gamma           = RPPO_PARAMS["gamma"],
            gae_lambda      = RPPO_PARAMS["gae_lambda"],
            clip_range      = RPPO_PARAMS["clip_range"],
            ent_coef        = RPPO_PARAMS["ent_coef"],
            seed            = RPPO_PARAMS["seed"],
            policy_kwargs   = RPPO_PARAMS["policy_kwargs"],
        )

    print("  [OK] Model ready")
    print(f"  Policy network:\n{model.policy}")

    # -- Training --------------------------------------------------------------
    print(f"\n[3/4] Training for {total_steps:,} timesteps ...\n")
    try:
        model.learn(total_timesteps=total_steps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted - saving current weights ...")

    print()   # newline after progress bar

    # -- Save ------------------------------------------------------------------
    model.save(MODEL_PATH_RPPO)
    print(f"\n[4/4] Model saved -> {MODEL_PATH_RPPO}.zip")
    print(f"      Best checkpoint -> ./logs/best_rppo/best_model.zip")
    print(f"      Reward log      -> {RPPO_REWARDS_CSV}")
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    return model


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the IRS POMDP agent with RecurrentPPO (LSTM)."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke-test run ({QUICK_TIMESTEPS:,} steps instead of {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume training from '{MODEL_PATH_RPPO}.zip' if it exists",
    )
    args = parser.parse_args()
    train(quick=args.quick, resume=args.resume)
