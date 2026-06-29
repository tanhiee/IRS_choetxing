import sys
import os
import random
import numpy as np

# Configure standard modules for pickle compatibility
try:
    import numpy._core.numeric as _numeric
    import numpy._core as _core
except ImportError:
    pass
else:
    sys.modules['numpy._core.numeric'] = _numeric
    sys.modules['numpy._core'] = _core

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from wrapper import SchoolIRSEnv
from baseline_agent import StaticPlaybookAgent
from config import (
    HOSTS,
    DOWNTIME_MAX,
    MAX_STEPS,
)

# ── Robust Subclass of Environment ───────────────────────────────────────────
class RobustnessSchoolIRSEnv(SchoolIRSEnv):
    def __init__(self, fp_rate=0.15, omission_rate=0.0, delay_steps=0):
        # We always evaluate with use_belief=False as the primary models use 18D obs
        super().__init__(use_belief=False)
        self.fp_rate = fp_rate
        self.omission_rate = omission_rate
        self.delay_steps = delay_steps
        self.alert_queue = []

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.alert_queue = []
        for _ in range(self.delay_steps):
            self.alert_queue.append(np.zeros(self.n_hosts, dtype=np.float32))
        return obs, info

    def _make_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_channels * self.n_hosts, dtype=np.float32)

        alert_signals = np.zeros(self.n_hosts, dtype=np.float32)
        investigated_prev = np.zeros(self.n_hosts, dtype=np.float32)
        downtime_frac = np.zeros(self.n_hosts, dtype=np.float32)

        for i, host in enumerate(HOSTS):
            remaining = self.downtime.get(host, 0)
            downtime_frac[i] = remaining / DOWNTIME_MAX
            investigated_prev[i] = 1.0 if host in self._prev_investigated else 0.0

            # Base SIEM alert logic
            if remaining > 0:
                alert_signals[i] = 0.0
            elif host in self.compromised:
                alert_signals[i] = 1.0
            elif host in self._prev_investigated:
                alert_signals[i] = 0.0
            else:
                alert_signals[i] = 1.0 if random.random() < self.fp_rate else 0.0

            # Inject Alert Omission
            if alert_signals[i] == 1.0 and random.random() < self.omission_rate:
                alert_signals[i] = 0.0

        # Inject Alert Delay
        if self.delay_steps > 0:
            self.alert_queue.append(alert_signals)
            alert_signals = self.alert_queue.pop(0)

        # Flat observation construction (18D: [alert, investigated_prev, downtime_frac] for each host)
        for i in range(self.n_hosts):
            base = i * self.obs_channels
            obs[base + 0] = alert_signals[i]
            obs[base + 1] = investigated_prev[i]
            obs[base + 2] = downtime_frac[i]

        return obs

# ── Observation Converter for MLP-PPO (18D -> 6D) ───────────────────────────
def convert_to_mlp_obs(obs):
    # obs is 18D. We want to extract a 6D vector where for each host i:
    # 2.0 if host is in downtime (downtime_frac > 0.0)
    # 1.0 if host alert is active (alert_signal > 0.5)
    # 0.0 otherwise
    mlp_obs = np.zeros(6, dtype=np.float32)
    for i in range(6):
        base = i * 3
        alert = obs[base + 0]
        downtime = obs[base + 2]
        if downtime > 0.0:
            mlp_obs[i] = 2.0
        elif alert > 0.5:
            mlp_obs[i] = 1.0
        else:
            mlp_obs[i] = 0.0
    return mlp_obs

# ── Evaluation Helper Functions ───────────────────────────────────────────────
def evaluate_model(model, model_type, env, n_episodes=50, max_steps=200):
    rewards = []
    ttrs = []
    clean_rates = []
    wasted_restores = []
    peak_comps = []

    # Dynamically inject max_steps if necessary
    env.step_limit = max_steps
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        lstm_state = None
        ep_start = np.ones((1,), dtype=bool)

        total_reward = 0.0
        step = 0
        wasted = 0
        peak = 0
        ttr = max_steps

        # We manually run step loop to allow dynamic episode length
        while step < max_steps:
            if model_type == 'lstm':
                action, lstm_state = model.predict(
                    obs[np.newaxis, :],
                    state=lstm_state,
                    episode_start=ep_start,
                    deterministic=True
                )
                ep_start = np.zeros((1,), dtype=bool)
                act = action[0]
            elif model_type == 'mlp':
                # Convert 18D observation to 6D
                mlp_obs = convert_to_mlp_obs(obs)
                action, _ = model.predict(mlp_obs, deterministic=True)
                # Map Discrete(7) action to MultiDiscrete([3]*6)
                act = np.zeros(6, dtype=np.int64)
                a_idx = int(action)
                if 1 <= a_idx <= 6:
                    act[a_idx - 1] = 2 # Restore host
            elif model_type == 'playbook':
                act, _ = model.predict(obs, episode_start=ep_start)
                ep_start = np.zeros((1,), dtype=bool)

            obs, reward, terminated, truncated, info = env.step(act)
            total_reward += reward
            wasted += info.get("wasted", 0)
            peak = max(peak, info.get("true_compromised", 0))
            
            # Record TTR when clean (TTR is steps to recover all active infections)
            if len(env.compromised) == 0 and ttr == max_steps:
                ttr = step + 1
            
            step += 1

        rewards.append(total_reward)
        ttrs.append(ttr)
        clean_rates.append(100.0 if len(env.compromised) == 0 else 0.0)
        wasted_restores.append(wasted)
        peak_comps.append(peak)

    return {
        "reward": np.mean(rewards),
        "ttr": np.mean(ttrs),
        "clean_rate": np.mean(clean_rates),
        "wasted": np.mean(wasted_restores),
        "peak": np.mean(peak_comps)
    }

# ── Main Experiment Execution ──────────────────────────────────────────────────
def run_experiments():
    # Load Models
    print("Loading models...")
    lstm_model = RecurrentPPO.load("rppo_irs_final.zip")
    mlp_model = PPO.load("ppo_school_irs_final.zip")
    playbook_agent = StaticPlaybookAgent()

    print("\n--- EXPERIMENT 1: SIEM NOISE (FALSE POSITIVE RATE) ---")
    fp_rates = [0.05, 0.15, 0.30, 0.45, 0.60]
    for fp in fp_rates:
        env = RobustnessSchoolIRSEnv(fp_rate=fp)
        lstm_res = evaluate_model(lstm_model, 'lstm', env)
        mlp_res = evaluate_model(mlp_model, 'mlp', env)
        play_res = evaluate_model(playbook_agent, 'playbook', env)
        print(f"FP Rate {fp:.2f}:")
        print(f"  [LSTM] Reward: {lstm_res['reward']:.2f}, TTR: {lstm_res['ttr']:.2f}, Clean%: {lstm_res['clean_rate']:.1f}%, Wasted: {lstm_res['wasted']:.2f}, Peak: {lstm_res['peak']:.2f}")
        print(f"  [MLP]  Reward: {mlp_res['reward']:.2f}, TTR: {mlp_res['ttr']:.2f}, Clean%: {mlp_res['clean_rate']:.1f}%, Wasted: {mlp_res['wasted']:.2f}, Peak: {mlp_res['peak']:.2f}")
        print(f"  [Play] Reward: {play_res['reward']:.2f}, TTR: {play_res['ttr']:.2f}, Clean%: {play_res['clean_rate']:.1f}%, Wasted: {play_res['wasted']:.2f}, Peak: {play_res['peak']:.2f}")

    print("\n--- EXPERIMENT 2: MISSING OBSERVATIONS (ALERT OMISSION) ---")
    omission_rates = [0.00, 0.10, 0.25, 0.40]
    for om in omission_rates:
        env = RobustnessSchoolIRSEnv(fp_rate=0.15, omission_rate=om)
        lstm_res = evaluate_model(lstm_model, 'lstm', env)
        mlp_res = evaluate_model(mlp_model, 'mlp', env)
        play_res = evaluate_model(playbook_agent, 'playbook', env)
        print(f"Omission Rate {om:.2f}:")
        print(f"  [LSTM] Reward: {lstm_res['reward']:.2f}, TTR: {lstm_res['ttr']:.2f}, Clean%: {lstm_res['clean_rate']:.1f}%, Wasted: {lstm_res['wasted']:.2f}, Peak: {lstm_res['peak']:.2f}")
        print(f"  [MLP]  Reward: {mlp_res['reward']:.2f}, TTR: {mlp_res['ttr']:.2f}, Clean%: {mlp_res['clean_rate']:.1f}%, Wasted: {mlp_res['wasted']:.2f}, Peak: {mlp_res['peak']:.2f}")
        print(f"  [Play] Reward: {play_res['reward']:.2f}, TTR: {play_res['ttr']:.2f}, Clean%: {play_res['clean_rate']:.1f}%, Wasted: {play_res['wasted']:.2f}, Peak: {play_res['peak']:.2f}")

    print("\n--- EXPERIMENT 3: ALERT DELAY ---")
    delays = [0, 1, 2, 3]
    for d in delays:
        env = RobustnessSchoolIRSEnv(fp_rate=0.15, delay_steps=d)
        lstm_res = evaluate_model(lstm_model, 'lstm', env)
        mlp_res = evaluate_model(mlp_model, 'mlp', env)
        play_res = evaluate_model(playbook_agent, 'playbook', env)
        print(f"Delay Steps {d}:")
        print(f"  [LSTM] Reward: {lstm_res['reward']:.2f}, TTR: {lstm_res['ttr']:.2f}, Clean%: {lstm_res['clean_rate']:.1f}%, Wasted: {lstm_res['wasted']:.2f}, Peak: {lstm_res['peak']:.2f}")
        print(f"  [MLP]  Reward: {mlp_res['reward']:.2f}, TTR: {mlp_res['ttr']:.2f}, Clean%: {mlp_res['clean_rate']:.1f}%, Wasted: {mlp_res['wasted']:.2f}, Peak: {mlp_res['peak']:.2f}")
        print(f"  [Play] Reward: {play_res['reward']:.2f}, TTR: {play_res['ttr']:.2f}, Clean%: {play_res['clean_rate']:.1f}%, Wasted: {play_res['wasted']:.2f}, Peak: {play_res['peak']:.2f}")

    print("\n--- EXPERIMENT 4: LONG-HORIZON EVALUATION ---")
    horizons = [200, 500, 1000]
    for h in horizons:
        env = RobustnessSchoolIRSEnv(fp_rate=0.15)
        lstm_res = evaluate_model(lstm_model, 'lstm', env, max_steps=h)
        mlp_res = evaluate_model(mlp_model, 'mlp', env, max_steps=h)
        play_res = evaluate_model(playbook_agent, 'playbook', env, max_steps=h)
        print(f"Horizon T={h}:")
        print(f"  [LSTM] Reward: {lstm_res['reward']:.2f}, TTR: {lstm_res['ttr']:.2f}, Clean%: {lstm_res['clean_rate']:.1f}%, Wasted: {lstm_res['wasted']:.2f}, Peak: {lstm_res['peak']:.2f}")
        print(f"  [MLP]  Reward: {mlp_res['reward']:.2f}, TTR: {mlp_res['ttr']:.2f}, Clean%: {mlp_res['clean_rate']:.1f}%, Wasted: {mlp_res['wasted']:.2f}, Peak: {mlp_res['peak']:.2f}")
        print(f"  [Play] Reward: {play_res['reward']:.2f}, TTR: {play_res['ttr']:.2f}, Clean%: {play_res['clean_rate']:.1f}%, Wasted: {play_res['wasted']:.2f}, Peak: {play_res['peak']:.2f}")

if __name__ == "__main__":
    run_experiments()
