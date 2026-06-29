import argparse
import os
import sys
import numpy as np

# stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("[WARNING] sb3-contrib is not installed. RecurrentPPO will not be available.")

from wrapper import SchoolIRSEnv
import config

# -- Progress Callback (ASCII safe) --------------------------------------------
class ProgressCallback(BaseCallback):
    def __init__(self, agent_name: str, log_interval: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name   = agent_name
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
                    f"\r  [{self.agent_name}] Step {self.num_timesteps:>8,d} | "
                    f"Ep {self._ep_num:>5,d} | "
                    f"Mean reward (last {self.log_interval}): {mean_r:>8.2f}"
                )
                sys.stdout.flush()
        return True

# -- Main Training Runner ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-agent Incident Response RL Trainer")
    parser.add_argument("--agent", type=str, choices=["rppo", "mlpppo"], required=True,
                        help="rppo for RecurrentPPO, mlpppo for standard PPO MlpPolicy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke-test of 20,000 steps")
    parser.add_argument("--ablation", type=str, choices=["none", "A", "B", "C", "D"], default="none",
                        help="Ablation variant: A (wasted=0), B (wasted=-10), C (inv_success=0), D (inv_cost=-2)")
    
    args = parser.parse_args()
    
    # 1. Apply Ablation Overrides to config.REWARD
    if args.ablation == "A":
        config.REWARD["wasted_restore"] = 0.0
        print("[Ablation A] Overriding wasted_restore = 0.0")
    elif args.ablation == "B":
        config.REWARD["wasted_restore"] = -10.0
        print("[Ablation B] Overriding wasted_restore = -10.0")
    elif args.ablation == "C":
        config.REWARD["investigate_success"] = 0.0
        print("[Ablation C] Overriding investigate_success = 0.0")
    elif args.ablation == "D":
        config.REWARD["investigate_cost"] = -2.0
        print("[Ablation D] Overriding investigate_cost = -2.0")
    else:
        print("[Normal Run] Using standard rebalanced rewards")
        
    print(f"Active Reward Config: {config.REWARD}")

    # 2. Setup output directories
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    timesteps = 20000 if args.quick else args.timesteps
    agent_label = f"{args.agent}_seed{args.seed}_ablation{args.ablation}"
    
    print("\n" + "=" * 70)
    print(f"  Training Agent : {args.agent.upper()}")
    print(f"  Seed           : {args.seed}")
    print(f"  Ablation       : {args.ablation}")
    print(f"  Timesteps      : {timesteps:,}")
    print("=" * 70)

    # 3. Environment construction and seed setting
    set_random_seed(args.seed)
    env = SchoolIRSEnv()
    env.reset(seed=args.seed)

    # 4. Callback list
    progress_cb = ProgressCallback(agent_name=agent_label, log_interval=50)
    callbacks = CallbackList([progress_cb])

    # 5. Initialize Model based on agent selection
    if args.agent == "rppo":
        policy_kwargs = {
            "lstm_hidden_size":   64,
            "n_lstm_layers":       1,
            "shared_lstm":       False,
            "enable_critic_lstm": True,
        }
        model = RecurrentPPO(
            policy          = "MlpLstmPolicy",
            env             = env,
            verbose         = 0,
            tensorboard_log = "./logs/",
            learning_rate   = 5e-4,
            n_steps         = 256,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.02,
            seed            = args.seed,
            policy_kwargs   = policy_kwargs,
        )
    elif args.agent == "mlpppo":
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            verbose         = 0,
            tensorboard_log = "./logs/",
            learning_rate   = 5e-4,
            n_steps         = 256,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.02,
            seed            = args.seed,
        )

    # 6. Train model
    print(f"Starting training run for {timesteps:,} steps ...")
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    print() # Newline after progress
    
    # 7. Save model
    save_path = f"results/models/{agent_label}"
    model.save(save_path)
    print(f"[OK] Training complete. Model saved to: {save_path}.zip\n")

if __name__ == "__main__":
    main()
