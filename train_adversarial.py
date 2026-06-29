import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("sb3-contrib is required for RecurrentPPO training.")

from adversarial_env import AdversarialSchoolEnv
from gnn_model import GNNFeatureExtractor
import config

class IterationCallback(BaseCallback):
    """
    Callback that logs training statistics for co-evolution monitoring.
    """
    def __init__(self, agent_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self._ep_rewards = []
        self._curr_reward = 0.0

    def _on_step(self) -> bool:
        self._curr_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self._ep_rewards.append(self._curr_reward)
            self._curr_reward = 0.0
        return True

    def get_mean_reward(self):
        if not self._ep_rewards:
            return 0.0
        return np.mean(self._ep_rewards[-50:]) # last 50 episodes

def train_adversarial():
    parser = argparse.ArgumentParser(description="Co-evolutionary Self-Play Training")
    parser.add_argument("--iterations", type=int, default=5, help="Number of co-evolution iterations")
    parser.add_argument("--steps", type=int, default=50000, help="Timesteps per agent per iteration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test run")
    args = parser.parse_args()

    # Apply quick settings if requested
    iterations = 3 if args.quick else args.iterations
    steps_per_iter = 10000 if args.quick else args.steps

    print("\n" + "=" * 70)
    print("  CO-EVOLUTION ADVERSARIAL SELF-PLAY TRAINING")
    print(f"  Iterations: {iterations}")
    print(f"  Steps per iteration: {steps_per_iter:,}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    set_random_seed(args.seed)

    # 1. Instantiate environments
    env_def = AdversarialSchoolEnv(mode="defender")
    env_att = AdversarialSchoolEnv(mode="attacker")
    
    env_def.reset(seed=args.seed)
    env_att.reset(seed=args.seed)

    # 2. Configure defender policy kwargs with GNN Feature Extractor
    policy_kwargs = {
        "features_extractor_class": GNNFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        "lstm_hidden_size": 64,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }

    # 3. Instantiate Defender (RecurrentPPO)
    print("\n[Init] Initializing GNN-Belief Recurrent PPO Defender...")
    defender = RecurrentPPO(
        "MlpLstmPolicy",
        env_def,
        verbose=0,
        learning_rate=5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/adversarial/"
    )

    # 4. Instantiate Attacker (PPO)
    print("[Init] Initializing RL Attacker...")
    attacker = PPO(
        "MlpPolicy",
        env_att,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
        tensorboard_log="./logs/adversarial/"
    )

    # 5. Link opponent policies in environments
    env_def.opponent_policy = attacker
    env_att.opponent_policy = defender

    os.makedirs("results/models", exist_ok=True)

    # 6. Training Loop
    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it}/{iterations} ---")
        
        # Train Defender
        print(f"  Training Defender against current Attacker for {steps_per_iter:,} steps...")
        def_callback = IterationCallback("Defender")
        defender.learn(total_timesteps=steps_per_iter, callback=def_callback, reset_num_timesteps=False)
        mean_def_r = def_callback.get_mean_reward()
        print(f"  [Defender] Mean Reward (last 50 eps): {mean_def_r:.2f}")

        # Train Attacker
        print(f"  Training Attacker against current Defender for {steps_per_iter:,} steps...")
        att_callback = IterationCallback("Attacker")
        attacker.learn(total_timesteps=steps_per_iter, callback=att_callback, reset_num_timesteps=False)
        mean_att_r = att_callback.get_mean_reward()
        print(f"  [Attacker] Mean Reward (last 50 eps): {mean_att_r:.2f}")

        # Save checkpoint after each iteration
        iter_def_path = f"results/models/rppo_defender_adv_iter{it}"
        iter_att_path = f"results/models/ppo_attacker_adv_iter{it}"
        defender.save(iter_def_path)
        attacker.save(iter_att_path)
        print(f"  [OK] Checkpoint saved for iteration {it}")

    # 7. Save final models
    final_def_path = "results/models/rppo_defender_adv"
    final_att_path = "results/models/ppo_attacker_adv"
    defender.save(final_def_path)
    attacker.save(final_att_path)
    print("\n" + "=" * 70)
    print("  CO-EVOLUTIONARY ADVERSARIAL TRAINING COMPLETE")
    print(f"  Defender Saved: {final_def_path}.zip")
    print(f"  Attacker Saved: {final_att_path}.zip")
    print("=" * 70)

if __name__ == "__main__":
    train_adversarial()
