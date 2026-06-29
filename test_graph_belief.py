import torch
import numpy as np
import gymnasium as gym
from gnn_model import GNNFeatureExtractor
from belief_estimator import BayesianBeliefEstimator
from adversarial_env import AdversarialSchoolEnv
from config import HOSTS

def test_belief_estimator():
    print("\n--- Testing BayesianBeliefEstimator ---")
    estimator = BayesianBeliefEstimator()
    print("Initial belief:", estimator.belief)
    assert estimator.belief[2] == 1.0, "Student_PC1 should start compromised (index 2)"
    
    # Run prediction + correction with no alerts, no investigations, no downtime
    alerts = np.zeros(6, dtype=np.float32)
    investigated = np.zeros(6, dtype=np.float32)
    downtime = np.zeros(6, dtype=np.float32)
    
    new_beliefs = estimator.update(alerts, investigated, downtime)
    print("After zero-alert step:", new_beliefs)
    assert np.all(new_beliefs == 0.0), "Zero alerts should yield zero compromise belief due to 100% TPR"
    
    # Now set alert on Student_PC2 (index 3) to 1.0
    alerts[3] = 1.0
    # Update - should trigger re-entry because net is clean, then Bayes correction
    new_beliefs = estimator.update(alerts, investigated, downtime)
    print("After alert on Student_PC2 (with clean net prior):", new_beliefs)
    assert new_beliefs[3] > 0.0, "Student_PC2 belief should increase after alert"
    
    # Test investigation verification
    # Investigated last step, alert is 1.0 -> should be 1.0 compromise probability
    investigated[3] = 1.0
    alerts[3] = 1.0
    new_beliefs = estimator.update(alerts, investigated, downtime)
    print("After investigation confirmation (alert=1):", new_beliefs)
    assert new_beliefs[3] == 1.0, "Investigated + Alert=1 should yield 1.0 belief"
    
    # Investigated last step, alert is 0.0 -> should be 0.0 compromise probability
    alerts[3] = 0.0
    new_beliefs = estimator.update(alerts, investigated, downtime)
    print("After investigation clearance (alert=0):", new_beliefs)
    assert new_beliefs[3] == 0.0, "Investigated + Alert=0 should yield 0.0 belief"
    
    print("[SUCCESS] Belief Estimator test passed!")

def test_gnn_extractor():
    print("\n--- Testing GNNFeatureExtractor ---")
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(24,), dtype=np.float32)
    extractor = GNNFeatureExtractor(obs_space, features_dim=64)
    
    # Mock observation batch: size 2
    mock_obs = torch.rand(2, 24)
    out = extractor(mock_obs)
    print("Input shape:", mock_obs.shape)
    print("Output shape:", out.shape)
    
    assert out.shape == (2, 64), f"Output shape should be (2, 64), got {out.shape}"
    print("[SUCCESS] GNN Feature Extractor test passed!")

def test_adversarial_env():
    print("\n--- Testing AdversarialSchoolEnv ---")
    
    # Test defender mode
    env_def = AdversarialSchoolEnv(mode="defender")
    obs, info = env_def.reset(seed=42)
    print("Defender Mode - Action Space:", env_def.action_space)
    print("Defender Mode - Obs Space:", env_def.observation_space)
    print("Defender Mode - Reset Obs Shape:", obs.shape)
    assert obs.shape == (24,), f"Defender obs shape should be (24,), got {obs.shape}"
    
    # Step in defender mode
    action_def = np.zeros(6, dtype=np.int64) # Do nothing on all hosts
    next_obs, reward, term, trunc, info = env_def.step(action_def)
    print("Defender Mode - Step Reward:", reward)
    
    # Test attacker mode
    env_att = AdversarialSchoolEnv(mode="attacker")
    obs_att, info_att = env_att.reset(seed=42)
    print("Attacker Mode - Action Space:", env_att.action_space)
    print("Attacker Mode - Obs Space:", env_att.observation_space)
    print("Attacker Mode - Reset Obs Shape:", obs_att.shape)
    assert obs_att.shape == (12,), f"Attacker obs shape should be (12,), got {obs_att.shape}"
    
    # Step in attacker mode (attack Student_PC1 which is action 3)
    action_att = 3
    next_obs_att, reward_att, term_att, trunc_att, info_att = env_att.step(action_att)
    print("Attacker Mode - Step Reward:", reward_att)
    
    print("[SUCCESS] Adversarial Environment test passed!")

if __name__ == "__main__":
    test_belief_estimator()
    test_gnn_extractor()
    test_adversarial_env()
    print("\n[ALL PASSED] All tests completed successfully!")
