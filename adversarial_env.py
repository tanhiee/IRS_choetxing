import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from wrapper import SchoolIRSEnv
from config import (
    HOSTS,
    DOWNTIME_MAX,
    SPREAD_PROB,
    RED_REENTRY_PROB,
    MAX_STEPS,
)
from baseline_agent import StaticPlaybookAgent

class AdversarialSchoolEnv(SchoolIRSEnv):
    """
    Adversarial School Incident Response environment supporting two-agent training.
    
    Modes:
      - "defender": Standard action/observation space. Attacker actions are selected
                    by querying self.opponent_policy (an RL attacker) or using the rule-based baseline.
      - "attacker": Action/observation spaces are flipped for the attacker. Defender actions
                    are selected by querying self.opponent_policy (an RL defender) or the Static Playbook.
    """
    def __init__(self, mode: str = "defender", render_mode: Optional[str] = None):
        super().__init__(render_mode=render_mode)
        assert mode in ["defender", "attacker"], "mode must be 'defender' or 'attacker'"
        self.mode = mode
        self.opponent_policy = None
        
        # Attacker action space: Choose 1 of 7 options:
        # 0: Do Nothing
        # 1-6: Target Host i for compromise/lateral movement
        self.attacker_action_space = spaces.Discrete(7)
        
        # Attacker observation space: 
        # Node status: [is_compromised (6), is_downtime (6)]
        self.attacker_observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(12,),
            dtype=np.float32
        )
        
        # Adjust spaces based on active mode
        if self.mode == "attacker":
            self.action_space = self.attacker_action_space
            self.observation_space = self.attacker_observation_space
            
        # Connectivity graph adjacency list (same as gnn_model topology)
        self.network_adj = {
            "Admin_PC": ["Teacher_PC", "File_Server", "Web_Server"],
            "Teacher_PC": ["Admin_PC", "File_Server", "Web_Server"],
            "Student_PC1": ["Student_PC2", "File_Server", "Web_Server"],
            "Student_PC2": ["Student_PC1", "File_Server", "Web_Server"],
            "File_Server": ["Admin_PC", "Teacher_PC", "Student_PC1", "Student_PC2", "Web_Server"],
            "Web_Server": ["Admin_PC", "Teacher_PC", "Student_PC1", "Student_PC2", "File_Server"]
        }
        
        # Static playbook defender fallback when training the attacker
        self.static_playbook = StaticPlaybookAgent()
        self.defender_lstm_state = None
        self.defender_ep_start = np.ones((1,), dtype=bool)

    def _make_attacker_obs(self) -> np.ndarray:
        obs = np.zeros(12, dtype=np.float32)
        for i, host in enumerate(HOSTS):
            obs[i] = 1.0 if host in self.compromised else 0.0
            obs[6 + i] = self.downtime.get(host, 0) / DOWNTIME_MAX
        return obs

    def _apply_rl_attacker_action(self, action: int):
        """
        Executes a targeted exploit attempt by the RL attacker.
        
        Action mapping:
          0: Do Nothing
          1-6: Attack host 1-6
        """
        # Age existing compromises
        for host in list(self.compromised):
            self.compromise_age[host] = self.compromise_age.get(host, 0) + 1
            
        if action == 0:
            return # Attacker chose not to attack this step
            
        target_idx = action - 1
        target_host = HOSTS[target_idx]
        
        # Cannot compromise a host in downtime
        if target_host in self.downtime:
            return
            
        # Already compromised
        if target_host in self.compromised:
            return
            
        # Case 1: Network is fully clean -> re-entry attempt
        if not self.compromised:
            if target_host in ["Student_PC1", "Student_PC2"]:
                if random.random() < RED_REENTRY_PROB:
                    self.compromised.add(target_host)
                    self.compromise_age[target_host] = 1
            return
            
        # Case 2: Lateral spread from a compromised neighbor
        neighbors = self.network_adj[target_host]
        has_compromised_neighbor = any(n in self.compromised for n in neighbors)
        if has_compromised_neighbor:
            if random.random() < SPREAD_PROB:
                self.compromised.add(target_host)
                self.compromise_age[target_host] = 1

    def step(self, action):
        self.current_step += 1
        
        if self.mode == "defender":
            defender_action = np.asarray(action, dtype=np.int64)
            
            # Get Attacker action
            if self.opponent_policy is not None:
                attacker_obs = self._make_attacker_obs()
                # Query RL attacker policy
                attacker_action, _ = self.opponent_policy.predict(attacker_obs, deterministic=True)
                attacker_action = int(attacker_action)
                
                # Apply defender actions
                restored, wasted, new_invest, true_positives, invalid = self._apply_actions(defender_action)
                self._prev_investigated = new_invest
                self._tick_downtime()
                
                # Compute rewards before Red action
                reward, reward_decomp = self._compute_reward(defender_action, restored, wasted, true_positives, invalid)
                
                # Apply RL attacker action
                self._apply_rl_attacker_action(attacker_action)
            else:
                # Fallback to standard rule-based lateral spread
                restored, wasted, new_invest, true_positives, invalid = self._apply_actions(defender_action)
                self._prev_investigated = new_invest
                self._tick_downtime()
                reward, reward_decomp = self._compute_reward(defender_action, restored, wasted, true_positives, invalid)
                self._red_spread()
                
            obs = self._make_obs()
            truncated = self.current_step >= MAX_STEPS
            
            info = {
                "step":               self.current_step,
                "true_compromised":   len(self.compromised),
                "true_clean":         self.n_hosts - len(self.compromised) - len(self.downtime),
                "hosts_in_downtime":  len(self.downtime),
                "restored":           len(restored),
                "wasted":             len(wasted),
                "investigated":       len(new_invest),
                "true_positives":     len(true_positives),
                "invalid_actions":    len(invalid),
                "max_compromise_age": max(self.compromise_age.values(), default=0),
            }
            info.update(reward_decomp)
            return obs, reward, False, truncated, info
            
        elif self.mode == "attacker":
            attacker_action = int(action)
            
            # Get Defender action
            def_obs = self._make_obs()
            if self.opponent_policy is not None:
                # Query Recurrent PPO defender policy
                # Maintain LSTM states
                defender_action, self.defender_lstm_state = self.opponent_policy.predict(
                    def_obs[np.newaxis, :],
                    state=self.defender_lstm_state,
                    episode_start=self.defender_ep_start,
                    deterministic=True,
                )
                defender_action = defender_action[0]
            else:
                # Fallback to Static Playbook agent
                defender_action, _ = self.static_playbook.predict(
                    def_obs,
                    episode_start=self.defender_ep_start,
                    deterministic=True,
                )
                
            self.defender_ep_start = np.zeros((1,), dtype=bool)
            
            # Apply defender action
            restored, wasted, new_invest, true_positives, invalid = self._apply_actions(defender_action)
            self._prev_investigated = new_invest
            self._tick_downtime()
            
            # Compute defender rewards (before Red spreads)
            defender_reward, reward_decomp = self._compute_reward(defender_action, restored, wasted, true_positives, invalid)
            
            # Apply attacker actions
            self._apply_rl_attacker_action(attacker_action)
            
            # Zero-sum reward for the attacker
            attacker_reward = -defender_reward
            
            obs = self._make_attacker_obs()
            truncated = self.current_step >= MAX_STEPS
            
            info = {
                "step":               self.current_step,
                "true_compromised":   len(self.compromised),
                "true_clean":         self.n_hosts - len(self.compromised) - len(self.downtime),
                "hosts_in_downtime":  len(self.downtime),
                "restored":           len(restored),
                "wasted":             len(wasted),
                "investigated":       len(new_invest),
                "true_positives":     len(true_positives),
                "invalid_actions":    len(invalid),
                "max_compromise_age": max(self.compromise_age.values(), default=0),
            }
            info.update(reward_decomp)
            return obs, attacker_reward, False, truncated, info

    def reset(self, seed=None, options=None):
        # Reset parent env parameters
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset defender-side variables for attacker mode
        self.defender_lstm_state = None
        self.defender_ep_start = np.ones((1,), dtype=bool)
        if self.mode == "attacker":
            self.static_playbook.reset()
            return self._make_attacker_obs(), {}
        return obs, info
