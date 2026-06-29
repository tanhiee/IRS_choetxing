import numpy as np
from config import HOSTS, SPREAD_PROB, RED_REENTRY_PROB

class BayesianBeliefEstimator:
    """
    Bayesian Belief State Estimator for partially observable incident response.
    Maintains a posterior probability of compromise for each network host,
    updating beliefs via threat propagation (prior) and SIEM alert telemetry (posterior).
    """
    def __init__(self):
        self.n_hosts = len(HOSTS)
        self.hosts = HOSTS
        self.host_to_idx = {h: i for i, h in enumerate(self.hosts)}
        
        # Communication graph adjacency matrix (matching gnn_model topology)
        self.adj = np.array([
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0], # Admin_PC
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0], # Teacher_PC
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], # Student_PC1
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], # Student_PC2
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # File_Server
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Web_Server
        ], dtype=np.float32)
        
        self.reset()
        
    def reset(self):
        # Initial belief: Student_PC1 starts compromised (1.0), others clean (0.0)
        self.belief = np.zeros(self.n_hosts, dtype=np.float32)
        for i, h in enumerate(self.hosts):
            if h == "Student_PC1":
                self.belief[i] = 1.0
                
    def update(self, alert_signals, investigated_prev, downtime_frac):
        """
        Updates the belief state vector based on transition models and new observations.
        
        Parameters:
        -----------
        alert_signals : np.ndarray (N,)
            Binary/Noisy SIEM alert signal for each host (0.0 or 1.0).
        investigated_prev : np.ndarray (N,)
            Binary flag indicating if host was investigated in previous step (0.0 or 1.0).
        downtime_frac : np.ndarray (N,)
            Remaining downtime fraction for each host (0.0 means online).
            
        Returns:
        --------
        np.ndarray (N,)
            Updated compromise probability belief vector.
        """
        prior_belief = np.zeros(self.n_hosts, dtype=np.float32)
        
        # ── 1. PREDICTION STEP (Attacker spreading probability transition) ───
        for i in range(self.n_hosts):
            if downtime_frac[i] > 0:
                # Downtime hosts are offline and guaranteed clean
                prior_belief[i] = 0.0
                continue
                
            # Probability of NOT being infected by neighbors via lateral movement
            p_not_infected = 1.0
            for j in range(self.n_hosts):
                if i != j and self.adj[j, i] > 0:
                    # Neighbor j spreads with probability: belief[j] * SPREAD_PROB
                    p_spread_j_to_i = self.belief[j] * SPREAD_PROB
                    p_not_infected *= (1.0 - p_spread_j_to_i)
                    
            # Prior compromise probability = current belief + (1 - current) * spread_prob
            prior_belief[i] = self.belief[i] + (1.0 - self.belief[i]) * (1.0 - p_not_infected)

        # Attacker Re-entry: if the network is predicted clean, Red re-enters student subnets
        total_prior_comp = np.sum(prior_belief)
        if total_prior_comp < 0.05:
            for i, h in enumerate(self.hosts):
                if h in ["Student_PC1", "Student_PC2"]:
                    prior_belief[i] = prior_belief[i] + (1.0 - prior_belief[i]) * RED_REENTRY_PROB

        # ── 2. CORRECTION STEP (Filtering noise using observation model) ───
        new_belief = np.zeros(self.n_hosts, dtype=np.float32)
        for i in range(self.n_hosts):
            if downtime_frac[i] > 0:
                new_belief[i] = 0.0
                continue
                
            if investigated_prev[i] > 0:
                # Investigation gives noise-free ground truth for the alert signal
                new_belief[i] = float(alert_signals[i])
            else:
                # No investigation: apply Bayes' rule to noisy alert signal
                p_c = prior_belief[i]
                p_not_c = 1.0 - p_c
                
                if alert_signals[i] > 0:
                    # P(Alert=1 | Compromised) = 1.0, P(Alert=1 | Clean) = FPR = 0.15
                    fpr = 0.15
                    den = 1.0 * p_c + fpr * p_not_c
                    new_belief[i] = p_c / den if den > 0 else 0.0
                else:
                    # P(Alert=0 | Compromised) = 0.0, P(Alert=0 | Clean) = 0.85
                    # Since compromised hosts ALWAYS trigger alerts, a zero alert means 0.0 compromise belief
                    new_belief[i] = 0.0
                    
        self.belief = np.clip(new_belief, 0.0, 1.0)
        return self.belief
