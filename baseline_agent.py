"""
baseline_agent.py  (v2: MultiDiscrete action space)
====================================================
Static Playbook (rule-based) agent used as a comparison baseline against
RecurrentPPO.

v2 changes
----------
The environment now uses a MultiDiscrete([3]*N) action space:
    0 = Do Nothing
    1 = Investigate
    2 = Restore

The observation vector is now 3-channel: [alert, investigated, downtime_frac]
for each host (stride of 3).  The alert channel (offset 0) is the noisy SIEM
signal the playbook uses to decide.

Policy logic (conservative playbook):
    For each host, if its alert channel == 1 AND it is not in downtime:
      Step A: If not yet investigated → Investigate  (wait for confirmation)
      Step B: If already investigated last step → Restore  (confirmed threat)
    Otherwise: Do Nothing.

This mirrors a realistic Tier-1/Tier-2 SOC workflow:
  1. Alert fires → Tier-1 investigates.
  2. Confirmed compromise → Tier-2 triggers remediation.

The `predict` method accepts (and ignores) SB3-style keyword arguments
so it can be used as a drop-in replacement in evaluate_rppo.py.
"""

import numpy as np
from config import HOSTS

# Observation stride per host (must match wrapper.py OBS_CHANNELS)
OBS_CHANNELS = 3

# Per-host action codes
ACTION_DO_NOTHING  = 0
ACTION_INVESTIGATE = 1
ACTION_RESTORE     = 2


class StaticPlaybookAgent:
    """
    Deterministic two-step rule-based incident response agent.

    Mirrors the SB3 / RecurrentPPO `.predict()` API, returning
      (action_array, None)
    where action_array is a numpy int array of shape (N,).
    """

    def __init__(self):
        self.name = "StaticPlaybook"
        self.n_hosts = len(HOSTS)
        # Track which hosts were investigated in the previous step
        # so the agent knows when it's safe to escalate to Restore.
        self._investigated_prev: set = set()

    def reset(self):
        """Call at the start of each episode to clear internal memory."""
        self._investigated_prev = set()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """
        Parameters
        ----------
        obs : np.ndarray  shape=(3*N,) or (1, 3*N)
            Multi-channel observation from SchoolIRSEnv.
        state : ignored  (kept for SB3 API compatibility)
        episode_start : np.ndarray | None
            If provided and episode_start[0] is True, reset internal state.
        deterministic : bool  (ignored — always deterministic)

        Returns
        -------
        action : np.ndarray  shape=(N,)   int  values in {0,1,2}
        state  : None
        """
        obs = np.asarray(obs, dtype=np.float32).flatten()

        # Reset internal investigate memory at episode boundaries
        if episode_start is not None and len(episode_start) > 0:
            if bool(episode_start[0]):
                self._investigated_prev = set()

        actions = np.zeros(self.n_hosts, dtype=np.int64)
        newly_investigated = set()

        for i, host in enumerate(HOSTS):
            base = i * OBS_CHANNELS

            alert_signal      = obs[base + 0]   # noisy SIEM alarm
            investigated_prev = obs[base + 1]    # was investigated last step?
            downtime_frac     = obs[base + 2]    # in cooldown?

            # Skip hosts currently in downtime — can't act on them
            if downtime_frac > 0:
                actions[i] = ACTION_DO_NOTHING
                continue

            if alert_signal == 1.0:
                if investigated_prev == 1.0:
                    # Confirmed threat (investigated last step + still alerting)
                    # -> Escalate to Restore
                    actions[i] = ACTION_RESTORE
                else:
                    # First alert, could be a false positive -> Investigate first
                    actions[i] = ACTION_INVESTIGATE
                    newly_investigated.add(host)
            else:
                actions[i] = ACTION_DO_NOTHING

        self._investigated_prev = newly_investigated
        return actions, None


# ── Legacy single-action API shim ─────────────────────────────────────────────
# evaluate.py (v1) called agent.predict(obs) and expected a scalar int.
# This is kept for backward compatibility with the old evaluate.py.

class StaticPlaybookAgentLegacy:
    """
    Legacy single-action version (for use with old evaluate.py / train.py).
    Maps the first alerting host to a scalar Restore action.
    """

    def __init__(self):
        self.name = "StaticPlaybook_Legacy"

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        # Old obs space was 1-D (1 value per host): 0=clean, 1=fp_alert, 2=compromised
        for i, val in enumerate(obs):
            if val >= 1.0:
                return i + 1, None
        return 0, None
