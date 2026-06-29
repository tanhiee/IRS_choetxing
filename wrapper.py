"""
wrapper.py  —  SchoolIRSEnv  (v2: POMDP + RecurrentPPO-ready)
=================================================================

Architecture Overview
---------------------
This environment implements a Partially Observable Markov Decision Process
(POMDP) for autonomous incident response in a school network.  Standard MLP
policies fail here because a raised SIEM alert may be a real threat OR a
false positive (15 % of the time), making a single timestep's observation
insufficient for correct action.

How the LSTM + Investigate interaction works
--------------------------------------------
RecurrentPPO maintains a hidden state (h_t, c_t) that accumulates context
across the full episode.  The "Investigate" action is the bridge:

  Step t  : Agent sees alert=1 on Host X (ambiguous — FP or real?).
             Agent plays action=1 (Investigate) on Host X.
             The environment marks Host X in `self.investigated`.
             Reward = -2  (small analyst cost).

  Step t+1: Because Host X was investigated, `_make_obs()` bypasses the
             FP noise and returns the TRUE alert state (1=real, 0=clean).
             Now the LSTM's memory of "I investigated X last step" plus
             "this step X still shows alert=1" provides unambiguous
             evidence of a real compromise → the agent should Restore.

Without the LSTM, the agent cannot associate the investigation action at t
with the clean observation at t+1.  The LSTM hidden state carries exactly
this short-term memory, allowing it to learn the 2-step pattern:
    Investigate → confirm → Restore.

Action Space
------------
MultiDiscrete([3, 3, ..., 3])  — one 3-way choice per host (N hosts).
    0 = Do Nothing   — no action on this host
    1 = Investigate  — dispatch a SOC analyst; costs -2 reward; reveals
                        true host state in the NEXT timestep only.
    2 = Restore      — wipe and reimage; if compromised → success (+8);
                        if clean → wasted (-10) + host enters downtime.

Observation Space
-----------------
Box(low=0, high=1, shape=(3*N,), dtype=float32)

For each host i (stride of 3):
    obs[3*i + 0]  alert_signal      noisy SIEM alarm  (0 or 1)
    obs[3*i + 1]  investigated      was host investigated *last* step? (0/1)
    obs[3*i + 2]  downtime_frac     remaining_ticks / DOWNTIME_MAX  ∈ [0,1]

Reward Function
---------------
  step_penalty                  : -0.5   (constant overhead)
  per clean+online host         : +weight * clean_host
  per compromised host          : -base * exp(scale * age_in_steps)  (exponential!)
  per host in downtime          : downtime_penalty   (-1.0 per step)
  Investigate action per host   : investigate_cost   (-2.0)
  Successful Restore            : +restore_bonus     (+8.0)
  Wasted Restore (clean host)   : wasted_restore     (-10.0)
  All-clean + Do Nothing        : +do_nothing_bonus  (+1.0)
"""

import random
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    HOSTS,
    HOST_WEIGHTS,
    FALSE_POSITIVE_RATE,
    MAX_STEPS,
    REWARD,
    DOWNTIME_MIN,
    DOWNTIME_MAX,
    SPREAD_PROB,
    RED_REENTRY_PROB,
    RED_START_HOST,
)
from belief_estimator import BayesianBeliefEstimator

# Number of channels per host in the observation vector
OBS_CHANNELS = 4   # [alert_signal, investigated_last_step, downtime_frac, belief_state]

# Per-host actions
ACTION_DO_NOTHING  = 0
ACTION_INVESTIGATE = 1
ACTION_RESTORE     = 2


class SchoolIRSEnv(gym.Env):
    """
    Standalone gymnasium environment for the School Incident Response System.

    Designed for RecurrentPPO (sb3-contrib):
      — Partial observability via false-positive SIEM alerts.
      — Investigate action that filters FP noise one step later.
      — Exponentially increasing penalty for unaddressed compromises.
      — Downtime / cooldown mechanic after Restore (3-5 steps).
      — MultiDiscrete action space: one [0,1,2] decision per host.

    No CybORG dependency — fully self-contained.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode: Optional[str] = None, use_belief: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.n_hosts     = len(HOSTS)
        self.use_belief  = use_belief
        self.obs_channels = 4 if use_belief else 3

        # ── Action Space ───────────────────────────────────────────────────────
        # One 3-way choice per host: [Do Nothing, Investigate, Restore]
        self.action_space = spaces.MultiDiscrete([3] * self.n_hosts)

        # ── Observation Space ──────────────────────────────────────────────────
        # All values normalised to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_channels * self.n_hosts,),
            dtype=np.float32,
        )

        # Internal state — initialised properly in reset()
        self.current_step:    int       = 0
        self.compromised:     set       = set()
        self.investigated:    set       = set()   # investigated THIS step
        self.downtime:        dict      = {}       # host → remaining ticks
        self.compromise_age:  dict      = {}       # host → steps continuously compromised
        self._prev_investigated: set   = set()    # investigated LAST step (for obs)
        self.belief_estimator = BayesianBeliefEstimator()

    # ── Observation Builder ────────────────────────────────────────────────────

    def _make_obs(self) -> np.ndarray:
        """
        Build the observation vector.

        Channel breakdown per host i:
          [0] alert_signal      : noisy SIEM alarm
          [1] investigated_prev : was this host investigated LAST timestep?
          [2] downtime_frac     : normalised remaining downtime ticks
          [3] belief_state      : compromise probability estimate (if use_belief is True)
        """
        obs = np.zeros(self.obs_channels * self.n_hosts, dtype=np.float32)

        alert_signals = np.zeros(self.n_hosts, dtype=np.float32)
        investigated_prev = np.zeros(self.n_hosts, dtype=np.float32)
        downtime_frac = np.zeros(self.n_hosts, dtype=np.float32)

        for i, host in enumerate(HOSTS):
            remaining = self.downtime.get(host, 0)
            downtime_frac[i] = remaining / DOWNTIME_MAX
            investigated_prev[i] = 1.0 if host in self._prev_investigated else 0.0

            # alert signal (noisy SIEM reading)
            if remaining > 0:
                alert_signals[i] = 0.0
            elif host in self.compromised:
                alert_signals[i] = 1.0
            elif host in self._prev_investigated:
                alert_signals[i] = 0.0
            else:
                alert_signals[i] = 1.0 if random.random() < FALSE_POSITIVE_RATE else 0.0

        if self.use_belief:
            # Update and get beliefs
            beliefs = self.belief_estimator.update(alert_signals, investigated_prev, downtime_frac)

        # Build flat observation vector
        for i in range(self.n_hosts):
            base = i * self.obs_channels
            obs[base + 0] = alert_signals[i]
            obs[base + 1] = investigated_prev[i]
            obs[base + 2] = downtime_frac[i]
            if self.use_belief:
                obs[base + 3] = beliefs[i]

        return obs

    # ── Reward Calculator ──────────────────────────────────────────────────────

    def _compute_reward(
        self,
        actions:        np.ndarray,
        restored:       list[str],   # hosts successfully cleared
        wasted:         list[str],   # hosts targeted by Restore but already clean
        true_positives: list[str],   # hosts investigated that were truly compromised
        invalid:        list[str],   # actions on downtime hosts
    ) -> tuple[float, dict]:
        """
        Compute the shaped reward and return its component-wise decomposition.
        """
        r_step = REWARD["step_penalty"]
        r_downtime = 0.0
        r_compromise = 0.0
        r_clean = 0.0

        base  = REWARD["compromise_base"]
        scale = REWARD["compromise_exp_scale"]

        for host in HOSTS:
            weight = HOST_WEIGHTS[host]

            if host in self.downtime:
                r_downtime += REWARD["downtime_penalty"]

            elif host in self.compromised:
                # Cap the compromise age at 20 to prevent exponential reward penalty explosion
                age = min(self.compromise_age.get(host, 1), 20)
                r_compromise -= base * float(np.exp(scale * age)) * weight

            else:
                r_clean += REWARD["clean_host"] * weight

        # Action outcome rewards
        r_restore_bonus = REWARD["restore_bonus"]       * len(restored)
        r_wasted_restore = REWARD["wasted_restore"]      * len(wasted)
        r_investigate_success = REWARD["investigate_success"] * len(true_positives)
        r_invalid_action = REWARD["invalid_action"]      * len(invalid)

        # Investigate cost
        n_investigated = int(np.sum(actions == ACTION_INVESTIGATE))
        r_investigate_cost = REWARD["investigate_cost"] * n_investigated

        # Do nothing bonus
        r_do_nothing_bonus = 0.0
        all_clean  = len(self.compromised) == 0
        all_online = len(self.downtime) == 0
        if all_clean and all_online and np.all(actions == ACTION_DO_NOTHING):
            r_do_nothing_bonus = REWARD["do_nothing_bonus"]

        total = r_step + r_downtime + r_compromise + r_clean + r_restore_bonus + r_wasted_restore + r_investigate_success + r_invalid_action + r_investigate_cost + r_do_nothing_bonus

        decomp = {
            "reward_step_penalty":        r_step,
            "reward_clean_host":          r_clean,
            "reward_compromise_penalty":  r_compromise,
            "reward_downtime_penalty":    r_downtime,
            "reward_restore_bonus":       r_restore_bonus,
            "reward_wasted_restore":      r_wasted_restore,
            "reward_investigate_success": r_investigate_success,
            "reward_invalid_action":      r_invalid_action,
            "reward_investigate_cost":    r_investigate_cost,
            "reward_do_nothing_bonus":    r_do_nothing_bonus,
        }

        return float(total), decomp

    # ── Action Processor ───────────────────────────────────────────────────────

    def _apply_actions(
        self, actions: np.ndarray
    ) -> tuple[list[str], list[str], set, list[str], list[str]]:
        """
        Process the per-host action vector.

        Returns
        -------
        restored       : hosts successfully wiped (compromised → now in downtime)
        wasted         : hosts Restore-targeted but already clean
        new_invest     : set of hosts investigated this step (for next obs)
        true_positives : hosts investigated that were TRULY compromised
                         → triggers investigate_success reward
        invalid        : hosts where a non-zero action was attempted on a
                         DOWNTIME host → triggers invalid_action penalty
        """
        restored       = []
        wasted         = []
        new_invest     = set()
        true_positives = []   # investigated + genuinely compromised
        invalid        = []   # acted on a downtime host

        for i, host in enumerate(HOSTS):
            act = int(actions[i])

            if act == ACTION_DO_NOTHING:
                pass  # no-op

            elif act == ACTION_INVESTIGATE:
                if host in self.downtime:
                    # Invalid: can't investigate a host that's offline/recovering
                    invalid.append(host)
                else:
                    new_invest.add(host)
                    # investigate_success: give immediate bonus if host is truly compromised
                    # The LSTM learns: "Investigating a compromised host is worth it"
                    if host in self.compromised:
                        true_positives.append(host)

            elif act == ACTION_RESTORE:
                if host in self.downtime:
                    # Invalid: can't restore a host already in cooldown
                    invalid.append(host)

                elif host in self.compromised:
                    # ── Successful Restore ───────────────────────────
                    self.compromised.discard(host)
                    self.compromise_age.pop(host, None)
                    cooldown = random.randint(DOWNTIME_MIN, DOWNTIME_MAX)
                    self.downtime[host] = cooldown
                    restored.append(host)

                else:
                    # ── Wasted Restore ──────────────────────────────
                    cooldown = random.randint(DOWNTIME_MIN, DOWNTIME_MAX)
                    self.downtime[host] = cooldown
                    wasted.append(host)

        return restored, wasted, new_invest, true_positives, invalid

    # ── Red Attacker ───────────────────────────────────────────────────────────

    def _red_spread(self):
        """
        Stochastic kill-chain attacker model.

        Red spreads from compromised student endpoints to servers first,
        modelling realistic lateral movement patterns.
        If the network is clean, there is a small re-entry probability.
        """
        if not self.compromised:
            if random.random() < RED_REENTRY_PROB:
                self.compromised.add(random.choice(["Student_PC1", "Student_PC2"]))
            return

        # Online-only hosts can be newly compromised
        reachable = [
            h for h in HOSTS
            if h not in self.compromised and h not in self.downtime
        ]
        if not reachable:
            return

        if random.random() < SPREAD_PROB:
            # Prefer servers if Red has a foothold on student endpoints
            has_student = any("Student" in h for h in self.compromised)
            server_targets = [h for h in reachable if "Server" in h]

            if has_student and server_targets:
                self.compromised.add(random.choice(server_targets))
            else:
                self.compromised.add(random.choice(reachable))

        # Tick up compression age for each existing compromised host
        for host in list(self.compromised):
            self.compromise_age[host] = self.compromise_age.get(host, 0) + 1

    # ── Downtime Tick ─────────────────────────────────────────────────────────

    def _tick_downtime(self):
        """Decrement all cooldown counters; remove hosts that have recovered."""
        expired = [h for h, t in self.downtime.items() if t <= 1]
        for h in expired:
            del self.downtime[h]
        for h in self.downtime:
            self.downtime[h] -= 1

    # ── Gymnasium Interface ────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        self.current_step += 1

        # Ensure action is a numpy array (gym vectorisation sometimes passes lists)
        actions = np.asarray(action, dtype=np.int64)

        # 1. Blue acts - process Investigate / Restore decisions
        restored, wasted, new_invest, true_positives, invalid = self._apply_actions(actions)

        # 2. Update investigation memory for next obs generation
        self._prev_investigated = new_invest

        # 3. Tick downtime counters
        self._tick_downtime()

        # 4. Compute reward AFTER Blue's actions but BEFORE Red spreads
        reward, reward_decomp = self._compute_reward(actions, restored, wasted, true_positives, invalid)

        # 5. Red spreads and ages existing compromises for the NEXT step
        self._red_spread()

        # 6. Generate noisy observation reflecting the new state
        obs = self._make_obs()

        truncated = self.current_step >= MAX_STEPS

        # -- Info dict --------------------------------------------------------
        info = {
            "step":               self.current_step,
            "true_compromised":   len(self.compromised),
            "true_clean":         self.n_hosts - len(self.compromised) - len(self.downtime),
            "hosts_in_downtime":  len(self.downtime),
            "restored":           len(restored),
            "wasted":             len(wasted),
            "investigated":       len(new_invest),
            "true_positives":     len(true_positives),   # NEW
            "invalid_actions":    len(invalid),           # NEW
            "max_compromise_age": max(self.compromise_age.values(), default=0),
        }
        info.update(reward_decomp)
        return obs, reward, False, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step       = 0
        self.compromised        = {RED_START_HOST}
        self.investigated       = set()
        self._prev_investigated = set()
        self.downtime           = {}
        self.compromise_age     = {RED_START_HOST: 1}
        self.belief_estimator.reset()

        return self._make_obs(), {}

    def render(self):
        """Simple text render for debugging."""
        print(f"\n── Step {self.current_step} ──")
        for host in HOSTS:
            status = "COMPROMISED" if host in self.compromised else (
                f"DOWNTIME({self.downtime[host]})" if host in self.downtime else "clean"
            )
            age = self.compromise_age.get(host, 0)
            print(f"  {host:<15} {status}  (age={age})")


# ── Backward-compatibility alias ──────────────────────────────────────────────
# The old wrapper.py exported SchoolIRSWrapper.  Code that imports it by name
# (e.g. evaluate.py, train.py) still works via this alias.
class SchoolIRSWrapper(SchoolIRSEnv):
    """
    Legacy alias: SchoolIRSWrapper → SchoolIRSEnv.

    The original constructor accepted a `cyborg_env` argument which is no longer
    used.  It is accepted and silently ignored for backward compatibility.
    """
    def __init__(self, cyborg_env=None, **kwargs):
        super().__init__(**kwargs)