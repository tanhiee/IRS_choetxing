"""
Central configuration for the IRS-RL project.
All tunable knobs live here so other modules import from this file.

v2: Added RecurrentPPO params, Investigate mechanic costs,
    Exponential compromise penalty, and Downtime/Cooldown constants.
"""

# ── Network Topology ──────────────────────────────────────────────────────────
HOSTS = [
    "Admin_PC",
    "Teacher_PC",
    "Student_PC1",
    "Student_PC2",
    "File_Server",
    "Web_Server",
]

SUBNETS = {
    "Admin":   "10.0.0.0/24",
    "Student": "10.0.1.0/24",
    "Server":  "10.0.2.0/24",
}

# ── Episode Settings ───────────────────────────────────────────────────────────
MAX_STEPS  = 200   # Hard cap per episode
STEP_LIMIT = 200   # Used by ScenarioGenerator.determine_done (legacy)

# ── Environment Difficulty ─────────────────────────────────────────────────────
# Probability that a CLEAN host fires a false-positive SIEM alert per step.
# This is the core source of partial observability (POMDP noise).
FALSE_POSITIVE_RATE = 0.15

# ── Downtime / Cooldown Mechanic ───────────────────────────────────────────────
# When a host is Restored, it enters a cooldown for DOWNTIME_MIN..DOWNTIME_MAX
# timesteps before returning to operational status.
DOWNTIME_MIN = 3
DOWNTIME_MAX = 5

# ── Asset Weights ──────────────────────────────────────────────────────────────
# Multiplier applied to the clean-host reward and compromise penalty per host.
# Servers are worth 5× more than student endpoints.
HOST_WEIGHTS = {
    "Admin_PC":   2.0,
    "Teacher_PC": 2.0,
    "Student_PC1": 1.0,
    "Student_PC2": 1.0,
    "File_Server": 10.0,   # Core asset — heavy penalty if compromised
    "Web_Server":  10.0,
}

# ── Reward Shaping ─────────────────────────────────────────────────────────────
#
# DESIGN RATIONALE (v3):
#
#   step_penalty         — small constant overhead to discourage stalling.
#
#   clean_host           — per-step bonus per ONLINE & CLEAN host.
#
#   compromise_base /    — exponential age-based penalty per compromised host:
#   compromise_exp_scale     penalty = -base * exp(scale * age)
#
#   investigate_cost     — REDUCED to -0.5 (was -2.0).
#                          Cheaper investigation = agent investigates more freely,
#                          learning the "look before you leap" pattern faster.
#
#   investigate_success  — NEW: +3.0 bonus when Investigate reveals a TRUE
#                          POSITIVE (host was genuinely compromised).
#                          Directly rewards the agent for finding real threats,
#                          giving the LSTM an immediate signal to value
#                          investigation as a detection tool, not just a cost.
#
#   restore_bonus        — REDUCED to +5.0 (was +8.0).
#                          Prevents spam-restore for the bonus alone; the agent
#                          must confirm via Investigate before it's worth it.
#
#   wasted_restore       — INCREASED to -30.0 (was -10.0).
#                          Now "really hurts": -30 wasted + downtime penalty
#                          completely wipes out multiple steps of clean_host gains.
#                          This is the primary force that drives Investigate-first.
#
#   invalid_action       — NEW: -15.0 penalty for any action (Restore or
#                          Investigate) on a host that is currently in DOWNTIME.
#                          Teaches the agent that acting on recovering hosts is
#                          wasteful. Alternative to Action Masking — simpler and
#                          compatible with MultiDiscrete spaces out-of-the-box.
#
#   downtime_penalty     — INCREASED to -2.0 (was -1.0).
#                          Stronger availability signal: downtime is expensive.
#
#   do_nothing_bonus     — small bonus for restraint when network is fully clean.
#
REWARD = {
    "step_penalty":           -0.5,   # constant per-step overhead
    "clean_host":             +1.0,   # per clean, online host per step
    "compromise_base":         2.0,   # base coefficient for exp penalty
    "compromise_exp_scale":    0.15,  # growth rate: penalty = base*exp(scale*age)
    "investigate_cost":       -0.1,   # SOC-analyst cost per Investigate (was -0.5)
    "investigate_success":    +3.0,   # NEW: bonus for detecting a true positive
    "restore_bonus":          +5.0,   # one-time: successfully cleared a host (was +8.0)
    "wasted_restore":         -30.0,  # restored a clean host - really hurts (was -10.0)
    "invalid_action":         -15.0,  # NEW: acted on a host in DOWNTIME
    "downtime_penalty":       -2.0,   # per-step while host is in cooldown (was -1.0)
    "do_nothing_bonus":       +1.0,   # clean + do-nothing bonus
}

# ── Red Attacker Parameters ────────────────────────────────────────────────────
SPREAD_PROB      = 0.35   # P(Red spreads to one new host per step if active)
RED_REENTRY_PROB = 0.10   # P(Red re-enters after being fully cleared)
RED_START_HOST   = "Student_PC1"

# ── RecurrentPPO Hyperparameters (v3 — speed-optimised) ──────────────────────
#
# Speed improvements vs v2:
#   n_steps:         512 → 256   — half-length rollouts → 2× more gradient
#                                   updates per wall-clock hour
#   lstm_hidden_size: 128 → 64  — smaller LSTM → ~40% faster forward/backward
#                                   pass; still sufficient for a 6-host POMDP
#   learning_rate:   3e-4 → 5e-4 — slightly more aggressive with sharper rewards
#   ent_coef:       0.01 → 0.02  — more entropy encourages exploration early on,
#                                   helping LSTM discover Investigate pattern
#
# Recommended total_timesteps: 500_000
#   With n_steps=256 and FPS~50+, 500k ≈ 25-35 min on CPU.
#   The sharper reward signals (wasted=-30, inv_success=+3) should make
#   convergence much faster than the 200k run.
RPPO_PARAMS = {
    "policy":           "MlpLstmPolicy",
    "verbose":          1,
    "tensorboard_log":  "./logs/",
    "learning_rate":    5e-4,       # slightly more aggressive (was 3e-4)
    "n_steps":          256,        # shorter rollouts for more frequent updates (was 512)
    "batch_size":       64,
    "n_epochs":         10,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "ent_coef":         0.02,       # more exploration early on (was 0.01)
    "seed":             42,
    "policy_kwargs": {
        "lstm_hidden_size":   64,   # smaller = faster (was 128); enough for 6 hosts
        "n_lstm_layers":       1,
        "shared_lstm":       False,
        "enable_critic_lstm": True,
    },
}

# Legacy PPO params (kept for backward compatibility with train.py)
PPO_PARAMS = {
    "policy":           "MlpPolicy",
    "verbose":          1,
    "tensorboard_log":  "./logs/",
    "learning_rate":    3e-4,
    "n_steps":          2048,
    "batch_size":       64,
    "n_epochs":         10,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "ent_coef":         0.01,
    "seed":             42,
}

# ── Run Configuration ──────────────────────────────────────────────────────────
TOTAL_TIMESTEPS      = 500_000    # 500k recommended with v3 reward shaping (~30min on CPU)
QUICK_TIMESTEPS      =    20_000   # fast smoke-test (~2 min on CPU)
EVAL_FREQ            =    10_000
EVAL_EPISODES        =        10
BENCHMARK_EPISODES   =        50

# ── File Paths ─────────────────────────────────────────────────────────────────
MODEL_PATH           = "ppo_school_irs_final"        # legacy PPO checkpoint
MODEL_PATH_RPPO      = "rppo_irs_final"              # RecurrentPPO checkpoint
RESULTS_DIR          = "results"
TRAINING_REWARDS_CSV = "logs/training_rewards.csv"
RPPO_REWARDS_CSV     = "logs/rppo_training_rewards.csv"