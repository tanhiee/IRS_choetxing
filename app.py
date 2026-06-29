"""
app.py — IRS-RL Dashboard (v2: RPPO-native)
============================================
Multi-tab Streamlit dashboard showing:
  Tab 1 — 📈 Training Progress   : RPPO reward curve with rolling average
  Tab 2 — 📊 Evaluation Results  : RPPO vs StaticPlaybook comparison charts
  Tab 3 — 🖥️  Live Simulation     : Real-time RPPO agent running the environment
  Tab 4 — ℹ️  Model Info          : Architecture & reward config summary
"""

import os
import time
import random

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pyvis.network import Network

# ── Project imports ───────────────────────────────────────────────────────────
from wrapper import SchoolIRSEnv
from config import (
    HOSTS, HOST_WEIGHTS, MODEL_PATH_RPPO,
    REWARD, RPPO_PARAMS, RPPO_REWARDS_CSV,
    TOTAL_TIMESTEPS,
)

try:
    from sb3_contrib import RecurrentPPO
    RPPO_AVAILABLE = True
except ImportError:
    RPPO_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IRS-RL RPPO Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #8892b0; margin-top: 4px; }
    .status-compromised { color: #ff4757; font-weight: 700; }
    .status-clean       { color: #2ed573; font-weight: 700; }
    .status-downtime    { color: #ffa502; font-weight: 700; }
    .log-box {
        background: #0d0f1a;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        color: #a8b2d8;
        max-height: 350px;
        overflow-y: auto;
    }
    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ IRS-RL · RecurrentPPO Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Autonomous Incident Response** — POMDP + LSTM · School Network", unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_train, tab_eval, tab_sim, tab_info = st.tabs([
    "📈 Training Progress",
    "📊 Evaluation Results",
    "🖥️  Live Simulation",
    "ℹ️  Model Info",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Progress
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("📈 RecurrentPPO Training Curve")

    csv_path = RPPO_REWARDS_CSV
    if not os.path.exists(csv_path):
        st.warning(f"Training log not found at `{csv_path}`. Run `python train_rppo.py` first.")
    else:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # ── Controls ─────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            window = st.slider("Rolling average window (episodes)", 10, 200, 50, 10)
        with c2:
            # Clip extreme outliers for display
            clip_pct = st.slider("Clip outlier percentile (%)", 0, 10, 2)
        with c3:
            auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

        if auto_refresh:
            time.sleep(5)
            st.rerun()

        # Clip
        lo = df["reward"].quantile(clip_pct / 100)
        hi = df["reward"].quantile(1 - clip_pct / 100)
        df_clipped = df[(df["reward"] >= lo) & (df["reward"] <= hi)].copy()
        df_clipped["rolling_mean"] = df_clipped["reward"].rolling(window, min_periods=1).mean()
        df_clipped["rolling_std"]  = df_clipped["reward"].rolling(window, min_periods=1).std().fillna(0)

        # ── Plot ─────────────────────────────────────────────────────────────
        fig = go.Figure()

        # Shaded std band
        fig.add_trace(go.Scatter(
            x=pd.concat([df_clipped["episode"], df_clipped["episode"].iloc[::-1]]),
            y=pd.concat([
                df_clipped["rolling_mean"] + df_clipped["rolling_std"],
                (df_clipped["rolling_mean"] - df_clipped["rolling_std"]).iloc[::-1],
            ]),
            fill="toself",
            fillcolor="rgba(123, 47, 247, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1 Std Dev",
            showlegend=True,
        ))

        # Raw rewards (faint)
        fig.add_trace(go.Scatter(
            x=df_clipped["episode"], y=df_clipped["reward"],
            mode="markers",
            marker=dict(size=2, color="#2d3561", opacity=0.5),
            name="Episode Reward",
        ))

        # Rolling mean
        fig.add_trace(go.Scatter(
            x=df_clipped["episode"], y=df_clipped["rolling_mean"],
            mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            name=f"Rolling Mean (w={window})",
        ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#ff4757", opacity=0.5,
                      annotation_text="Break-even", annotation_position="right")

        # Key milestone annotations
        first_positive = df_clipped[df_clipped["rolling_mean"] > 0]
        if not first_positive.empty:
            ep = int(first_positive["episode"].iloc[0])
            fig.add_vline(x=ep, line_dash="dot", line_color="#2ed573", opacity=0.6,
                          annotation_text=f"First positive avg (ep {ep})",
                          annotation_position="top right")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            title=dict(text="Episode Reward over Training", font=dict(size=16, color="#ccd6f6")),
            xaxis_title="Episode",
            yaxis_title="Episode Reward",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=430,
            margin=dict(l=50, r=30, t=60, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        st.markdown("#### Summary Statistics")
        total_eps = len(df)
        last_100_mean = df["reward"].tail(100).mean()
        best_mean     = df_clipped["rolling_mean"].max()
        worst_mean    = df_clipped["rolling_mean"].min()
        pct_positive  = (df["reward"] > 0).mean() * 100

        cols = st.columns(5)
        for col, label, val, fmt in zip(cols, [
            "Total Episodes", "Last 100 Mean", "Best Rolling Mean",
            "Worst Rolling Mean", "% Positive Episodes",
        ], [total_eps, last_100_mean, best_mean, worst_mean, pct_positive], [
            "{:.0f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}%"
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fmt.format(val)}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("📊 RecurrentPPO vs Static Playbook — 50 Episodes")

    eval_path = "results/evaluation_rppo.csv"
    if not os.path.exists(eval_path):
        st.warning(f"Evaluation data not found at `{eval_path}`. Run `python evaluate_rppo.py` first.")
    else:
        ev = pd.read_csv(eval_path)
        rppo = ev[ev["agent"] == "RecurrentPPO"]
        play = ev[ev["agent"] == "StaticPlaybook"]

        palette = {"RecurrentPPO": "#00d4ff", "StaticPlaybook": "#7b2ff7"}

        # ── KPI cards ────────────────────────────────────────────────────────
        st.markdown("#### Key Performance Indicators")
        kpi_cols = st.columns(4)
        kpis = [
            ("Mean Reward",        rppo["total_reward"].mean(),  play["total_reward"].mean(),  "{:.0f}"),
            ("Clean Finish %",     rppo["ended_clean"].mean()*100, play["ended_clean"].mean()*100, "{:.0f}%"),
            ("Mean Time to Recover", rppo["time_to_recovery"].mean(), play["time_to_recovery"].mean(), "{:.1f} steps"),
            ("Wasted Restores/Ep", rppo["total_wasted"].mean(),  play["total_wasted"].mean(),  "{:.1f}"),
        ]
        for col, (label, rv, pv, fmt) in zip(kpi_cols, kpis):
            better = rv >= pv if label != "Wasted Restores/Ep" else rv <= pv
            delta_color = "normal" if better else "inverse"
            col.metric(
                label=label,
                value=fmt.format(rv),
                delta=f"Playbook: {fmt.format(pv)}",
                delta_color=delta_color,
            )

        st.divider()

        # ── Side-by-side charts ───────────────────────────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            # Reward distribution box plot
            fig_box = go.Figure()
            for agent, df_ag in [("RecurrentPPO", rppo), ("StaticPlaybook", play)]:
                fig_box.add_trace(go.Box(
                    y=df_ag["total_reward"],
                    name=agent,
                    marker_color=palette[agent],
                    boxmean="sd",
                    line_width=2,
                ))
            fig_box.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                title="Total Reward Distribution", height=340,
                margin=dict(l=40, r=20, t=50, b=40),
                yaxis_title="Episode Reward",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            # Action breakdown stacked bar
            metrics_rppo = {
                "Investigate":       rppo["total_invested"].mean(),
                "Successful Restore": rppo["total_restored"].mean(),
                "Wasted Restore":    rppo["total_wasted"].mean(),
            }
            metrics_play = {
                "Investigate":       play["total_invested"].mean(),
                "Successful Restore": play["total_restored"].mean(),
                "Wasted Restore":    play["total_wasted"].mean(),
            }
            fig_bar = go.Figure(data=[
                go.Bar(name="RecurrentPPO",  x=list(metrics_rppo.keys()), y=list(metrics_rppo.values()),
                       marker_color="#00d4ff"),
                go.Bar(name="StaticPlaybook", x=list(metrics_play.keys()), y=list(metrics_play.values()),
                       marker_color="#7b2ff7"),
            ])
            fig_bar.update_layout(
                barmode="group",
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                title="Mean Action Counts per Episode",
                height=340,
                margin=dict(l=40, r=20, t=50, b=40),
                yaxis_title="Count",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Time to Recovery scatter ────────────────────────────────────────
        fig_ttr = go.Figure()
        for agent, df_ag in [("RecurrentPPO", rppo), ("StaticPlaybook", play)]:
            fig_ttr.add_trace(go.Scatter(
                x=df_ag["episode"],
                y=df_ag["time_to_recovery"],
                mode="markers+lines",
                name=agent,
                marker=dict(size=5, color=palette[agent]),
                line=dict(width=1.5, color=palette[agent]),
            ))
        fig_ttr.add_hline(y=rppo["time_to_recovery"].mean(), line_dash="dot",
                          line_color="#00d4ff", opacity=0.7,
                          annotation_text=f"RPPO Mean: {rppo['time_to_recovery'].mean():.1f}",
                          annotation_position="left")
        fig_ttr.add_hline(y=play["time_to_recovery"].mean(), line_dash="dot",
                          line_color="#7b2ff7", opacity=0.7,
                          annotation_text=f"Playbook Mean: {play['time_to_recovery'].mean():.1f}",
                          annotation_position="right")
        fig_ttr.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title="Time to Recovery per Episode (steps — lower is better)",
            height=320, margin=dict(l=40, r=20, t=50, b=40),
            xaxis_title="Episode", yaxis_title="Steps to Recovery",
        )
        st.plotly_chart(fig_ttr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.subheader("🖥️ Live RPPO Agent Simulation")

    OBS_CHANNELS = 4
    ACTION_LABELS = {0: "DoNothing", 1: "Investigate", 2: "RESTORE"}
    ACTION_EMOJI  = {0: "💤", 1: "🔍", 2: "🔴"}

    # ── Network graph ─────────────────────────────────────────────────────────
    def draw_network_rppo(env: SchoolIRSEnv, obs: np.ndarray) -> str:
        net = Network(height="420px", width="100%", bgcolor="#0e1117", font_color="white")
        net.add_node("Core_Switch", label="Core Switch\n🔷",
                     color="#0078D7", shape="database", size=35)

        for i, host in enumerate(HOSTS):
            base  = i * OBS_CHANNELS
            alert = int(obs[base + 0])
            dt    = obs[base + 2]
            belief = obs[base + 3]

            if host in env.compromised:
                color, shape, size = "#ff4757", "dot", 38
                label = f"{host}\n💀 COMPROMISED\n(Belief: {belief:.1%})"
            elif host in env.downtime:
                color, shape, size = "#ffa502", "dot", 28
                label = f"{host}\n⚙️ DOWNTIME\n(Belief: {belief:.1%})"
            elif alert == 1:
                color, shape, size = "#ffdd59", "dot", 26
                label = f"{host}\n⚠️ ALERT\n(Belief: {belief:.1%})"
            else:
                color, shape, size = "#2ed573", "dot", 22
                label = f"{host}\n✅ clean\n(Belief: {belief:.1%})"

            weight = HOST_WEIGHTS.get(host, 1.0)
            label += f" (w={weight:.0f})"
            net.add_node(host, label=label, color=color, shape=shape, size=size)
            net.add_edge("Core_Switch", host, color="#2d3561")

        path = "network_map_rppo.html"
        net.save_graph(path)
        return path

    # ── Session state init ────────────────────────────────────────────────────
    if "sim_env" not in st.session_state:
        st.session_state.sim_env      = SchoolIRSEnv()
        st.session_state.sim_obs, _   = st.session_state.sim_env.reset()
        st.session_state.sim_model    = None
        st.session_state.sim_logs     = []
        st.session_state.sim_running  = False
        st.session_state.sim_reward   = 0.0
        st.session_state.sim_step     = 0
        st.session_state.lstm_state   = None
        st.session_state.ep_start     = np.ones((1,), dtype=bool)
        st.session_state.action_counts = {0: 0, 1: 0, 2: 0}
        st.session_state.reward_history = []

    # ── Model loader ──────────────────────────────────────────────────────────
    model_zip  = f"{MODEL_PATH_RPPO}.zip"
    best_zip   = "logs/best_rppo/best_model.zip"
    adv_zip    = "results/models/rppo_defender_adv.zip"
    avail_models = {}
    if os.path.exists(adv_zip):   avail_models["Adversarial Defender (Ours)"] = "results/models/rppo_defender_adv"
    if os.path.exists(model_zip): avail_models["Last checkpoint"] = MODEL_PATH_RPPO
    if os.path.exists(best_zip):  avail_models["Best checkpoint"] = "logs/best_rppo/best_model"

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Simulation Control")

        if avail_models:
            chosen = st.selectbox("Model checkpoint", list(avail_models.keys()))
            model_path = avail_models[chosen]
            if st.button("🔌 Load Model", use_container_width=True):
                with st.spinner("Loading RPPO model…"):
                    st.session_state.sim_model = RecurrentPPO.load(model_path) if RPPO_AVAILABLE else None
                st.success(f"Loaded: {model_path}.zip")
        else:
            st.warning("No trained model found. Run `python train_rppo.py` first.")

        step_delay = st.slider("Step delay (sec)", 0.1, 2.0, 0.6, 0.1)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶️ Start", use_container_width=True):
                if st.session_state.sim_model:
                    st.session_state.sim_running = True
                else:
                    st.warning("Load a model first!")
        with col_b:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.sim_running = False

        if st.button("🔄 Reset Episode", use_container_width=True):
            st.session_state.sim_running   = False
            st.session_state.sim_obs, _   = st.session_state.sim_env.reset()
            st.session_state.sim_logs     = []
            st.session_state.sim_reward   = 0.0
            st.session_state.sim_step     = 0
            st.session_state.lstm_state   = None
            st.session_state.ep_start     = np.ones((1,), dtype=bool)
            st.session_state.action_counts = {0: 0, 1: 0, 2: 0}
            st.session_state.reward_history = []
            st.rerun()

    # ── Layout ────────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([6, 4])

    with left_col:
        graph_ph = st.empty()
        # Draw initial graph
        html_path = draw_network_rppo(st.session_state.sim_env, st.session_state.sim_obs)
        with open(html_path, "r", encoding="utf-8") as f:
            graph_ph.empty()
            components.html(f.read(), height=430)

    with right_col:
        # ── Metrics row ───────────────────────────────────────────────────────
        env_ref  = st.session_state.sim_env
        n_comp   = len(env_ref.compromised)
        n_down   = len(env_ref.downtime)
        n_clean  = len(HOSTS) - n_comp - n_down

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Step",   st.session_state.sim_step)
        m2.metric("Reward", f"{st.session_state.sim_reward:+.0f}")
        m3.metric("🔴 Comp",  n_comp,  delta=None)
        m4.metric("⚙️ Down",  n_down,  delta=None)

        # ── Action breakdown mini-bar ─────────────────────────────────────────
        ac = st.session_state.action_counts
        total_ac = max(sum(ac.values()), 1)
        fig_ac = go.Figure(go.Bar(
            x=[ACTION_EMOJI[k] + " " + ACTION_LABELS[k] for k in [0,1,2]],
            y=[ac.get(k, 0) / total_ac * 100 for k in [0,1,2]],
            marker_color=["#2d3561", "#00d4ff", "#ff4757"],
        ))
        fig_ac.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            height=200, margin=dict(l=20, r=20, t=20, b=30),
            yaxis_title="%", showlegend=False,
        )
        st.plotly_chart(fig_ac, use_container_width=True, key="ac_bar")

        # ── Reward sparkline ──────────────────────────────────────────────────
        if st.session_state.reward_history:
            fig_spark = go.Figure(go.Scatter(
                y=st.session_state.reward_history[-60:],
                mode="lines",
                line=dict(color="#00d4ff", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.08)",
            ))
            fig_spark.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                height=160, margin=dict(l=20, r=20, t=5, b=20),
                xaxis=dict(showticklabels=False),
                yaxis_title="Reward",
            )
            st.plotly_chart(fig_spark, use_container_width=True, key="spark")

        # ── SIEM log ─────────────────────────────────────────────────────────
        log_lines = st.session_state.sim_logs[-18:]
        log_html  = "<br>".join(log_lines) if log_lines else "Awaiting events…"
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # ── Step logic ────────────────────────────────────────────────────────────
    if st.session_state.sim_running and st.session_state.sim_model:
        obs    = st.session_state.sim_obs
        model  = st.session_state.sim_model
        env    = st.session_state.sim_env

        # Check expected observation dimension
        expected_dim = model.observation_space.shape[0]
        if expected_dim == 18:
            # Reshape & slice out the belief channel for legacy models
            obs_reshaped = obs.reshape(6, 4)
            obs_sliced = obs_reshaped[:, :3]
            model_obs = obs_sliced.flatten()
        else:
            model_obs = obs

        # LSTM-aware prediction
        actions_batch, new_lstm = model.predict(
            model_obs[np.newaxis, :],
            state=st.session_state.lstm_state,
            episode_start=st.session_state.ep_start,
            deterministic=True,
        )
        st.session_state.lstm_state = new_lstm
        st.session_state.ep_start   = np.zeros((1,), dtype=bool)
        actions = actions_batch[0]

        # Log decisions
        for i, host in enumerate(HOSTS):
            act   = int(actions[i])
            base  = i * OBS_CHANNELS
            alert = int(obs[base])
            st.session_state.action_counts[act] = st.session_state.action_counts.get(act, 0) + 1

            if alert == 1 or host in env.compromised:
                emoji  = ACTION_EMOJI[act]
                label  = ACTION_LABELS[act]
                status = "🔴 REAL" if host in env.compromised else "⚠️ ALERT"
                st.session_state.sim_logs.append(
                    f"<span style='color:#8892b0'>[{st.session_state.sim_step+1:03d}]</span> "
                    f"{status} <b style='color:#ccd6f6'>{host}</b> → {emoji} {label}"
                )

        # Step
        new_obs, reward, _, truncated, info = env.step(actions)
        st.session_state.sim_obs    = new_obs
        st.session_state.sim_reward += reward
        st.session_state.sim_step   += 1
        st.session_state.reward_history.append(reward)

        if info["wasted"]:
            st.session_state.sim_logs.append(
                f"<span style='color:#ff6b81'>⚠️ {info['wasted']} WASTED restore(s) this step!</span>"
            )
        if info["true_compromised"] == 0:
            st.session_state.sim_logs.append(
                "<span style='color:#2ed573'>✅ Network CLEAN!</span>"
            )

        time.sleep(step_delay)

        if truncated:
            st.session_state.sim_running = False
            st.session_state.sim_logs.append(
                f"<span style='color:#ffd32a'>━━━ EPISODE END (step {st.session_state.sim_step}) "
                f"│ Total Reward: {st.session_state.sim_reward:+.1f} ━━━</span>"
            )

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Info
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("ℹ️ Architecture & Configuration")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🧠 RPPO Hyperparameters")
        params_display = {
            "Policy":          RPPO_PARAMS["policy"],
            "Learning Rate":   str(RPPO_PARAMS["learning_rate"]),
            "n_steps":         str(RPPO_PARAMS["n_steps"]),
            "Batch Size":      str(RPPO_PARAMS["batch_size"]),
            "n_epochs":        str(RPPO_PARAMS["n_epochs"]),
            "Gamma (γ)":       str(RPPO_PARAMS["gamma"]),
            "GAE λ":           str(RPPO_PARAMS["gae_lambda"]),
            "Entropy Coef":    str(RPPO_PARAMS["ent_coef"]),
            "LSTM Hidden":     str(RPPO_PARAMS["policy_kwargs"]["lstm_hidden_size"]),
            "LSTM Layers":     str(RPPO_PARAMS["policy_kwargs"]["n_lstm_layers"]),
            "Total Timesteps": f"{TOTAL_TIMESTEPS:,}",
        }
        st.table(pd.DataFrame(params_display.items(), columns=["Parameter", "Value"]))

    with c2:
        st.markdown("#### 💰 Reward Function (v3)")
        reward_display = {k: str(v) for k, v in REWARD.items()}
        df_r = pd.DataFrame(reward_display.items(), columns=["Key", "Value"])
        st.table(df_r)

    st.divider()
    st.markdown("#### 🌐 Network Topology & Host Weights")
    hw_data = [(h, HOST_WEIGHTS[h]) for h in HOSTS]
    fig_hw = go.Figure(go.Bar(
        x=[h[0] for h in hw_data],
        y=[h[1] for h in hw_data],
        marker_color=["#ffa502" if HOST_WEIGHTS[h] > 5 else "#00d4ff" for h in HOSTS],
        text=[str(v) for _, v in hw_data],
        textposition="outside",
    ))
    fig_hw.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Host Asset Weights (higher = more critical)", height=320,
        yaxis_title="Weight", margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_hw, use_container_width=True)

    st.markdown("""
    #### 🔑 How the LSTM + Investigate Mechanic Works

    | Step | Observation | Agent Action | Result |
    |------|------------|-------------|--------|
    | `t`   | `alert=1, inv_prev=0` (ambiguous) | **Investigate** Host X | Marks X; cost = -0.5 |
    | `t+1` | `alert=1, inv_prev=1` (verified!) | **RESTORE** Host X | Clears threat; bonus = +5.0 |
    | `t+2` | `alert=0, inv_prev=0, downtime>0` | **DoNothing** | Wait for cooldown |

    The LSTM hidden state `(h, c)` carries memory of the Investigate action at step `t`,
    allowing the agent to correctly interpret `inv_prev=1` at step `t+1` as "this is real" 
    and safely commit to a Restore.
    """)