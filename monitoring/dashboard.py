#!/usr/bin/env python3
"""
Real-time RL Agent Monitoring Dashboard

Provides live monitoring of:
- PnL performance
- Win rate and trade statistics
- Action distributions
- Risk metrics
- Model behavior

Usage:
    streamlit run monitoring/dashboard.py

Access:
    http://localhost:8501
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import time
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Backtest engine imports
from monitoring.backtest_engine import (
    load_all_backtest_trades,
    load_backtest_trades_for_date,
    load_daily_summaries,
    load_patterns,
    load_processed_dates_list,
    detect_patterns,
)

st.set_page_config(
    page_title="RL Trading Agent Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_trades(db_path: str = "data/trades.db", hours: int = 24) -> pd.DataFrame:
    """Load recent trades from SQLite"""
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)

    try:
        # Load recent trades (opened_at is TEXT timestamp)
        df = pd.read_sql_query("""
            SELECT * FROM trades
            ORDER BY id DESC
            LIMIT 10000
        """, conn)
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def load_metrics(db_path: str = "data/metrics.db", hours: int = 24) -> pd.DataFrame:
    """Load performance metrics from SQLite"""
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    cutoff = time.time() - (hours * 3600)

    try:
        rows = conn.execute("""
            SELECT timestamp, metrics_json FROM performance_metrics
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (cutoff,)).fetchall()

        if not rows:
            return pd.DataFrame()

        import json
        data = [json.loads(row[1]) for row in rows]
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def get_model_version(asset: str = "btc") -> str:
    """Get current model version for asset"""
    try:
        from model_versioning import ModelRegistry
        registry = ModelRegistry(f"model_registry/{asset}")
        version = registry.get_current_version()
        return version if version else "No model deployed"
    except:
        return "Unknown"


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# Header (asset selected below in sidebar, so we use placeholder first)
# Title will be updated after asset selection

# Sidebar controls
st.sidebar.header("⚙️ Settings")
time_window = st.sidebar.selectbox(
    "Time Window",
    ["1 Hour", "6 Hours", "12 Hours", "24 Hours", "7 Days"],
    index=3
)
window_hours = {
    "1 Hour": 1,
    "6 Hours": 6,
    "12 Hours": 12,
    "24 Hours": 24,
    "7 Days": 168
}[time_window]

auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
show_raw_data = st.sidebar.checkbox("Show Raw Data Tables", value=False)

# Asset selector
asset = st.sidebar.selectbox(
    "Asset",
    ["BTC", "ETH", "SOL", "XRP",
     "Bucket9-BTC", "Bucket9-ETH", "Bucket9-SOL", "Bucket9-XRP",
     "Oracle-BTC", "Oracle-ETH", "Oracle-SOL", "Oracle-XRP",
     "Ensemble-BTC", "Ensemble-ETH", "Ensemble-SOL", "Ensemble-XRP",
     "Optimal-BTC", "Optimal-ETH", "Optimal-SOL", "Optimal-XRP",
     "Backtest-BTC", "Backtest-ETH", "Backtest-SOL", "Backtest-XRP"],
    index=0
)

# Determine if backtest mode
is_backtest = asset.startswith("Backtest-")
backtest_asset = asset.replace("Backtest-", "").lower() if is_backtest else None

# Map display name to data directory
asset_lower_map = {
    "BTC": "btc",
    "ETH": "eth",
    "SOL": "sol",
    "XRP": "xrp",
    "Bucket9-BTC": "bucket9-btc",
    "Bucket9-ETH": "bucket9-eth",
    "Bucket9-SOL": "bucket9-sol",
    "Bucket9-XRP": "bucket9-xrp",
    "Oracle-BTC": "oracle-btc",
    "Oracle-ETH": "oracle-eth",
    "Oracle-SOL": "oracle-sol",
    "Oracle-XRP": "oracle-xrp",
    "Ensemble-BTC": "ensemble-btc",
    "Ensemble-ETH": "ensemble-eth",
    "Ensemble-SOL": "ensemble-sol",
    "Ensemble-XRP": "ensemble-xrp",
    "Optimal-BTC": "optimal-btc",
    "Optimal-ETH": "optimal-eth",
    "Optimal-SOL": "optimal-sol",
    "Optimal-XRP": "optimal-xrp",
    "Backtest-BTC": "btc",
    "Backtest-ETH": "eth",
    "Backtest-SOL": "sol",
    "Backtest-XRP": "xrp",
}
asset_lower = asset_lower_map[asset]

# Sync start time option - now syncs with Actual Trades learning start
if not is_backtest:
    sync_with_actual_trades = st.sidebar.checkbox("Show only Actual Trades era", value=True,
        help="Filter to show only trades after we started learning from actual trades")

    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Info")
    model_version = get_model_version(asset_lower)
    st.sidebar.text(f"Version: {model_version}")

    # Title (after asset selection)
    st.title(f"🤖 {asset} RL Trading Agent Monitor")
    st.caption("Online Q-Learning from Actual Trades")
else:
    sync_with_actual_trades = False
    model_version = "N/A"

# ============================================================================
# BACKTEST MODE — Full backtest dashboard
# ============================================================================

if is_backtest:
    st.title(f"🔬 {backtest_asset.upper()} Backtest Dashboard")
    st.caption("Pre-computed strategy backtesting on historical recording data")

    # Load pre-computed data
    all_trades_df = load_all_backtest_trades(backtest_asset)
    processed_dates = load_processed_dates_list(backtest_asset)
    patterns = load_patterns(backtest_asset)
    daily_summaries = load_daily_summaries(backtest_asset)

    if all_trades_df.empty:
        st.warning(
            f"⚠️ No backtest results found for {backtest_asset.upper()}.\n\n"
            f"Run the backtest runner first:\n"
            f"```\npython3 backtest_runner.py --asset {backtest_asset}\n```"
        )
        st.stop()

    # Sidebar: Strategy filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Backtest Settings")
    available_strategies = sorted(all_trades_df["strategy"].unique().tolist())
    selected_strategies = st.sidebar.multiselect(
        "Strategies",
        available_strategies,
        default=available_strategies,
    )

    # Filter trades by selected strategies
    filtered_df = all_trades_df[all_trades_df["strategy"].isin(selected_strategies)].copy()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.text(f"Dates processed: {len(processed_dates)}")
    st.sidebar.text(f"Total trades: {len(filtered_df)}")
    if processed_dates:
        st.sidebar.text(f"Range: {processed_dates[0]} → {processed_dates[-1]}")

    # ── TABS: Holistic vs Daily ──
    tab_holistic, tab_daily = st.tabs(["🔬 Holistic (All Days)", "📅 Daily (Per-Day)"])

    # ==================================================================
    # TAB 1: HOLISTIC (all days combined)
    # ==================================================================
    with tab_holistic:
        if filtered_df.empty:
            st.info("No trades for selected strategies.")
        else:
            # ── Key Metrics Row ──
            total_pnl = filtered_df["pnl"].sum()
            num_trades = len(filtered_df)
            wins = int((filtered_df["pnl"] > 0).sum())
            losses = num_trades - wins
            win_rate = wins / num_trades if num_trades > 0 else 0
            avg_pnl = filtered_df["pnl"].mean()
            std_pnl = filtered_df["pnl"].std()
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

            col1, col2, col3, col4, col5 = st.columns(5)
            pnl_display = total_pnl * 100  # convert to cents / dollars
            col1.metric("Total PnL", f"${pnl_display:+.2f}")
            col2.metric("Trades", num_trades)
            col3.metric("Win Rate", f"{win_rate:.1%}")
            col4.metric("Wins / Losses", f"{wins} / {losses}")
            col5.metric("Sharpe", f"{sharpe:.2f}")

            st.markdown("---")

            # ── Strategy Comparison Table ──
            if len(selected_strategies) > 1:
                st.subheader("📊 Strategy Comparison")
                comp_data = []
                for strat in selected_strategies:
                    strat_df = filtered_df[filtered_df["strategy"] == strat]
                    s_trades = len(strat_df)
                    s_wins = int((strat_df["pnl"] > 0).sum())
                    s_pnl = strat_df["pnl"].sum() * 100
                    s_wr = s_wins / s_trades * 100 if s_trades > 0 else 0
                    s_avg = strat_df["pnl"].mean() * 100 if s_trades > 0 else 0
                    s_std = strat_df["pnl"].std() if s_trades > 1 else 0
                    s_sharpe = (strat_df["pnl"].mean() / s_std) if s_std > 0 else 0
                    comp_data.append({
                        "Strategy": strat,
                        "Trades": s_trades,
                        "Wins": s_wins,
                        "Win Rate": f"{s_wr:.1f}%",
                        "Total PnL": f"${s_pnl:+.2f}",
                        "Avg PnL": f"${s_avg:+.2f}",
                        "Sharpe": f"{s_sharpe:.2f}",
                    })
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
                st.markdown("---")

            # ── Cumulative PnL Chart ──
            st.subheader("📈 Cumulative PnL (Equity Curve)")
            sorted_df = filtered_df.sort_values("timestamp_ms").copy()
            sorted_df["datetime"] = pd.to_datetime(sorted_df["timestamp_ms"], unit="ms")
            sorted_df["cum_pnl"] = sorted_df["pnl"].cumsum() * 100

            fig_eq = go.Figure()

            if len(selected_strategies) > 1:
                # Overlay per strategy
                colors = ["#00D9FF", "#FF6B6B", "#00FF88", "#FFD93D"]
                for idx, strat in enumerate(selected_strategies):
                    strat_sorted = filtered_df[filtered_df["strategy"] == strat].sort_values("timestamp_ms").copy()
                    strat_sorted["datetime"] = pd.to_datetime(strat_sorted["timestamp_ms"], unit="ms")
                    strat_sorted["cum_pnl"] = strat_sorted["pnl"].cumsum() * 100
                    fig_eq.add_trace(go.Scatter(
                        x=strat_sorted["datetime"], y=strat_sorted["cum_pnl"],
                        mode="lines", name=strat,
                        line=dict(color=colors[idx % len(colors)], width=2),
                    ))
            else:
                fig_eq.add_trace(go.Scatter(
                    x=sorted_df["datetime"], y=sorted_df["cum_pnl"],
                    mode="lines+markers", name="Cumulative PnL",
                    line=dict(color="#00D9FF", width=2),
                    marker=dict(size=4),
                ))

            fig_eq.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_eq.update_layout(
                xaxis_title="Date", yaxis_title="Cumulative PnL ($)",
                template="plotly_dark", height=450, hovermode="x unified",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── Daily PnL Bar Chart ──
            st.subheader("📅 Daily PnL")
            daily_pnl = filtered_df.groupby("date")["pnl"].sum().reset_index()
            daily_pnl["pnl_dollars"] = daily_pnl["pnl"] * 100
            daily_pnl["color"] = daily_pnl["pnl_dollars"].apply(lambda x: "green" if x >= 0 else "red")

            fig_daily = go.Figure()
            fig_daily.add_trace(go.Bar(
                x=daily_pnl["date"], y=daily_pnl["pnl_dollars"],
                marker_color=daily_pnl["color"].map({"green": "#00FF88", "red": "#FF4444"}),
                name="Daily PnL",
            ))
            fig_daily.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_daily.update_layout(
                xaxis_title="Date", yaxis_title="PnL ($)",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig_daily, use_container_width=True)

            st.markdown("---")

            # ── Pattern Detection Section ──
            st.subheader("🔍 Pattern Detection")

            # Recompute patterns from filtered data for current strategy selection
            live_patterns = detect_patterns(filtered_df)

            if live_patterns:
                pat_col1, pat_col2 = st.columns(2)

                # Best Hours Heatmap
                with pat_col1:
                    st.markdown("##### ⏰ Performance by Hour (UTC)")
                    hourly = live_patterns.get("hourly_performance", {})
                    if hourly:
                        hours = sorted(hourly.keys(), key=lambda x: int(x))
                        hour_labels = [f"{int(h):02d}:00" for h in hours]
                        hour_pnl = [hourly[h]["total_pnl"] * 100 for h in hours]
                        hour_wr = [hourly[h]["win_rate"] for h in hours]
                        hour_trades = [hourly[h]["trades"] for h in hours]

                        fig_hours = go.Figure()
                        fig_hours.add_trace(go.Bar(
                            x=hour_labels, y=hour_pnl,
                            marker_color=["#00FF88" if p >= 0 else "#FF4444" for p in hour_pnl],
                            text=[f"{t} trades" for t in hour_trades],
                            hovertemplate="%{x}<br>PnL: $%{y:.2f}<br>%{text}<extra></extra>",
                        ))
                        fig_hours.update_layout(
                            template="plotly_dark", height=300,
                            xaxis_title="Hour (UTC)", yaxis_title="PnL ($)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_hours, use_container_width=True)

                        best_h = live_patterns.get("best_hour")
                        worst_h = live_patterns.get("worst_hour")
                        if best_h is not None:
                            st.success(f"✅ Best hour: **{best_h:02d}:00 UTC** — ${hourly.get(str(best_h), hourly.get(best_h, {})).get('total_pnl', 0)*100:+.2f}")
                        if worst_h is not None:
                            st.error(f"❌ Worst hour: **{worst_h:02d}:00 UTC** — ${hourly.get(str(worst_h), hourly.get(worst_h, {})).get('total_pnl', 0)*100:+.2f}")

                # Day of Week Performance
                with pat_col2:
                    st.markdown("##### 📅 Performance by Day of Week")
                    dow = live_patterns.get("dow_performance", {})
                    if dow:
                        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        days = [d for d in day_order if d in dow]
                        day_pnl = [dow[d]["total_pnl"] * 100 for d in days]
                        day_trades = [dow[d]["trades"] for d in days]
                        day_wr = [dow[d]["win_rate"] for d in days]

                        fig_dow = go.Figure()
                        fig_dow.add_trace(go.Bar(
                            x=[d[:3] for d in days], y=day_pnl,
                            marker_color=["#00FF88" if p >= 0 else "#FF4444" for p in day_pnl],
                            text=[f"{t} trades, {wr:.0f}% WR" for t, wr in zip(day_trades, day_wr)],
                            hovertemplate="%{x}<br>PnL: $%{y:.2f}<br>%{text}<extra></extra>",
                        ))
                        fig_dow.update_layout(
                            template="plotly_dark", height=300,
                            xaxis_title="Day", yaxis_title="PnL ($)",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_dow, use_container_width=True)

                # Side Performance + Streak Analysis
                pat_col3, pat_col4 = st.columns(2)

                with pat_col3:
                    st.markdown("##### 🎯 YES vs NO Performance")
                    side_perf = live_patterns.get("side_performance", {})
                    if side_perf:
                        side_data = []
                        for side_name, stats in side_perf.items():
                            side_data.append({
                                "Side": side_name.upper(),
                                "Trades": stats["trades"],
                                "PnL": f"${stats['total_pnl']*100:+.2f}",
                                "Win Rate": f"{stats['win_rate']:.1f}%",
                            })
                        st.dataframe(pd.DataFrame(side_data), use_container_width=True, hide_index=True)

                with pat_col4:
                    st.markdown("##### 🔥 Streak Analysis")
                    max_win = live_patterns.get("max_win_streak", 0)
                    max_loss = live_patterns.get("max_loss_streak", 0)
                    st.metric("Max Win Streak", f"{max_win} trades")
                    st.metric("Max Loss Streak", f"{max_loss} trades")

            st.markdown("---")

            # ── PnL Distribution ──
            st.subheader("📊 PnL Distribution")
            fig_dist = go.Figure()
            pnl_cents = filtered_df["pnl"] * 100
            fig_dist.add_trace(go.Histogram(
                x=pnl_cents, nbinsx=30,
                marker=dict(color=pnl_cents, colorscale="RdYlGn", showscale=False),
                name="PnL",
            ))
            fig_dist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
            fig_dist.update_layout(
                xaxis_title="PnL ($)", yaxis_title="Frequency",
                template="plotly_dark", height=300, showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # ── Performance by Entry Price ──
            st.subheader("💰 Performance by Entry Price")
            bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
            labels = ['$0.00-0.10', '$0.10-0.20', '$0.20-0.30', '$0.30-0.40',
                      '$0.40-0.50', '$0.50-0.60', '$0.60-0.70', '$0.70-0.80',
                      '$0.80-0.90', '$0.90-1.00']
            filtered_df["price_interval"] = pd.cut(
                filtered_df["entry_price"], bins=bins, labels=labels, include_lowest=True
            )
            interval_stats = filtered_df.groupby("price_interval", observed=True).agg(
                Trades=("pnl", "count"),
                TotalPnL=("pnl", lambda x: x.sum() * 100),
                Wins=("pnl", lambda x: (x > 0).sum()),
                WinRate=("pnl", lambda x: (x > 0).mean() * 100),
            ).round(2)
            interval_stats["Losses"] = interval_stats["Trades"] - interval_stats["Wins"]
            interval_stats["Avg PnL"] = (interval_stats["TotalPnL"] / interval_stats["Trades"]).round(2)

            display_iv = interval_stats.copy()
            display_iv["TotalPnL"] = display_iv["TotalPnL"].apply(lambda x: f"${x:+.2f}")
            display_iv["Avg PnL"] = display_iv["Avg PnL"].apply(lambda x: f"${x:+.2f}")
            display_iv["WinRate"] = display_iv["WinRate"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(
                display_iv[["Trades", "Wins", "Losses", "WinRate", "TotalPnL", "Avg PnL"]],
                use_container_width=True,
            )

            st.markdown("---")

            # ── Recent Trades Table ──
            st.subheader("📋 All Backtest Trades")
            display_trades = filtered_df[[
                "date", "strategy", "ticker", "side", "entry_price",
                "exit_price", "pnl", "exit_reason", "hour_of_day"
            ]].copy()
            display_trades["pnl"] = display_trades["pnl"].apply(lambda x: f"${x*100:+.2f}")
            display_trades["entry_price"] = display_trades["entry_price"].apply(lambda x: f"${x:.3f}")
            display_trades["exit_price"] = display_trades["exit_price"].apply(lambda x: f"${x:.3f}")
            st.dataframe(display_trades, use_container_width=True, height=400)

    # ==================================================================
    # TAB 2: DAILY (per-day view with date picker)
    # ==================================================================
    with tab_daily:
        if not processed_dates:
            st.info("No processed dates found. Run backtest_runner.py first.")
        else:
            # Date picker
            selected_date = st.selectbox(
                "📅 Select Date",
                processed_dates[::-1],  # Most recent first
                index=0,
                help=f"Available: {processed_dates[0]} → {processed_dates[-1]}"
            )

            day_df = load_backtest_trades_for_date(backtest_asset, selected_date)
            day_df = day_df[day_df["strategy"].isin(selected_strategies)].copy()

            if day_df.empty:
                st.info(f"No trades on {selected_date} for selected strategies.")
            else:
                # Day summary metrics
                d_pnl = day_df["pnl"].sum() * 100
                d_trades = len(day_df)
                d_wins = int((day_df["pnl"] > 0).sum())
                d_wr = d_wins / d_trades * 100 if d_trades > 0 else 0

                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                dcol1.metric("Day PnL", f"${d_pnl:+.2f}")
                dcol2.metric("Trades", d_trades)
                dcol3.metric("Win Rate", f"{d_wr:.1f}%")
                dcol4.metric("Wins / Losses", f"{d_wins} / {d_trades - d_wins}")

                # Compare vs average
                if not all_trades_df.empty:
                    all_daily_pnl = all_trades_df.groupby("date")["pnl"].sum() * 100
                    avg_daily_pnl = all_daily_pnl.mean()
                    diff = d_pnl - avg_daily_pnl
                    if diff >= 0:
                        st.success(f"📈 This day was **${diff:+.2f}** better than average (avg: ${avg_daily_pnl:+.2f}/day)")
                    else:
                        st.error(f"📉 This day was **${diff:+.2f}** worse than average (avg: ${avg_daily_pnl:+.2f}/day)")

                st.markdown("---")

                # Intraday equity curve
                st.subheader(f"📈 Intraday Equity Curve — {selected_date}")
                day_sorted = day_df.sort_values("timestamp_ms").copy()
                day_sorted["datetime"] = pd.to_datetime(day_sorted["timestamp_ms"], unit="ms")
                day_sorted["cum_pnl"] = day_sorted["pnl"].cumsum() * 100

                fig_intra = go.Figure()
                fig_intra.add_trace(go.Scatter(
                    x=day_sorted["datetime"], y=day_sorted["cum_pnl"],
                    mode="lines+markers", name="Cumulative PnL",
                    line=dict(color="#00D9FF", width=2),
                    marker=dict(size=6, color=day_sorted["pnl"].apply(
                        lambda x: "#00FF88" if x > 0 else "#FF4444"
                    )),
                ))
                fig_intra.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_intra.update_layout(
                    xaxis_title="Time (UTC)", yaxis_title="PnL ($)",
                    template="plotly_dark", height=400, hovermode="x unified",
                )
                st.plotly_chart(fig_intra, use_container_width=True)

                # Strategy breakdown for the day
                if len(selected_strategies) > 1:
                    st.subheader("📊 Strategy Breakdown")
                    day_comp = []
                    for strat in selected_strategies:
                        sd = day_df[day_df["strategy"] == strat]
                        if sd.empty:
                            continue
                        day_comp.append({
                            "Strategy": strat,
                            "Trades": len(sd),
                            "Wins": int((sd["pnl"] > 0).sum()),
                            "Win Rate": f"{(sd['pnl'] > 0).mean()*100:.1f}%",
                            "PnL": f"${sd['pnl'].sum()*100:+.2f}",
                        })
                    if day_comp:
                        st.dataframe(pd.DataFrame(day_comp), use_container_width=True, hide_index=True)

                st.markdown("---")

                # Hourly breakdown for the day
                st.subheader("⏰ Hourly Breakdown")
                hourly_day = day_df.groupby("hour_of_day").agg(
                    Trades=("pnl", "count"),
                    PnL=("pnl", lambda x: x.sum() * 100),
                    WinRate=("pnl", lambda x: (x > 0).mean() * 100),
                ).round(2)

                fig_hd = go.Figure()
                fig_hd.add_trace(go.Bar(
                    x=[f"{h:02d}:00" for h in hourly_day.index],
                    y=hourly_day["PnL"],
                    marker_color=["#00FF88" if p >= 0 else "#FF4444" for p in hourly_day["PnL"]],
                    text=[f"{t} trades" for t in hourly_day["Trades"]],
                ))
                fig_hd.update_layout(
                    xaxis_title="Hour (UTC)", yaxis_title="PnL ($)",
                    template="plotly_dark", height=300, showlegend=False,
                )
                st.plotly_chart(fig_hd, use_container_width=True)

                # Trade log for the day
                st.subheader("📋 Trade Log")
                day_display = day_df[[
                    "strategy", "ticker", "side", "entry_price",
                    "exit_price", "pnl", "exit_reason", "seconds_to_expiry", "hour_of_day"
                ]].copy()
                day_display["pnl"] = day_display["pnl"].apply(lambda x: f"${x*100:+.2f}")
                day_display["entry_price"] = day_display["entry_price"].apply(lambda x: f"${x:.3f}")
                day_display["exit_price"] = day_display["exit_price"].apply(lambda x: f"${x:.3f}")
                st.dataframe(day_display, use_container_width=True, height=400)

    # Backtest footer
    st.markdown("---")
    last_processed = processed_dates[-1] if processed_dates else "N/A"
    st.caption(f"Last processed date: {last_processed} | Asset: {backtest_asset.upper()} | "
               f"Strategies: {', '.join(selected_strategies) if 'selected_strategies' in dir() else 'All'}")
    st.stop()  # Don't render live dashboard below

# ============================================================================
# ALL BOTS SUMMARY (always visible)
# ============================================================================

def get_asset_pnl(asset_name: str, hours: int, apply_sync: bool = False, cutoff_time: str = None) -> tuple:
    """Get PnL and trade count for an asset

    Args:
        asset_name: Name of the asset (e.g., 'btc', 'bucket9-xrp')
        hours: Time window in hours
        apply_sync: If True, apply ACTUAL_TRADES_START_TIME filter
        cutoff_time: Optional specific cutoff time (ISO format)
    """
    db_path = f"data/{asset_name}/trades.db"
    if not os.path.exists(db_path):
        return 0.0, 0, 0

    try:
        conn = sqlite3.connect(db_path)

        # Build query with cutoff in SQL for efficiency (no LIMIT issues)
        if cutoff_time:
            query = f"SELECT pnl, opened_at FROM trades WHERE opened_at >= '{cutoff_time}' ORDER BY id DESC"
        elif apply_sync:
            query = f"SELECT pnl, opened_at FROM trades WHERE opened_at >= '{ACTUAL_TRADES_START_TIME}' ORDER BY id DESC"
        else:
            query = "SELECT pnl, opened_at FROM trades ORDER BY id DESC LIMIT 10000"

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return 0.0, 0, 0

        total_pnl = df['pnl'].sum()
        trades = len(df)
        wins = (df['pnl'] > 0).sum()
        return total_pnl, trades, wins
    except:
        return 0.0, 0, 0

# Get all asset PnLs
# When we started learning from actual trades (online Q-learning deployment)
# Updated after retraining policies and adding longshot filter
ACTUAL_TRADES_START_TIME = "2026-02-11T15:00:00+00:00"

# Bucket9 optimization start times - asset-specific cutoffs
# SOL: Both YES+NO work, use earlier optimization time
BUCKET9_SOL_START = "2026-02-13T14:25:00+00:00"  # SOL: YES+NO directional trades
# BTC/ETH/XRP: NO-only trades after analysis showed YES causes 98%+ of losses
BUCKET9_NO_ONLY_START = "2026-02-13T23:39:00+00:00"  # BTC/ETH/XRP: NO trades only
btc_pnl, btc_trades, btc_wins = get_asset_pnl("btc", window_hours, sync_with_actual_trades)
eth_pnl, eth_trades, eth_wins = get_asset_pnl("eth", window_hours, sync_with_actual_trades)
sol_pnl, sol_trades, sol_wins = get_asset_pnl("sol", window_hours, sync_with_actual_trades)
xrp_pnl, xrp_trades, xrp_wins = get_asset_pnl("xrp", window_hours, sync_with_actual_trades)
# Bucket9: Asset-specific cutoff times
b9_btc_pnl, b9_btc_trades, b9_btc_wins = get_asset_pnl("bucket9-btc", window_hours, False, BUCKET9_NO_ONLY_START)
b9_eth_pnl, b9_eth_trades, b9_eth_wins = get_asset_pnl("bucket9-eth", window_hours, False, BUCKET9_NO_ONLY_START)
b9_sol_pnl, b9_sol_trades, b9_sol_wins = get_asset_pnl("bucket9-sol", window_hours, False, BUCKET9_SOL_START)
b9_xrp_pnl, b9_xrp_trades, b9_xrp_wins = get_asset_pnl("bucket9-xrp", window_hours, False, BUCKET9_NO_ONLY_START)

# Oracle bots
or_btc_pnl, or_btc_trades, or_btc_wins = get_asset_pnl("oracle-btc", window_hours)
or_eth_pnl, or_eth_trades, or_eth_wins = get_asset_pnl("oracle-eth", window_hours)
or_sol_pnl, or_sol_trades, or_sol_wins = get_asset_pnl("oracle-sol", window_hours)
or_xrp_pnl, or_xrp_trades, or_xrp_wins = get_asset_pnl("oracle-xrp", window_hours)
or_total_pnl = or_btc_pnl + or_eth_pnl + or_sol_pnl + or_xrp_pnl
or_total_trades = or_btc_trades + or_eth_trades + or_sol_trades + or_xrp_trades

# Ensemble bots (paper)
ens_btc_pnl, ens_btc_trades, ens_btc_wins = get_asset_pnl("ensemble-btc", window_hours)
ens_eth_pnl, ens_eth_trades, ens_eth_wins = get_asset_pnl("ensemble-eth", window_hours)
ens_sol_pnl, ens_sol_trades, ens_sol_wins = get_asset_pnl("ensemble-sol", window_hours)
ens_xrp_pnl, ens_xrp_trades, ens_xrp_wins = get_asset_pnl("ensemble-xrp", window_hours)
ens_total_pnl = ens_btc_pnl + ens_eth_pnl + ens_sol_pnl + ens_xrp_pnl
ens_total_trades = ens_btc_trades + ens_eth_trades + ens_sol_trades + ens_xrp_trades

# Optimal bot (paper, BTC only for now)
opt_btc_pnl, opt_btc_trades, opt_btc_wins = get_asset_pnl("optimal-btc", window_hours)

total_pnl_all = btc_pnl + eth_pnl + sol_pnl + xrp_pnl
total_trades_all = btc_trades + eth_trades + sol_trades + xrp_trades
total_wins_all = btc_wins + eth_wins + sol_wins + xrp_wins

# Bucket9 totals
b9_total_pnl = b9_btc_pnl + b9_eth_pnl + b9_sol_pnl + b9_xrp_pnl
b9_total_trades = b9_btc_trades + b9_eth_trades + b9_sol_trades + b9_xrp_trades
b9_total_wins = b9_btc_wins + b9_eth_wins + b9_sol_wins + b9_xrp_wins

# Summary bar - All RL bots with actual trades learning
st.markdown(
    f"""
    <div style="background-color: #1e1e1e; padding: 10px 20px; border-radius: 10px; margin-bottom: 10px;">
        <span style="font-size: 16px; color: #888;">📊 All Bots (Actual Trades Era):</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if btc_pnl >= 0 else '#ff4444'};">BTC: ${btc_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if eth_pnl >= 0 else '#ff4444'};">ETH: ${eth_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if sol_pnl >= 0 else '#ff4444'};">SOL: ${sol_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if xrp_pnl >= 0 else '#ff4444'};">XRP: ${xrp_pnl:+.2f}</span>
        <span style="margin-left: 25px; font-size: 18px; font-weight: bold; color: {'#00ff88' if total_pnl_all >= 0 else '#ff4444'};">Total: ${total_pnl_all:+.2f}</span>
        <span style="margin-left: 10px; color: #888;">({total_trades_all} trades)</span>
    </div>
    <div style="background-color: #2a2a1e; padding: 8px 20px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #444400;">
        <span style="font-size: 14px; color: #888;">🎯 Bucket9 Maximizer (Experimental):</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if b9_btc_pnl >= 0 else '#ff4444'};">B9-BTC: ${b9_btc_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if b9_eth_pnl >= 0 else '#ff4444'};">B9-ETH: ${b9_eth_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if b9_sol_pnl >= 0 else '#ff4444'};">B9-SOL: ${b9_sol_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if b9_xrp_pnl >= 0 else '#ff4444'};">B9-XRP: ${b9_xrp_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-size: 16px; font-weight: bold; color: {'#00ff88' if b9_total_pnl >= 0 else '#ff4444'};">Total: ${b9_total_pnl:+.2f}</span>
        <span style="margin-left: 10px; color: #888;">({b9_total_trades} trades)</span>
    </div>
    <div style="background-color: #1e2a2e; padding: 8px 20px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #004444;">
        <span style="font-size: 14px; color: #888;">Oracle (Black-Scholes):</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if or_btc_pnl >= 0 else '#ff4444'};">OR-BTC: ${or_btc_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if or_eth_pnl >= 0 else '#ff4444'};">OR-ETH: ${or_eth_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if or_sol_pnl >= 0 else '#ff4444'};">OR-SOL: ${or_sol_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if or_xrp_pnl >= 0 else '#ff4444'};">OR-XRP: ${or_xrp_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-size: 16px; font-weight: bold; color: {'#00ff88' if or_total_pnl >= 0 else '#ff4444'};">Total: ${or_total_pnl:+.2f}</span>
        <span style="margin-left: 10px; color: #888;">({or_total_trades} trades)</span>
    </div>
    <div style="background-color: #1e1e2e; padding: 8px 20px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #440088;">
        <span style="font-size: 14px; color: #888;">🤝 Ensemble (Paper — consensus voting):</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if ens_btc_pnl >= 0 else '#ff4444'};">ENS-BTC: ${ens_btc_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if ens_eth_pnl >= 0 else '#ff4444'};">ENS-ETH: ${ens_eth_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if ens_sol_pnl >= 0 else '#ff4444'};">ENS-SOL: ${ens_sol_pnl:+.2f}</span>
        <span style="margin-left: 10px; font-weight: bold; color: {'#00ff88' if ens_xrp_pnl >= 0 else '#ff4444'};">ENS-XRP: ${ens_xrp_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-size: 16px; font-weight: bold; color: {'#00ff88' if ens_total_pnl >= 0 else '#ff4444'};">Total: ${ens_total_pnl:+.2f}</span>
        <span style="margin-left: 10px; color: #888;">({ens_total_trades} trades)</span>
    </div>
    <div style="background-color: #1e2a1e; padding: 8px 20px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #004400;">
        <span style="font-size: 14px; color: #888;">⭐ Optimal Policy (Paper — frozen best model):</span>
        <span style="margin-left: 15px; font-weight: bold; color: {'#00ff88' if opt_btc_pnl >= 0 else '#ff4444'};">OPT-BTC: ${opt_btc_pnl:+.2f}</span>
        <span style="margin-left: 15px; font-size: 16px; font-weight: bold; color: {'#00ff88' if opt_btc_pnl >= 0 else '#ff4444'};">Total: ${opt_btc_pnl:+.2f}</span>
        <span style="margin-left: 10px; color: #888;">({opt_btc_trades} trades)</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Load data
trades_df = load_trades(db_path=f"data/{asset_lower}/trades.db", hours=window_hours)
metrics_df = load_metrics(db_path=f"data/{asset_lower}/metrics.db", hours=window_hours)

# Filter to Actual Trades era if sync enabled
if sync_with_actual_trades and not trades_df.empty:
    trades_df['timestamp_dt'] = pd.to_datetime(trades_df['opened_at'])
    actual_start = pd.to_datetime(ACTUAL_TRADES_START_TIME)
    original_count = len(trades_df)
    trades_df = trades_df[trades_df['timestamp_dt'] >= actual_start]
    filtered_count = len(trades_df)
    if original_count > filtered_count:
        st.info(f"📊 Showing {filtered_count} trades since Actual Trades era ({original_count - filtered_count} older trades filtered)")

# Bucket9 bots: Filter to show only trades after asset-specific optimization
if asset_lower.startswith("bucket9-") and not trades_df.empty:
    if 'timestamp_dt' not in trades_df.columns:
        trades_df['timestamp_dt'] = pd.to_datetime(trades_df['opened_at'])
    # SOL uses earlier cutoff (both YES+NO work), others use NO-only cutoff
    if asset_lower == "bucket9-sol":
        opt_start = pd.to_datetime(BUCKET9_SOL_START)
        strategy_note = "YES+NO directional"
    else:
        opt_start = pd.to_datetime(BUCKET9_NO_ONLY_START)
        strategy_note = "NO-only"
    original_count = len(trades_df)
    trades_df = trades_df[trades_df['timestamp_dt'] >= opt_start]
    filtered_count = len(trades_df)
    if original_count > filtered_count:
        st.info(f"🎯 {asset}: Showing {filtered_count} trades since {strategy_note} optimization ({original_count - filtered_count} pre-optimization trades filtered)")

# ============================================================================
# KEY METRICS ROW
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

if not trades_df.empty:
    total_pnl = trades_df['pnl'].sum()
    num_trades = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = wins / num_trades if num_trades > 0 else 0

    col1.metric("Total PnL", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
    col2.metric("Trades", num_trades)
    col3.metric("Win Rate", f"{win_rate:.1%}", delta=f"{(win_rate-0.5)*100:.1f}%")
    col4.metric("Wins / Losses", f"{wins} / {num_trades - wins}")
else:
    col1.metric("Total PnL", "$0.00")
    col2.metric("Trades", 0)
    col3.metric("Win Rate", "N/A")
    col4.metric("Wins / Losses", "0 / 0")

st.markdown("---")

# ============================================================================
# CUMULATIVE PNL CHART
# ============================================================================

st.subheader("📈 Cumulative PnL")

if not trades_df.empty:
    # Ensure timestamp_dt exists (may already be set by sync filter)
    if 'timestamp_dt' not in trades_df.columns:
        trades_df['timestamp_dt'] = pd.to_datetime(trades_df['opened_at'])
    trades_df = trades_df.sort_values('timestamp_dt')
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df['timestamp_dt'],
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative PnL',
        line=dict(color='#00D9FF', width=2),
        marker=dict(size=6)
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="PnL ($)",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )

    st.plotly_chart(fig, width="stretch")
else:
    st.info("No trades in selected time window")

# ============================================================================
# TWO-COLUMN LAYOUT
# ============================================================================

col_left, col_right = st.columns(2)

# ============================================================================
# LEFT COLUMN: Action Distribution
# ============================================================================

with col_left:
    st.subheader("🎯 Action Distribution")

    if not trades_df.empty:
        action_counts = trades_df['side'].value_counts()

        fig_actions = go.Figure(data=[go.Pie(
            labels=action_counts.index,
            values=action_counts.values,
            hole=.3,
            marker=dict(colors=['#00D9FF', '#FF6B6B'])
        )])

        fig_actions.update_layout(
            template='plotly_dark',
            height=300,
            showlegend=True
        )

        st.plotly_chart(fig_actions, use_container_width=True)
    else:
        st.info("No action data")

# ============================================================================
# RIGHT COLUMN: Risk Metrics
# ============================================================================

with col_right:
    st.subheader("⚠️ Risk Metrics")

    if not trades_df.empty:
        # Calculate drawdown
        trades_df['cumulative'] = trades_df['pnl'].cumsum()
        trades_df['peak'] = trades_df['cumulative'].cummax()
        trades_df['drawdown'] = trades_df['peak'] - trades_df['cumulative']

        max_dd = trades_df['drawdown'].max()

        # Consecutive losses
        losses = (trades_df['pnl'] < 0).astype(int)
        consecutive = (losses * (losses.groupby((losses != losses.shift()).cumsum()).cumcount() + 1))
        max_consecutive = consecutive.max()

        # Sharpe ratio (simplified)
        avg_pnl = trades_df['pnl'].mean()
        std_pnl = trades_df['pnl'].std()
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Max Drawdown", f"${max_dd:.2f}")
        metric_col1.metric("Sharpe Ratio", f"{sharpe:.2f}")

        metric_col2.metric("Max Consecutive Losses", int(max_consecutive))
        metric_col2.metric("Avg Trade PnL", f"${avg_pnl:.2f}")
    else:
        st.info("No risk data")

st.markdown("---")

# ============================================================================
# PNL DISTRIBUTION HISTOGRAM
# ============================================================================

st.subheader("📊 PnL Distribution")

if not trades_df.empty:
    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=trades_df['pnl'],
        nbinsx=20,
        marker=dict(
            color=trades_df['pnl'],
            colorscale='RdYlGn',
            showscale=False
        ),
        name='PnL'
    ))

    # Add vertical line at zero
    fig_dist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)

    fig_dist.update_layout(
        xaxis_title="PnL ($)",
        yaxis_title="Frequency",
        template='plotly_dark',
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("No PnL data")

st.markdown("---")

# ============================================================================
# ENTRY PRICE INTERVAL ANALYSIS
# ============================================================================

st.subheader("💰 Performance by Entry Price")

if not trades_df.empty:
    # Create price interval bins
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ['$0.00-0.10', '$0.10-0.20', '$0.20-0.30', '$0.30-0.40',
              '$0.40-0.50', '$0.50-0.60', '$0.60-0.70', '$0.70-0.80',
              '$0.80-0.90', '$0.90-1.00']

    trades_df['price_interval'] = pd.cut(trades_df['entry_price'], bins=bins, labels=labels, include_lowest=True)

    # Group by interval and calculate stats
    interval_stats = trades_df.groupby('price_interval', observed=True).agg({
        'pnl': ['count', 'sum', lambda x: (x > 0).sum(), lambda x: (x <= 0).sum()]
    }).round(2)

    interval_stats.columns = ['Trades', 'Total PnL', 'Wins', 'Losses']
    interval_stats['Win Rate'] = (interval_stats['Wins'] / interval_stats['Trades'] * 100).round(1)
    interval_stats['Avg PnL'] = (interval_stats['Total PnL'] / interval_stats['Trades']).round(2)

    # Format for display
    display_interval = interval_stats.copy()
    display_interval['Total PnL'] = display_interval['Total PnL'].apply(lambda x: f"${x:+.2f}")
    display_interval['Avg PnL'] = display_interval['Avg PnL'].apply(lambda x: f"${x:+.2f}")
    display_interval['Win Rate'] = display_interval['Win Rate'].apply(lambda x: f"{x:.0f}%")

    # Display as table
    st.dataframe(display_interval[['Trades', 'Wins', 'Losses', 'Win Rate', 'Total PnL', 'Avg PnL']],
                 use_container_width=True)

    # Bar chart visualization
    chart_data = trades_df.groupby('price_interval', observed=True)['pnl'].sum().reset_index()
    fig_intervals = px.bar(chart_data, x='price_interval', y='pnl',
                           title='PnL by Entry Price Interval',
                           color='pnl',
                           color_continuous_scale=['red', 'yellow', 'green'])
    fig_intervals.update_layout(
        template='plotly_dark',
        height=300,
        xaxis_title='Entry Price Interval',
        yaxis_title='Total PnL ($)'
    )
    st.plotly_chart(fig_intervals, use_container_width=True)
else:
    st.info("No trade data for interval analysis")

st.markdown("---")

# ============================================================================
# RECENT TRADES TABLE
# ============================================================================

st.subheader("📊 Recent Trades")

if not trades_df.empty:
    # Format for display
    display_df = trades_df[[
        'timestamp_dt', 'strike', 'side', 'entry_price', 'pnl', 'roi'
    ]].head(20).copy()

    display_df.columns = ['Time', 'Strike', 'Side', 'Entry', 'PnL', 'ROI']
    display_df['PnL'] = display_df['PnL'].apply(lambda x: f"${x:+.2f}")
    display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.1f}%")
    display_df['Entry'] = display_df['Entry'].apply(lambda x: f"${x:.3f}")

    st.dataframe(display_df, width="stretch", height=400)
else:
    st.info("No recent trades")

# ============================================================================
# RAW DATA (Optional)
# ============================================================================

if show_raw_data:
    st.markdown("---")
    st.subheader("🔍 Raw Data")

    if not trades_df.empty:
        with st.expander("Trades Data"):
            st.dataframe(trades_df)

    if not metrics_df.empty:
        with st.expander("Metrics Data"):
            st.dataframe(metrics_df)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Asset: {asset} | Model: {model_version}")

# ============================================================================
# AUTO REFRESH
# ============================================================================

if auto_refresh:
    time.sleep(30)
    st.rerun()
