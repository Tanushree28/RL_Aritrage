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
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    ["BTC", "ETH", "SOL", "XRP", "Bucket9-BTC", "Bucket9-ETH", "Bucket9-SOL", "Bucket9-XRP"],
    index=0
)
# Map display name to data directory
asset_lower = {
    "BTC": "btc",
    "ETH": "eth",
    "SOL": "sol",
    "XRP": "xrp",
    "Bucket9-BTC": "bucket9-btc",
    "Bucket9-ETH": "bucket9-eth",
    "Bucket9-SOL": "bucket9-sol",
    "Bucket9-XRP": "bucket9-xrp"
}[asset]

# Sync start time option - now syncs with Actual Trades learning start
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
