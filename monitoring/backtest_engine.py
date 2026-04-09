#!/usr/bin/env python3
"""
Backtest Engine — Parameterized strategies that return DataFrames.

Used by:
  - backtest_runner.py (cron job, pre-computes results)
  - dashboard.py (reads pre-computed results)

Strategies:
  1. Spread Compression
  2. Momentum Chase
  3. Rolling Average Arbitrage
"""

import os
import csv
import math
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Tick:
    timestamp_ms: int
    seconds_to_expiry: int
    strike: float
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    spot_price: float
    rolling_60s: float
    model_prob: float

    @property
    def mid_price(self) -> float:
        return (self.yes_ask + self.yes_bid) / 2

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def spread_pct(self) -> float:
        return self.spread * 100


@dataclass
class BacktestTrade:
    strategy: str
    ticker: str
    date: str
    timestamp_ms: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    edge: float
    exit_reason: str
    seconds_to_expiry: int
    entry_spread_pct: float
    hour_of_day: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ASSET_TICKER_PATTERNS = {
    "btc": "KXBTC15M",
    "eth": "KXETH15M",
    "sol": "KXSOL15M",
    "xrp": "KXXRP15M",
}


def get_recordings_dir(asset: str) -> str:
    """Get the recordings directory for an asset."""
    return os.path.join("data", asset, "recordings")


def get_available_dates(asset: str) -> List[str]:
    """Get list of available recording dates for an asset."""
    rec_dir = get_recordings_dir(asset)
    if not os.path.exists(rec_dir):
        return []
    dates = sorted([
        d for d in os.listdir(rec_dir)
        if os.path.isdir(os.path.join(rec_dir, d)) and d.startswith("202")
    ])
    return dates


def load_recordings_for_date(asset: str, date_str: str) -> Dict[str, List[Tick]]:
    """Load all CSV files for a given asset and date."""
    rec_dir = get_recordings_dir(asset)
    date_dir = os.path.join(rec_dir, date_str)
    if not os.path.exists(date_dir):
        return {}

    prefix = ASSET_TICKER_PATTERNS.get(asset, "")
    data = {}

    for filename in os.listdir(date_dir):
        if not filename.endswith(".csv"):
            continue
        if prefix and not filename.startswith(prefix):
            continue

        ticker = filename.replace(".csv", "")
        filepath = os.path.join(date_dir, filename)

        ticks = []
        try:
            with open(filepath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tick = Tick(
                            timestamp_ms=int(row["timestamp_unix_ms"]),
                            seconds_to_expiry=int(row["seconds_to_expiry"]),
                            strike=float(row["strike_price"]),
                            yes_ask=float(row["kalshi_yes_ask"]),
                            yes_bid=float(row["kalshi_yes_bid"]),
                            no_ask=float(row["kalshi_no_ask"]),
                            no_bid=float(row["kalshi_no_bid"]),
                            spot_price=float(row.get("synth_index_spot", 0) or 0),
                            rolling_60s=float(row.get("synth_index_60s", 0) or 0),
                            model_prob=float(row.get("model_probability", 0) or 0),
                        )
                        if tick.spot_price > 0 and tick.yes_ask > 0:
                            ticks.append(tick)
                    except (KeyError, ValueError):
                        continue
        except Exception:
            continue

        if ticks:
            data[ticker] = sorted(ticks, key=lambda t: t.timestamp_ms)

    return data


# ---------------------------------------------------------------------------
# Strategy 1: Spread Compression
# ---------------------------------------------------------------------------

def strategy_spread_compression(
    data: Dict[str, List[Tick]],
    date_str: str,
    min_spread_pct: float = 30.0,
    min_itm_pct: float = 0.03,
    min_secs: int = 300,
    max_secs: int = 3600,
    exit_spread_pct: float = 10.0,
) -> List[BacktestTrade]:
    """
    Buy deep ITM/OTM contracts at mid-price when spreads are wide.
    Exit when spread compresses or at expiry.
    """
    trades = []

    for ticker, ticks in data.items():
        if len(ticks) < 10:
            continue

        for i, tick in enumerate(ticks):
            if tick.seconds_to_expiry < min_secs or tick.seconds_to_expiry > max_secs:
                continue

            # Deep ITM/OTM check
            itm_pct = (tick.spot_price - tick.strike) / tick.strike
            is_deep_itm = itm_pct > min_itm_pct
            is_deep_otm = itm_pct < -min_itm_pct

            if not (is_deep_itm or is_deep_otm):
                continue

            if tick.spread_pct < min_spread_pct:
                continue

            # Entry
            side = "yes" if is_deep_itm else "no"
            entry_price = tick.mid_price if side == "yes" else (1 - tick.mid_price)

            # Find exit
            exit_tick = ticks[-1]
            exit_reason = "expiry"
            for j in range(i + 1, len(ticks)):
                if ticks[j].spread_pct < exit_spread_pct:
                    exit_tick = ticks[j]
                    exit_reason = "spread_compressed"
                    break

            # Settlement
            if exit_reason == "expiry":
                won = (exit_tick.rolling_60s > tick.strike) == (side == "yes")
                exit_price = 1.0 if won else 0.0
            else:
                exit_price = exit_tick.mid_price if side == "yes" else (1 - exit_tick.mid_price)

            pnl = exit_price - entry_price
            hour = datetime.utcfromtimestamp(tick.timestamp_ms / 1000).hour

            trades.append(BacktestTrade(
                strategy="Spread Compression",
                ticker=ticker,
                date=date_str,
                timestamp_ms=tick.timestamp_ms,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                edge=tick.spread_pct / 2,
                exit_reason=exit_reason,
                seconds_to_expiry=tick.seconds_to_expiry,
                entry_spread_pct=tick.spread_pct,
                hour_of_day=hour,
            ))
            break  # One trade per ticker

    return trades


# ---------------------------------------------------------------------------
# Strategy 2: Momentum Chase
# ---------------------------------------------------------------------------

def strategy_momentum_chase(
    data: Dict[str, List[Tick]],
    date_str: str,
    cross_threshold_pct: float = 0.3,
    prob_diff_threshold: float = 0.10,
    min_secs: int = 120,
    max_secs: int = 600,
) -> List[BacktestTrade]:
    """
    Trade when price crosses strike decisively but market hasn't repriced.
    """
    trades = []

    for ticker, ticks in data.items():
        if len(ticks) < 20:
            continue

        prev_tick = None
        in_trade = False
        entry_tick = None
        entry_price = 0
        entry_side = ""
        true_prob = 0

        for i, tick in enumerate(ticks):
            if prev_tick is None:
                prev_tick = tick
                continue

            if in_trade:
                market_caught_up = abs(tick.mid_price - true_prob) < 0.05
                near_expiry = tick.seconds_to_expiry < 60

                if market_caught_up or near_expiry:
                    exit_price = tick.mid_price if entry_side == "yes" else (1 - tick.mid_price)
                    pnl = exit_price - entry_price
                    hour = datetime.utcfromtimestamp(entry_tick.timestamp_ms / 1000).hour
                    exit_reason = "market_caught_up" if market_caught_up else "near_expiry"

                    trades.append(BacktestTrade(
                        strategy="Momentum Chase",
                        ticker=ticker,
                        date=date_str,
                        timestamp_ms=entry_tick.timestamp_ms,
                        side=entry_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        edge=abs(true_prob - tick.mid_price) * 100,
                        exit_reason=exit_reason,
                        seconds_to_expiry=entry_tick.seconds_to_expiry,
                        entry_spread_pct=entry_tick.spread_pct,
                        hour_of_day=hour,
                    ))
                    in_trade = False

                prev_tick = tick
                continue

            # Entry checks
            if tick.seconds_to_expiry < min_secs or tick.seconds_to_expiry > max_secs:
                prev_tick = tick
                continue

            prev_above = prev_tick.spot_price > tick.strike
            curr_above = tick.spot_price > tick.strike

            if prev_above == curr_above:
                prev_tick = tick
                continue

            move_pct = abs(tick.spot_price - prev_tick.spot_price) / tick.strike
            if move_pct < (cross_threshold_pct / 100):
                prev_tick = tick
                continue

            # Calculate true probability from rolling avg
            if tick.rolling_60s > tick.strike:
                true_prob = min(0.95, 0.5 + (tick.rolling_60s - tick.strike) / tick.strike * 10)
            else:
                true_prob = max(0.05, 0.5 - (tick.strike - tick.rolling_60s) / tick.strike * 10)

            market_prob = tick.mid_price
            prob_diff = abs(true_prob - market_prob)

            if prob_diff < prob_diff_threshold:
                prev_tick = tick
                continue

            in_trade = True
            entry_tick = tick
            entry_side = "yes" if true_prob > market_prob else "no"
            entry_price = tick.yes_ask if entry_side == "yes" else tick.no_ask

            prev_tick = tick

    return trades


# ---------------------------------------------------------------------------
# Strategy 3: Rolling Average Arbitrage
# ---------------------------------------------------------------------------

def strategy_rolling_avg_arb(
    data: Dict[str, List[Tick]],
    date_str: str,
    divergence_threshold: float = 0.12,
    min_secs: int = 30,
    max_secs: int = 180,
) -> List[BacktestTrade]:
    """
    Trade when spot differs significantly from rolling average.
    Settlement uses 60s rolling avg, not spot. When they diverge, there's mispricing.
    """
    trades = []

    def to_prob(dist: float) -> float:
        k = 20
        return 1 / (1 + math.exp(-k * dist))

    for ticker, ticks in data.items():
        if len(ticks) < 10:
            continue

        for i, tick in enumerate(ticks):
            if tick.seconds_to_expiry > max_secs or tick.seconds_to_expiry < min_secs:
                continue

            spot_vs_strike = (tick.spot_price - tick.strike) / tick.strike
            avg_vs_strike = (tick.rolling_60s - tick.strike) / tick.strike

            true_prob = to_prob(avg_vs_strike)
            market_prob = tick.mid_price

            arb_edge = abs(true_prob - market_prob)

            if arb_edge < divergence_threshold:
                continue

            side = "yes" if true_prob > market_prob else "no"
            entry_price = tick.yes_ask if side == "yes" else tick.no_ask

            # Simulate to expiry
            exit_tick = ticks[-1]
            won = (exit_tick.rolling_60s > tick.strike) == (side == "yes")
            exit_price = 1.0 if won else 0.0

            pnl = exit_price - entry_price
            hour = datetime.utcfromtimestamp(tick.timestamp_ms / 1000).hour

            trades.append(BacktestTrade(
                strategy="Rolling Avg Arb",
                ticker=ticker,
                date=date_str,
                timestamp_ms=tick.timestamp_ms,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                edge=arb_edge * 100,
                exit_reason="settled",
                seconds_to_expiry=tick.seconds_to_expiry,
                entry_spread_pct=tick.spread_pct,
                hour_of_day=hour,
            ))
            break  # One trade per ticker

    return trades


# ---------------------------------------------------------------------------
# Run all strategies for a single date
# ---------------------------------------------------------------------------

ALL_STRATEGIES = {
    "Spread Compression": strategy_spread_compression,
    "Momentum Chase": strategy_momentum_chase,
    "Rolling Avg Arb": strategy_rolling_avg_arb,
}


def run_all_strategies(asset: str, date_str: str) -> List[BacktestTrade]:
    """Run all strategies on a single date and return combined trades."""
    data = load_recordings_for_date(asset, date_str)
    if not data:
        return []

    all_trades = []
    for name, strategy_fn in ALL_STRATEGIES.items():
        trades = strategy_fn(data, date_str)
        all_trades.extend(trades)

    return all_trades


# ---------------------------------------------------------------------------
# Trades to DataFrame
# ---------------------------------------------------------------------------

def trades_to_df(trades: List[BacktestTrade]) -> pd.DataFrame:
    """Convert list of BacktestTrade to a DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])


# ---------------------------------------------------------------------------
# Pattern Detection
# ---------------------------------------------------------------------------

def detect_patterns(df: pd.DataFrame) -> Dict:
    """Analyze trades DataFrame for patterns. Returns dict of results."""
    if df.empty:
        return {}

    patterns = {}

    # --- Best Hours ---
    hourly = df.groupby("hour_of_day").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl=("pnl", "mean"),
    ).round(4)
    patterns["hourly_performance"] = hourly.to_dict("index")

    best_hour = hourly["total_pnl"].idxmax() if not hourly.empty else None
    worst_hour = hourly["total_pnl"].idxmin() if not hourly.empty else None
    patterns["best_hour"] = int(best_hour) if best_hour is not None else None
    patterns["worst_hour"] = int(worst_hour) if worst_hour is not None else None

    # --- Best Days of Week ---
    df_copy = df.copy()
    df_copy["datetime"] = pd.to_datetime(df_copy["timestamp_ms"], unit="ms")
    df_copy["day_of_week"] = df_copy["datetime"].dt.day_name()

    daily_dow = df_copy.groupby("day_of_week").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).round(4)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_dow = daily_dow.reindex([d for d in day_order if d in daily_dow.index])
    patterns["dow_performance"] = daily_dow.to_dict("index")

    # --- Strategy Comparison ---
    by_strategy = df.groupby("strategy").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl=("pnl", "mean"),
        max_win=("pnl", "max"),
        max_loss=("pnl", "min"),
    ).round(4)
    patterns["strategy_comparison"] = by_strategy.to_dict("index")

    # --- Streak Analysis ---
    win_streaks = []
    loss_streaks = []
    current_streak = 0
    current_type = None

    for _, row in df.sort_values("timestamp_ms").iterrows():
        is_win = row["pnl"] > 0
        if current_type is None:
            current_type = is_win
            current_streak = 1
        elif current_type == is_win:
            current_streak += 1
        else:
            if current_type:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_type = is_win
            current_streak = 1

    if current_type is not None:
        if current_type:
            win_streaks.append(current_streak)
        else:
            loss_streaks.append(current_streak)

    patterns["max_win_streak"] = max(win_streaks) if win_streaks else 0
    patterns["max_loss_streak"] = max(loss_streaks) if loss_streaks else 0

    # --- Entry Price Performance ---
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ['0-10¢', '10-20¢', '20-30¢', '30-40¢', '40-50¢',
              '50-60¢', '60-70¢', '70-80¢', '80-90¢', '90-$1']
    df_copy["price_bucket"] = pd.cut(df_copy["entry_price"], bins=bins, labels=labels, include_lowest=True)
    price_perf = df_copy.groupby("price_bucket", observed=True).agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).round(4)
    patterns["price_performance"] = price_perf.to_dict("index")

    # --- Side Performance ---
    by_side = df.groupby("side").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).round(4)
    patterns["side_performance"] = by_side.to_dict("index")

    return patterns


# ---------------------------------------------------------------------------
# Database helpers (for pre-computed results)
# ---------------------------------------------------------------------------

def get_db_path(asset: str) -> str:
    """Get path to backtest results database."""
    return os.path.join("data", asset, "backtest_results.db")


def init_db(asset: str):
    """Initialize backtest results database."""
    db_path = get_db_path(asset)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT,
            strategy        TEXT,
            ticker          TEXT,
            timestamp_ms    INTEGER,
            side            TEXT,
            entry_price     REAL,
            exit_price      REAL,
            pnl             REAL,
            edge            REAL,
            exit_reason     TEXT,
            seconds_to_expiry INTEGER,
            entry_spread_pct REAL,
            hour_of_day     INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            date            TEXT,
            strategy        TEXT,
            total_trades    INTEGER,
            wins            INTEGER,
            total_pnl       REAL,
            avg_pnl         REAL,
            sharpe          REAL,
            PRIMARY KEY (date, strategy)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            asset           TEXT,
            pattern_type    TEXT,
            details_json    TEXT,
            updated_at      TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_dates (
            date            TEXT PRIMARY KEY,
            processed_at    TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_processed_dates(asset: str) -> set:
    """Get set of already processed dates."""
    db_path = get_db_path(asset)
    if not os.path.exists(db_path):
        return set()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT date FROM processed_dates").fetchall()
        return {r[0] for r in rows}
    except:
        return set()
    finally:
        conn.close()


def save_trades_to_db(asset: str, trades: List[BacktestTrade], date_str: str):
    """Save backtest trades to the database."""
    db_path = get_db_path(asset)
    conn = sqlite3.connect(db_path)

    for t in trades:
        conn.execute("""
            INSERT INTO backtest_trades
            (date, strategy, ticker, timestamp_ms, side, entry_price, exit_price,
             pnl, edge, exit_reason, seconds_to_expiry, entry_spread_pct, hour_of_day)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (t.date, t.strategy, t.ticker, t.timestamp_ms, t.side,
              t.entry_price, t.exit_price, t.pnl, t.edge, t.exit_reason,
              t.seconds_to_expiry, t.entry_spread_pct, t.hour_of_day))

    # Compute daily summaries per strategy
    df = trades_to_df(trades)
    if not df.empty:
        for strategy, group in df.groupby("strategy"):
            total_trades = len(group)
            wins = int((group["pnl"] > 0).sum())
            total_pnl = float(group["pnl"].sum())
            avg_pnl = float(group["pnl"].mean())
            std_pnl = float(group["pnl"].std()) if len(group) > 1 else 0
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

            conn.execute("""
                INSERT OR REPLACE INTO daily_summaries
                (date, strategy, total_trades, wins, total_pnl, avg_pnl, sharpe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (date_str, strategy, total_trades, wins, total_pnl, avg_pnl, sharpe))

    # Mark as processed
    conn.execute("""
        INSERT OR REPLACE INTO processed_dates (date, processed_at)
        VALUES (?, ?)
    """, (date_str, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


def save_patterns_to_db(asset: str, patterns: Dict):
    """Save pattern detection results to DB."""
    db_path = get_db_path(asset)
    conn = sqlite3.connect(db_path)

    # Clear old patterns
    conn.execute("DELETE FROM patterns WHERE asset = ?", (asset,))

    for pattern_type, details in patterns.items():
        conn.execute("""
            INSERT INTO patterns (asset, pattern_type, details_json, updated_at)
            VALUES (?, ?, ?, ?)
        """, (asset, pattern_type, json.dumps(details, default=str),
              datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


def load_all_backtest_trades(asset: str) -> pd.DataFrame:
    """Load all backtest trades from DB for the dashboard."""
    db_path = get_db_path(asset)
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM backtest_trades ORDER BY timestamp_ms", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()


def load_backtest_trades_for_date(asset: str, date_str: str) -> pd.DataFrame:
    """Load backtest trades for a specific date."""
    db_path = get_db_path(asset)
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM backtest_trades WHERE date = ? ORDER BY timestamp_ms",
            conn, params=(date_str,))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()


def load_daily_summaries(asset: str) -> pd.DataFrame:
    """Load daily summary data."""
    db_path = get_db_path(asset)
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM daily_summaries ORDER BY date", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()


def load_patterns(asset: str) -> Dict:
    """Load pre-computed patterns from DB."""
    db_path = get_db_path(asset)
    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT pattern_type, details_json FROM patterns WHERE asset = ?",
            (asset,)).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}
    except:
        return {}
    finally:
        conn.close()


def load_processed_dates_list(asset: str) -> List[str]:
    """Load the list of processed dates for a backtest asset."""
    db_path = get_db_path(asset)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT date FROM processed_dates ORDER BY date").fetchall()
        return [r[0] for r in rows]
    except:
        return []
    finally:
        conn.close()
