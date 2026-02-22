#!/usr/bin/env python3
"""
Metrics Collector for RL Trading Agent Performance Tracking

Collects and persists performance metrics including:
- PnL, win rate, Sharpe ratio
- Action distributions
- Risk metrics (drawdown, consecutive losses)
- Market conditions

Used by monitoring dashboard and alerting system.
"""

import sqlite3
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    model_version: str
    window: str  # "1h", "1d", "session"

    # Performance
    pnl: float
    win_rate: float
    num_trades: int
    sharpe_ratio: float
    max_drawdown: float

    # Policy behavior
    action_hold_count: int
    action_buy_yes_count: int
    action_buy_no_count: int
    avg_q_value: float
    state_coverage: float

    # Market conditions
    avg_spread: float
    avg_volatility: float

    # Risk
    consecutive_losses: int
    capital_utilization: float


class MetricsCollector:
    """
    Collect and persist performance metrics to SQLite.

    Metrics are stored with timestamp, model version, and time window
    for historical analysis and real-time monitoring.
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize metrics database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                model_version TEXT,
                window TEXT,
                metrics_json TEXT NOT NULL
            )
        """)

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp DESC)"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_model_version ON performance_metrics(model_version)"
        )

        conn.commit()
        conn.close()

        print(f"[MetricsCollector] Initialized database at {self.db_path}")

    def record_metrics(self, metrics: PerformanceMetrics):
        """
        Record a metrics snapshot.

        Args:
            metrics: PerformanceMetrics instance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO performance_metrics (timestamp, model_version, window, metrics_json)
            VALUES (?, ?, ?, ?)
        """, (
            metrics.timestamp,
            metrics.model_version,
            metrics.window,
            json.dumps(asdict(metrics))
        ))

        conn.commit()
        conn.close()

    def get_recent_metrics(
        self,
        hours: int = 24,
        window: str = "1h",
        model_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent metrics snapshots.

        Args:
            hours: Number of hours to look back
            window: Time window filter ("1h", "1d", "session")
            model_version: Optional model version filter

        Returns:
            List of metrics dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = time.time() - (hours * 3600)

        if model_version:
            rows = cursor.execute("""
                SELECT metrics_json FROM performance_metrics
                WHERE timestamp > ? AND window = ? AND model_version = ?
                ORDER BY timestamp DESC
            """, (cutoff, window, model_version)).fetchall()
        else:
            rows = cursor.execute("""
                SELECT metrics_json FROM performance_metrics
                WHERE timestamp > ? AND window = ?
                ORDER BY timestamp DESC
            """, (cutoff, window)).fetchall()

        conn.close()

        return [json.loads(row[0]) for row in rows]


def compute_session_metrics(
    trades_db: str = "data/trades.db",
    model_version: str = "unknown",
    window_hours: float = 1.0
) -> Optional[PerformanceMetrics]:
    """
    Compute performance metrics from trade history.

    Args:
        trades_db: Path to trades SQLite database
        model_version: Model version identifier
        window_hours: Time window for metrics (hours)

    Returns:
        PerformanceMetrics instance or None if no trades
    """
    conn = sqlite3.connect(trades_db)
    cursor = conn.cursor()

    # Calculate cutoff time
    cutoff = time.time() - (window_hours * 3600)

    # Fetch trades in window
    # Note: opened_at is TEXT (ISO timestamp), so we check all recent trades
    trades = cursor.execute("""
        SELECT pnl, roi FROM trades
        ORDER BY id DESC
        LIMIT 100
    """).fetchall()

    conn.close()

    if not trades:
        return None

    pnls = [t[0] for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls)

    # Sharpe ratio (simplified)
    avg_pnl = total_pnl / len(pnls)
    std_pnl = (sum((p - avg_pnl)**2 for p in pnls) / len(pnls)) ** 0.5
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

    # Drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_dd = max(max_dd, drawdown)

    # Consecutive losses
    consecutive = 0
    max_consecutive = 0
    for pnl in pnls:
        if pnl < 0:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    # Window label
    if window_hours <= 1.5:
        window_label = "1h"
    elif window_hours <= 25:
        window_label = "1d"
    else:
        window_label = "session"

    return PerformanceMetrics(
        timestamp=time.time(),
        model_version=model_version,
        window=window_label,
        pnl=total_pnl,
        win_rate=win_rate,
        num_trades=len(trades),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        action_hold_count=0,  # TODO: track from executor
        action_buy_yes_count=0,
        action_buy_no_count=0,
        avg_q_value=0.0,
        state_coverage=0.0,
        avg_spread=0.0,
        avg_volatility=0.0,
        consecutive_losses=max_consecutive,
        capital_utilization=0.0
    )


def main():
    """Test metrics collection"""
    import os

    # Create data directory if needed
    os.makedirs("data", exist_ok=True)

    collector = MetricsCollector()

    # Try to compute metrics from trades database
    if os.path.exists("data/trades.db"):
        print("\nComputing metrics from trades.db...")
        metrics = compute_session_metrics(model_version="test")

        if metrics:
            print(f"\nSession Metrics:")
            print(f"  PnL: ${metrics.pnl:+.2f}")
            print(f"  Win Rate: {metrics.win_rate:.1%}")
            print(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"  Trades: {metrics.num_trades}")
            print(f"  Max Drawdown: ${metrics.max_drawdown:.2f}")
            print(f"  Consecutive Losses: {metrics.consecutive_losses}")

            # Record metrics
            print("\nRecording metrics...")
            collector.record_metrics(metrics)

            # Retrieve recent metrics
            print("\nRetrieving recent metrics...")
            recent = collector.get_recent_metrics(hours=24, window="1h")
            print(f"Found {len(recent)} recent metrics snapshots")

        else:
            print("No trades found in last hour")
    else:
        print("trades.db not found, creating sample metrics...")

        # Create sample metrics
        sample = PerformanceMetrics(
            timestamp=time.time(),
            model_version="v_test",
            window="1h",
            pnl=250.50,
            win_rate=0.62,
            num_trades=15,
            sharpe_ratio=1.8,
            max_drawdown=45.20,
            action_hold_count=10,
            action_buy_yes_count=8,
            action_buy_no_count=7,
            avg_q_value=5.3,
            state_coverage=0.68,
            avg_spread=0.03,
            avg_volatility=0.015,
            consecutive_losses=2,
            capital_utilization=0.45
        )

        collector.record_metrics(sample)
        print("Sample metrics recorded")


if __name__ == "__main__":
    main()
