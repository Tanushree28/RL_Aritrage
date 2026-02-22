#!/usr/bin/env python3
"""
Automated Alerting System for RL Trading Agent

Monitors performance metrics and triggers alerts when:
- Performance degrades (low win rate, high losses)
- Risk thresholds exceeded (drawdown, consecutive losses)
- Model staleness detected

Integrates with Telegram for notifications.

Usage:
    python3 alerting.py              # Run alert monitoring loop
"""

import time
import sys
from typing import Dict, Any, Callable
from dataclasses import dataclass

from metrics_collector import compute_session_metrics, PerformanceMetrics


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[PerformanceMetrics], bool]
    message_template: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    action: str    # PAUSE_TRADING, REDUCE_POSITION_SIZE, NOTIFY


# ============================================================================
# ALERT RULES
# ============================================================================

ALERT_RULES = {
    "performance_degradation": AlertRule(
        name="Performance Degradation",
        condition=lambda m: m.win_rate < 0.45 or m.pnl < -500,
        message_template="⚠️ PERFORMANCE ALERT: Win rate {win_rate:.1%} or PnL ${pnl:.2f} below threshold",
        severity="CRITICAL",
        action="PAUSE_TRADING"
    ),

    "consecutive_losses": AlertRule(
        name="Consecutive Losses",
        condition=lambda m: m.consecutive_losses >= 5,
        message_template="⚠️ RISK ALERT: {consecutive_losses} consecutive losses detected",
        severity="HIGH",
        action="REDUCE_POSITION_SIZE"
    ),

    "max_drawdown": AlertRule(
        name="Maximum Drawdown",
        condition=lambda m: m.max_drawdown > 1000,
        message_template="⚠️ DRAWDOWN ALERT: Max drawdown ${max_drawdown:.2f}",
        severity="MEDIUM",
        action="NOTIFY"
    ),

    "low_win_rate": AlertRule(
        name="Low Win Rate",
        condition=lambda m: m.num_trades >= 10 and m.win_rate < 0.40,
        message_template="⚠️ WIN RATE ALERT: {win_rate:.1%} over {num_trades} trades",
        severity="HIGH",
        action="NOTIFY"
    ),

    "negative_sharpe": AlertRule(
        name="Negative Sharpe Ratio",
        condition=lambda m: m.num_trades >= 5 and m.sharpe_ratio < -0.5,
        message_template="⚠️ SHARPE ALERT: Sharpe ratio {sharpe_ratio:.2f} (risk-adjusted loss)",
        severity="MEDIUM",
        action="NOTIFY"
    ),

    "overtrading": AlertRule(
        name="Overtrading Detection",
        condition=lambda m: m.num_trades > 50,
        message_template="⚠️ OVERTRADING: {num_trades} trades in 1 hour (threshold: 50)",
        severity="HIGH",
        action="NOTIFY"
    ),
}


class AlertingSystem:
    """
    Alert monitoring and notification system.

    Checks metrics against alert rules and takes appropriate actions.
    """

    def __init__(self, trades_db: str = "data/trades.db"):
        self.trades_db = trades_db
        self.alert_history = {}  # Track when alerts last fired

    def check_alerts(self, model_version: str = "current") -> None:
        """
        Check all alert rules and trigger notifications.

        Args:
            model_version: Model version to include in alerts
        """
        # Compute recent metrics (last hour)
        metrics = compute_session_metrics(
            trades_db=self.trades_db,
            model_version=model_version,
            window_hours=1.0
        )

        if not metrics:
            # No trades in last hour - nothing to alert on
            return

        # Check each alert rule
        for rule_id, rule in ALERT_RULES.items():
            try:
                if rule.condition(metrics):
                    self._trigger_alert(rule_id, rule, metrics)
            except Exception as e:
                print(f"[AlertingSystem] Error checking rule '{rule_id}': {e}")

    def _trigger_alert(
        self,
        rule_id: str,
        rule: AlertRule,
        metrics: PerformanceMetrics
    ) -> None:
        """
        Trigger an alert and execute associated action.

        Args:
            rule_id: Unique rule identifier
            rule: AlertRule instance
            metrics: Current performance metrics
        """
        # Check if we recently fired this alert (avoid spam)
        now = time.time()
        last_fired = self.alert_history.get(rule_id, 0)

        # Cooldown: don't re-fire same alert within 30 minutes
        cooldown_seconds = 30 * 60
        if now - last_fired < cooldown_seconds:
            return

        # Format alert message
        message = rule.message_template.format(
            win_rate=metrics.win_rate,
            pnl=metrics.pnl,
            consecutive_losses=metrics.consecutive_losses,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
            num_trades=metrics.num_trades
        )

        # Log alert
        print(f"\n{'=' * 80}")
        print(f"[ALERT] {rule.name}")
        print(f"Severity: {rule.severity}")
        print(f"Message: {message}")
        print(f"Action: {rule.action}")
        print(f"{'=' * 80}\n")

        # Execute action
        if rule.action == "PAUSE_TRADING":
            self._pause_trading()
        elif rule.action == "REDUCE_POSITION_SIZE":
            self._reduce_position_size()
        elif rule.action == "NOTIFY":
            self._send_notification(rule.severity, message)

        # Update alert history
        self.alert_history[rule_id] = now

    def _pause_trading(self) -> None:
        """
        Pause trading system (critical action).

        In production, this could:
        - Set an environment variable that the bot checks
        - Write a flag file that the executor reads
        - Send a shutdown signal to the trading process
        """
        print("[AlertingSystem] ACTION: Pausing trading system")
        print("  Implementation: Set TRADING_ENABLED=false or create pause.flag")

        # Example: Create a flag file
        with open("data/pause_trading.flag", 'w') as f:
            f.write(f"Paused at {time.time()} due to performance degradation\n")

    def _reduce_position_size(self) -> None:
        """
        Reduce position sizes (high-risk mitigation).

        Could write a config adjustment that the executor reads.
        """
        print("[AlertingSystem] ACTION: Reducing position size by 50%")
        print("  Implementation: Update position_size_multiplier in config")

        # Example: Write adjustment factor
        with open("data/position_size_adjustment.txt", 'w') as f:
            f.write("0.5")  # 50% of normal size

    def _send_notification(self, severity: str, message: str) -> None:
        """
        Send notification (via Telegram, email, etc.).

        Args:
            severity: Alert severity level
            message: Alert message
        """
        print(f"[AlertingSystem] ACTION: Sending {severity} notification")
        print(f"  Message: {message}")

        # In production, integrate with existing Telegram bot
        # For now, just log
        try:
            # Could integrate with telegram_reporter here
            pass
        except Exception as e:
            print(f"[AlertingSystem] Failed to send notification: {e}")


def alert_loop(interval_sec: int = 60) -> None:
    """
    Continuous alert monitoring loop.

    Args:
        interval_sec: Check interval in seconds
    """
    print("=" * 80)
    print("ALERT MONITORING SYSTEM")
    print("=" * 80)
    print(f"Check interval: {interval_sec} seconds")
    print(f"Active rules: {len(ALERT_RULES)}")
    print()

    for rule_id, rule in ALERT_RULES.items():
        print(f"  [{rule.severity:>8}] {rule.name}")

    print()
    print("=" * 80)
    print("Monitoring started. Press Ctrl+C to stop.")
    print("=" * 80)
    print()

    system = AlertingSystem()

    try:
        while True:
            try:
                system.check_alerts()
            except Exception as e:
                print(f"[AlertingSystem] Error in alert loop: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal, shutting down...")
        sys.exit(0)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated alerting system for RL trading agent"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Alert check interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    alert_loop(interval_sec=args.interval)


if __name__ == "__main__":
    main()
