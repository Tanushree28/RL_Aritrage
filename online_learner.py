#!/usr/bin/env python3
"""
Online Q-Learning from Actual Trades

Trains a Q-table ONLY from actual trades in trades.db, not synthetic experiences.
This provides clean signal from real execution outcomes.

Usage:
    python3 online_learner.py --asset eth --analyze    # Analyze current trades
    python3 online_learner.py --asset eth --train      # Train and save policy
    python3 online_learner.py --asset eth --backtest   # Backtest with filtering
    python3 online_learner.py --asset eth --loop       # Continuous online learning
"""

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional


# State discretization
PRICE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 10 buckets (0-9)
TIME_BUCKETS = [6, 12, 18, 24]  # 4 buckets: night(0-6), morning(6-12), afternoon(12-18), evening(18-24)


def get_price_bucket(price: float) -> int:
    """Convert price to bucket index (0-9)"""
    for i, threshold in enumerate(PRICE_BUCKETS):
        if price < threshold:
            return i
    return len(PRICE_BUCKETS)


def get_time_bucket(timestamp_str: str) -> int:
    """Convert timestamp to time-of-day bucket (0-3)"""
    try:
        # Parse ISO format timestamp
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        hour = dt.hour
        for i, threshold in enumerate(TIME_BUCKETS):
            if hour < threshold:
                return i
        return len(TIME_BUCKETS) - 1
    except:
        return 1  # Default to morning


def get_side_direction(side: str) -> int:
    """Convert side to direction (0=NO, 1=YES)"""
    return 1 if side.upper() == "YES" else 0


def extract_state(trade: dict) -> Tuple[int, int, int]:
    """
    Extract state tuple from trade record.

    State = (price_bucket, side_direction, time_bucket)
    - price_bucket: 0-9 based on entry price
    - side_direction: 0=NO, 1=YES
    - time_bucket: 0-3 based on time of day
    """
    price_bucket = get_price_bucket(trade['entry_price'])
    side_dir = get_side_direction(trade['side'])
    time_bucket = get_time_bucket(trade['opened_at'])

    return (price_bucket, side_dir, time_bucket)


class OnlineQLearner:
    """
    Online Q-learning agent that learns from actual trades only.

    Unlike HER, this only uses real trade outcomes - no counterfactuals.
    """

    def __init__(self, decay_factor: float = 0.995):
        # Q-table: state -> (sum_pnl, count, sum_squared for variance)
        self.q_table: Dict[Tuple, Dict] = defaultdict(lambda: {
            'sum_pnl': 0.0,
            'count': 0,
            'sum_squared': 0.0,
            'last_updated': time.time()
        })
        self.decay_factor = decay_factor
        self.total_trades_processed = 0

    def update(self, state: Tuple, pnl: float):
        """Update Q-value for state based on actual PnL outcome"""
        entry = self.q_table[state]

        # Apply decay to old observations (recency weighting)
        if entry['count'] > 0:
            age_seconds = time.time() - entry['last_updated']
            age_days = age_seconds / 86400
            decay = self.decay_factor ** age_days
            entry['sum_pnl'] *= decay
            entry['sum_squared'] *= decay
            entry['count'] = int(entry['count'] * decay)

        # Update with new observation
        entry['sum_pnl'] += pnl
        entry['sum_squared'] += pnl ** 2
        entry['count'] += 1
        entry['last_updated'] = time.time()

        self.total_trades_processed += 1

    def get_stats(self, state: Tuple) -> Optional[Dict]:
        """Get statistics for a state"""
        if state not in self.q_table:
            return None

        entry = self.q_table[state]
        if entry['count'] == 0:
            return None

        avg_pnl = entry['sum_pnl'] / entry['count']
        variance = (entry['sum_squared'] / entry['count']) - (avg_pnl ** 2)
        std_pnl = variance ** 0.5 if variance > 0 else 0

        return {
            'avg_pnl': avg_pnl,
            'count': entry['count'],
            'std_pnl': std_pnl,
            'confidence': 'high' if entry['count'] >= 10 else 'medium' if entry['count'] >= 5 else 'low'
        }

    def should_trade(self, state: Tuple, min_avg_pnl: float = 0.0, min_observations: int = 3) -> Tuple[bool, str]:
        """
        Decide if we should trade in this state based on historical performance.

        Returns (should_trade, reason)
        """
        stats = self.get_stats(state)

        if stats is None:
            return True, "no_data"  # Explore unknown states

        if stats['count'] < min_observations:
            return True, "insufficient_data"  # Not enough data yet

        if stats['avg_pnl'] < min_avg_pnl:
            return False, f"losing_state (avg=${stats['avg_pnl']:.2f})"

        return True, f"profitable (avg=${stats['avg_pnl']:.2f})"

    def save_policy(self, path: str):
        """Save policy in JSON format for Rust consumption"""
        policy = {
            'type': 'actual_trades_policy',
            'version': '1.0',
            'q_table': {},
            'thresholds': {
                'min_avg_pnl': 0.0,
                'min_observations': 3,
                'avoid_losing_states': True
            },
            'stats': {
                'total_trades_processed': self.total_trades_processed,
                'unique_states': len(self.q_table)
            },
            'timestamp': time.time()
        }

        for state, entry in self.q_table.items():
            if entry['count'] > 0:
                key = ','.join(map(str, state))
                avg_pnl = entry['sum_pnl'] / entry['count']
                policy['q_table'][key] = {
                    'avg_pnl': round(avg_pnl, 2),
                    'count': entry['count'],
                    'confidence': 'high' if entry['count'] >= 10 else 'medium' if entry['count'] >= 5 else 'low'
                }

        with open(path, 'w') as f:
            json.dump(policy, f, indent=2)

        print(f"[OnlineLearner] Saved policy to {path}")
        print(f"[OnlineLearner] {len(policy['q_table'])} states, {self.total_trades_processed} trades processed")

    def load_policy(self, path: str) -> bool:
        """Load existing policy to continue learning"""
        if not Path(path).exists():
            return False

        try:
            with open(path, 'r') as f:
                policy = json.load(f)

            for key, stats in policy.get('q_table', {}).items():
                state = tuple(map(int, key.split(',')))
                self.q_table[state] = {
                    'sum_pnl': stats['avg_pnl'] * stats['count'],
                    'count': stats['count'],
                    'sum_squared': (stats['avg_pnl'] ** 2) * stats['count'],  # Approximate
                    'last_updated': policy.get('timestamp', time.time())
                }

            self.total_trades_processed = policy.get('stats', {}).get('total_trades_processed', 0)
            print(f"[OnlineLearner] Loaded policy from {path} ({len(self.q_table)} states)")
            return True
        except Exception as e:
            print(f"[OnlineLearner] Failed to load policy: {e}")
            return False


def load_trades(db_path: str, limit: int = None, since: str = None) -> List[dict]:
    """Load trades from SQLite database, optionally filtered by date"""
    if not Path(db_path).exists():
        print(f"[OnlineLearner] Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if since:
        query = "SELECT * FROM trades WHERE opened_at >= ? ORDER BY id ASC"
        params = (since,)
    else:
        query = "SELECT * FROM trades ORDER BY id ASC"
        params = ()

    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def analyze_trades(trades: List[dict]):
    """Analyze trades and print summary statistics"""
    print(f"\n{'='*60}")
    print(f"Trade Analysis ({len(trades)} trades)")
    print(f"{'='*60}\n")

    # Group by state
    state_stats = defaultdict(lambda: {'pnls': [], 'wins': 0, 'count': 0})

    for trade in trades:
        state = extract_state(trade)
        pnl = trade['pnl']

        state_stats[state]['pnls'].append(pnl)
        state_stats[state]['count'] += 1
        if pnl > 0:
            state_stats[state]['wins'] += 1

    # Print by price bucket and side
    print("Performance by Price Bucket and Side:")
    print("-" * 70)
    print(f"{'Price Bucket':<15} {'Side':<6} {'Trades':>8} {'Win%':>8} {'Avg PnL':>10} {'Total PnL':>12} {'Verdict':<10}")
    print("-" * 70)

    # Aggregate by (price_bucket, side)
    agg_stats = defaultdict(lambda: {'pnls': [], 'count': 0, 'wins': 0})
    for state, stats in state_stats.items():
        price_bucket, side_dir, time_bucket = state
        key = (price_bucket, side_dir)
        agg_stats[key]['pnls'].extend(stats['pnls'])
        agg_stats[key]['count'] += stats['count']
        agg_stats[key]['wins'] += stats['wins']

    total_pnl = 0
    avoided_pnl = 0

    for (price_bucket, side_dir), stats in sorted(agg_stats.items()):
        side = "YES" if side_dir == 1 else "NO"
        price_range = f"${price_bucket/10:.2f}-{(price_bucket+1)/10:.2f}"
        count = stats['count']
        wins = stats['wins']
        win_pct = (wins / count * 100) if count > 0 else 0
        avg_pnl = sum(stats['pnls']) / count if count > 0 else 0
        total = sum(stats['pnls'])

        # Verdict based on avg PnL
        if avg_pnl < -10:
            verdict = "AVOID"
            avoided_pnl += total
        elif avg_pnl < 0:
            verdict = "WEAK"
            avoided_pnl += total
        elif avg_pnl < 5:
            verdict = "NEUTRAL"
        else:
            verdict = "GOOD"

        total_pnl += total

        print(f"{price_range:<15} {side:<6} {count:>8} {win_pct:>7.1f}% {avg_pnl:>+10.2f} {total:>+12.2f} {verdict:<10}")

    print("-" * 70)
    print(f"{'TOTAL':<15} {'':<6} {len(trades):>8} {'':<8} {'':<10} {total_pnl:>+12.2f}")
    print(f"\nPotential improvement from avoiding losing states: ${-avoided_pnl:+.2f}")


def backtest(trades: List[dict], learner: OnlineQLearner, min_avg_pnl: float = 0.0):
    """Backtest: what if we had used the learned policy to filter trades?"""
    print(f"\n{'='*60}")
    print(f"Backtest Results (min_avg_pnl threshold: ${min_avg_pnl:.2f})")
    print(f"{'='*60}\n")

    # First pass: learn from all trades
    for trade in trades:
        state = extract_state(trade)
        learner.update(state, trade['pnl'])

    # Second pass: simulate filtering
    original_pnl = 0
    filtered_pnl = 0
    trades_taken = 0
    trades_avoided = 0
    avoided_details = defaultdict(lambda: {'count': 0, 'pnl': 0})

    # Reset and replay with filtering
    for trade in trades:
        state = extract_state(trade)
        pnl = trade['pnl']
        original_pnl += pnl

        should_trade, reason = learner.should_trade(state, min_avg_pnl=min_avg_pnl)

        if should_trade:
            filtered_pnl += pnl
            trades_taken += 1
        else:
            trades_avoided += 1
            avoided_details[state]['count'] += 1
            avoided_details[state]['pnl'] += pnl

    print(f"Original trades: {len(trades)}, PnL: ${original_pnl:+.2f}")
    print(f"Filtered trades: {trades_taken}, PnL: ${filtered_pnl:+.2f}")
    print(f"Trades avoided:  {trades_avoided}")
    print(f"Improvement:     ${filtered_pnl - original_pnl:+.2f}")
    print(f"\nAvoided trades by state:")

    for state, details in sorted(avoided_details.items(), key=lambda x: x[1]['pnl']):
        price_bucket, side_dir, time_bucket = state
        side = "YES" if side_dir == 1 else "NO"
        stats = learner.get_stats(state)
        print(f"  ({price_bucket}, {side}, t={time_bucket}): "
              f"{details['count']} trades, ${details['pnl']:+.2f} avoided "
              f"(avg=${stats['avg_pnl']:.2f} from {stats['count']} samples)")


def train(asset: str, save_path: str = None, since: str = None):
    """Train Q-table from trades and save policy"""
    db_path = f"data/{asset}/trades.db"
    trades = load_trades(db_path, since=since)

    if not trades:
        print(f"[OnlineLearner] No trades found in {db_path}" + (f" since {since}" if since else ""))
        return

    if since:
        print(f"[OnlineLearner] Filtering trades since {since}")

    learner = OnlineQLearner()

    # Train on all trades
    for trade in trades:
        state = extract_state(trade)
        learner.update(state, trade['pnl'])

    # Save policy
    if save_path is None:
        save_path = f"model/{asset}/rl_policy_actual.json"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    learner.save_policy(save_path)

    # Print summary
    print(f"\n[OnlineLearner] Training complete!")
    print(f"[OnlineLearner] Processed {len(trades)} trades")
    print(f"[OnlineLearner] Learned {len(learner.q_table)} unique states")


def continuous_loop(asset: str, interval_seconds: int = 300):
    """Continuous online learning loop"""
    db_path = f"data/{asset}/trades.db"
    policy_path = f"model/{asset}/rl_policy_actual.json"

    learner = OnlineQLearner()
    learner.load_policy(policy_path)

    last_trade_id = 0

    print(f"[OnlineLearner] Starting continuous learning loop (interval={interval_seconds}s)")

    while True:
        try:
            # Load new trades since last check
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                "SELECT * FROM trades WHERE id > ? ORDER BY id ASC",
                (last_trade_id,)
            ).fetchall()
            conn.close()

            if rows:
                for row in rows:
                    trade = dict(row)
                    state = extract_state(trade)
                    learner.update(state, trade['pnl'])
                    last_trade_id = trade['id']

                    # Log update
                    stats = learner.get_stats(state)
                    print(f"[OnlineLearner] Updated state {state}: "
                          f"pnl=${trade['pnl']:+.2f}, avg=${stats['avg_pnl']:.2f} (n={stats['count']})")

                # Save updated policy
                learner.save_policy(policy_path)

            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n[OnlineLearner] Stopping continuous learning")
            learner.save_policy(policy_path)
            break
        except Exception as e:
            print(f"[OnlineLearner] Error: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Online Q-Learning from Actual Trades")
    parser.add_argument('--asset', type=str, default='eth', help='Asset to train (btc, eth, sol)')
    parser.add_argument('--analyze', action='store_true', help='Analyze trades and print summary')
    parser.add_argument('--train', action='store_true', help='Train and save policy')
    parser.add_argument('--backtest', action='store_true', help='Backtest with filtering')
    parser.add_argument('--loop', action='store_true', help='Continuous online learning')
    parser.add_argument('--interval', type=int, default=300, help='Loop interval in seconds')
    parser.add_argument('--threshold', type=float, default=0.0, help='Min avg PnL threshold for backtest')
    parser.add_argument('--since', type=str, default=None, help='Only use trades since this date (ISO format, e.g. 2026-02-09)')

    args = parser.parse_args()

    db_path = f"data/{args.asset}/trades.db"

    if args.analyze:
        trades = load_trades(db_path, since=args.since)
        if trades:
            analyze_trades(trades)

    elif args.train:
        train(args.asset, since=args.since)

    elif args.backtest:
        trades = load_trades(db_path, since=args.since)
        if trades:
            learner = OnlineQLearner()
            backtest(trades, learner, min_avg_pnl=args.threshold)

    elif args.loop:
        continuous_loop(args.asset, args.interval)

    else:
        # Default: analyze
        trades = load_trades(db_path, since=args.since)
        if trades:
            analyze_trades(trades)


if __name__ == "__main__":
    main()
