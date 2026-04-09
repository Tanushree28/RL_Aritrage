#!/usr/bin/env python3
"""
Backtest Runner — Cron job that pre-computes backtest results.

Runs periodically (e.g., every 2 days via cron) and processes any new recording
dates that haven't been backtested yet. Skips today since data is incomplete.

Usage:
    python3 backtest_runner.py              # Process all assets
    python3 backtest_runner.py --asset btc  # Process single asset
    python3 backtest_runner.py --force      # Reprocess all dates

Cron Example (every 2 days at 3 AM UTC):
    0 3 */2 * * cd ~/kalshi-arb && python3 backtest_runner.py >> logs/backtest.log 2>&1
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitoring.backtest_engine import (
    get_available_dates,
    get_processed_dates,
    init_db,
    run_all_strategies,
    save_trades_to_db,
    save_patterns_to_db,
    trades_to_df,
    detect_patterns,
    load_all_backtest_trades,
    get_db_path,
)

import sqlite3

ASSETS = ["btc", "eth", "sol", "xrp"]


def process_asset(asset: str, force: bool = False):
    """Process all unprocessed dates for an asset."""
    print(f"\n{'=' * 60}")
    print(f"Processing asset: {asset.upper()}")
    print(f"{'=' * 60}")

    # Initialize DB
    init_db(asset)

    # Get available and processed dates
    available = get_available_dates(asset)
    processed = get_processed_dates(asset) if not force else set()

    # Skip today (incomplete data)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    available = [d for d in available if d < today]

    # Find new dates to process
    new_dates = [d for d in available if d not in processed]

    if not new_dates:
        print(f"  No new dates to process (have {len(processed)} dates already)")
        return

    print(f"  Available dates: {len(available)}")
    print(f"  Already processed: {len(processed)}")
    print(f"  New dates to process: {len(new_dates)}")
    print(f"  Dates: {new_dates}")

    # Force: clear existing data
    if force:
        db_path = get_db_path(asset)
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM backtest_trades")
        conn.execute("DELETE FROM daily_summaries")
        conn.execute("DELETE FROM processed_dates")
        conn.execute("DELETE FROM patterns")
        conn.commit()
        conn.close()
        print("  Cleared existing data (force mode)")

    # Process each new date
    total_trades = 0
    for date_str in new_dates:
        print(f"\n  Processing {date_str}...", end=" ", flush=True)

        trades = run_all_strategies(asset, date_str)
        total_trades += len(trades)

        if trades:
            save_trades_to_db(asset, trades, date_str)
            print(f"{len(trades)} trades found")

            # Print strategy breakdown
            by_strat = {}
            for t in trades:
                by_strat.setdefault(t.strategy, []).append(t)
            for strat, strat_trades in by_strat.items():
                wins = sum(1 for t in strat_trades if t.pnl > 0)
                pnl = sum(t.pnl for t in strat_trades)
                print(f"    {strat}: {len(strat_trades)} trades, "
                      f"{wins} wins, PnL: ${pnl*100:.2f}")
        else:
            # Mark as processed even if no trades (so we don't re-process)
            save_trades_to_db(asset, [], date_str)
            print("0 trades")

    # Run pattern detection on ALL data
    print(f"\n  Running pattern detection across all dates...")
    all_trades_df = load_all_backtest_trades(asset)
    if not all_trades_df.empty:
        patterns = detect_patterns(all_trades_df)
        save_patterns_to_db(asset, patterns)

        # Print pattern highlights
        if "best_hour" in patterns and patterns["best_hour"] is not None:
            print(f"    Best trading hour: {patterns['best_hour']}:00 UTC")
        if "worst_hour" in patterns and patterns["worst_hour"] is not None:
            print(f"    Worst trading hour: {patterns['worst_hour']}:00 UTC")
        if "max_win_streak" in patterns:
            print(f"    Max win streak: {patterns['max_win_streak']}")
            print(f"    Max loss streak: {patterns['max_loss_streak']}")

    print(f"\n  Done! Total trades across all dates: {total_trades}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Runner")
    parser.add_argument("--asset", type=str, default=None,
                        help="Process single asset (btc/eth/sol/xrp)")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocess all dates")
    args = parser.parse_args()

    print(f"Backtest Runner — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if args.asset:
        if args.asset.lower() not in ASSETS:
            print(f"Unknown asset: {args.asset}. Valid: {ASSETS}")
            sys.exit(1)
        process_asset(args.asset.lower(), force=args.force)
    else:
        for asset in ASSETS:
            process_asset(asset, force=args.force)

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
