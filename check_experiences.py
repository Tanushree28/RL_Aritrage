#!/usr/bin/env python3
"""
Check all experiences databases for data quality issues.
Verifies: correct 15M contract series, reward distribution, duplicates, missing data.
"""

import sqlite3
import os
import re
from collections import defaultdict

# All experience databases
DB_PATHS = [
    "data/btc/experiences.db",
    "data/eth/experiences.db",
    "data/sol/experiences.db",
    "data/xrp/experiences.db",
    "data/ensemble-btc/experiences.db",
    "data/ensemble-eth/experiences.db",
    "data/ensemble-sol/experiences.db",
    "data/ensemble-xrp/experiences.db",
    "data/optimal-btc/experiences.db",
]

# Expected 15M series per asset
EXPECTED_SERIES = {
    "btc":          "KXBTC15M",
    "eth":          "KXETH15M",
    "sol":          "KXSOL15M",
    "xrp":          "KXXRP15M",
    "ensemble-btc": "KXBTC15M",
    "ensemble-eth": "KXETH15M",
    "ensemble-sol": "KXSOL15M",
    "ensemble-xrp": "KXXRP15M",
    "optimal-btc":  "KXBTC15M",
}

def check_db(db_path):
    asset = db_path.split("/")[1]  # e.g. "btc", "ensemble-btc"
    expected = EXPECTED_SERIES.get(asset, "UNKNOWN")

    if not os.path.exists(db_path):
        print(f"  ⚠️  FILE NOT FOUND: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'")
    if not cur.fetchone():
        print(f"  ❌  No 'experiences' table in {db_path}")
        conn.close()
        return

    # Total count
    cur.execute("SELECT COUNT(*) FROM experiences")
    total = cur.fetchone()[0]

    if total == 0:
        print(f"  ⚠️  EMPTY: {db_path}")
        conn.close()
        return

    # Ticker series breakdown
    cur.execute("SELECT ticker, COUNT(*) FROM experiences GROUP BY ticker ORDER BY COUNT(*) DESC")
    ticker_rows = cur.fetchall()

    series_counts = defaultdict(int)
    for ticker, count in ticker_rows:
        # Extract series prefix (e.g. "KXBTC15M" from "KXBTC15M-26MAR260045-45")
        series = ticker.split("-")[0] if ticker else "UNKNOWN"
        series_counts[series] += count

    correct = series_counts.get(expected, 0)
    wrong   = total - correct
    pct_ok  = correct / total * 100 if total > 0 else 0

    # Reward distribution
    cur.execute("""
        SELECT
            MIN(reward), MAX(reward), AVG(reward),
            SUM(CASE WHEN reward > 50  THEN 1 ELSE 0 END),
            SUM(CASE WHEN reward < -50 THEN 1 ELSE 0 END),
            SUM(CASE WHEN reward = 0   THEN 1 ELSE 0 END),
            SUM(CASE WHEN reward > 0   THEN 1 ELSE 0 END),
            SUM(CASE WHEN reward < 0   THEN 1 ELSE 0 END)
        FROM experiences
    """)
    min_r, max_r, avg_r, over50, under_neg50, zeros, positives, negatives = cur.fetchone()

    # Date range
    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM experiences")
    min_ts, max_ts = cur.fetchone()
    import datetime
    min_date = datetime.datetime.fromtimestamp(min_ts).strftime('%b %d %Y') if min_ts else "?"
    max_date = datetime.datetime.fromtimestamp(max_ts).strftime('%b %d %Y') if max_ts else "?"

    conn.close()

    # Print report
    status = "✅" if wrong == 0 else "❌"
    print(f"\n{'─'*60}")
    print(f"  {status}  {db_path}")
    print(f"{'─'*60}")
    print(f"  Total experiences : {total:,}")
    print(f"  Date range        : {min_date} → {max_date}")
    print(f"  Expected series   : {expected}")
    print(f"  Correct 15M       : {correct:,} ({pct_ok:.1f}%)")
    if wrong > 0:
        print(f"  ❌ WRONG SERIES  : {wrong:,} experiences from wrong contract!")
        for series, count in series_counts.items():
            if series != expected:
                print(f"       → {series}: {count:,}")
    print(f"  Reward range      : ${min_r:.2f} to ${max_r:.2f}  (avg ${avg_r:.2f})")
    print(f"  Zeros (HOLD)      : {zeros:,}")
    print(f"  Positive rewards  : {positives:,}")
    print(f"  Negative rewards  : {negatives:,}")
    if over50 > 0 or under_neg50 > 0:
        pct_noisy = (over50 + under_neg50) / total * 100
        print(f"  ⚠️  Noisy rewards  : {over50 + under_neg50:,} outside ±$50 ({pct_noisy:.1f}%) — from pre-clip era")
        print(f"       → Over +$50  : {over50:,}")
        print(f"       → Under -$50 : {under_neg50:,}")


def main():
    print("=" * 60)
    print("  EXPERIENCES DATABASE AUDIT")
    print("  Checking 15M contract validity + reward quality")
    print("=" * 60)

    for db_path in DB_PATHS:
        check_db(db_path)

    print(f"\n{'='*60}")
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
