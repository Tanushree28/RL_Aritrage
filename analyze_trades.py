import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

import sys

# Database path
DB_PATH = 'data/trades.db'

def analyze_trades():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    # Parse days from command line, default to 2
    try:
        days = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    except ValueError:
        print("Invalid number of days. Using default: 2")
        days = 2

    conn = sqlite3.connect(DB_PATH)

    # Calculate timeframe
    # Using UTC as the timestamps in DB seem to be UTC (+00:00)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    start_time_str = start_time.isoformat()

    print(f"Analyzing trades from {start_time_str} to {end_time.isoformat()} (UTC) ({days} days)\n")

    # Read trades
    # We'll filter in pandas to handle timezone parsing robustly if needed, 
    # but SQL filtering is more efficient.
    # The format in DB is ISO8601, so string comparison works.
    query = f"SELECT * FROM trades WHERE opened_at >= '{start_time_str}'"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No trades found in the last 2 days.")
        return

    # Convert numeric columns
    df['entry_price'] = pd.to_numeric(df['entry_price'])
    df['pnl'] = pd.to_numeric(df['pnl'])

    # Helper to define buckets
    def get_bucket(price):
        # Floor to nearest 0.05
        lower = int(price / 0.05) * 0.05
        upper = lower + 0.05
        return f"{lower:.2f}-{upper:.2f}$"

    df['bucket'] = df['entry_price'].apply(get_bucket)

    # Sort buckets for display
    def sort_key(bucket_str):
        return float(bucket_str.split('-')[0])

    # Analysis per side
    for side in ['YES', 'NO']:
        print(f"=== Analysis for {side} ===")
        side_df = df[df['side'] == side]
        
        if side_df.empty:
            print(f"No {side} trades found.\n")
            continue
            
        # Group by bucket
        # aggregation: count trades, sum wins (pnl > 0), sum pnl
        stats = side_df.groupby('bucket').agg(
            Taken_Trades=('id', 'count'),
            Wins=('pnl', lambda x: (x > 0).sum()),
            Total_PnL=('pnl', 'sum')
        ).reset_index()

        # Sort by bucket price descending
        stats['sort_val'] = stats['bucket'].apply(sort_key)
        stats = stats.sort_values('sort_val', ascending=False).drop('sort_val', axis=1)

        # Print formatted
        for _, row in stats.iterrows():
            # Format: 0.95-1.00$ 5 Taken Trades 5 Wins 5$ PnL
            print(f"{row['bucket']} {row['Taken_Trades']} Taken Trades {row['Wins']} Wins {row['Total_PnL']:.2f}$ PnL")
        
        # Side totals
        total_trades = side_df.shape[0]
        total_wins = (side_df['pnl'] > 0).sum()
        total_pnl = side_df['pnl'].sum()
        print(f"\nTotal {side}: {total_trades} Trades, {total_wins} Wins, {total_pnl:.2f}$ PnL")
        print("-" * 30 + "\n")

    # Overall totals
    print("=== Overall Performance (Last 2 Days) ===")
    total_trades = df.shape[0]
    total_wins = (df['pnl'] > 0).sum()
    total_pnl = df['pnl'].sum()
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Total Wins:   {total_wins}")
    print(f"Win Rate:     {win_rate:.1f}%")
    print(f"Total PnL:    {total_pnl:.2f}$")

if __name__ == "__main__":
    analyze_trades()
