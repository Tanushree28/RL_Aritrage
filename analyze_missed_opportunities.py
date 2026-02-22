#!/usr/bin/env python3
"""
Analyze Today's Markets for Missed Opportunities

For each market that has settled:
1. Determine the settlement outcome (YES or NO won)
2. Look at what prices were available during the trading window
3. Calculate potential profit if we had bought the winning side
4. Identify why the bot might have missed the trade
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DATA_DIR = "data/recordings/2026-02-06"

# Strategy parameters (match current bot settings)
MIN_EDGE = 0.05              # 5% minimum edge
DYNAMIC_MIN_EDGE_HIGH = 0.10 # 10% at 5 min remaining
DYNAMIC_MIN_EDGE_LOW = 0.025 # 2.5% at 30s remaining
BET_SIZE = 100.0

def load_market(filepath):
    """Load market data and extract key info."""
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        if len(df) < 5:
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp_unix_ms'], unit='ms')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_market(filepath):
    """Analyze a single market for missed opportunities."""
    ticker = os.path.basename(filepath).replace('.csv', '')
    df = load_market(filepath)
    
    if df is None or len(df) < 10:
        return None
    
    # Get market info
    strike = df['strike_price'].iloc[-1]
    if strike < 100:
        strike = df['strike_price'].max()
    if strike < 100:
        return None
    
    # Get final BTC price for settlement
    final_btc = df['coinbase_price'].iloc[-1]
    first_btc = df['coinbase_price'].iloc[0]
    
    # Determine outcome
    yes_won = final_btc > strike
    winning_side = "YES" if yes_won else "NO"
    
    # Analyze trading opportunities
    results = {
        'ticker': ticker,
        'strike': strike,
        'first_btc': first_btc,
        'final_btc': final_btc,
        'outcome': winning_side,
        'duration_secs': len(df),
        'opportunities': []
    }
    
    # Look for opportunities throughout the market
    for idx in range(0, len(df), 10):  # Sample every 10 rows
        row = df.iloc[idx]
        
        yes_ask = row['kalshi_yes_ask']
        no_ask = row['kalshi_no_ask']
        btc = row['coinbase_price']
        secs_left = row.get('seconds_to_expiry', 0)
        
        if secs_left <= 0:
            continue
        
        # Calculate what we would have paid vs what we would have won
        if yes_won:
            entry_price = yes_ask
            pnl_per_contract = 1.0 - entry_price
        else:
            entry_price = no_ask
            pnl_per_contract = 1.0 - entry_price
        
        contracts = int(BET_SIZE / entry_price) if entry_price > 0.01 else 0
        potential_pnl = pnl_per_contract * contracts
        
        # Calculate edge vs model prediction (assume model would predict correctly)
        # Edge = (1.0 - entry_price) since we know it wins
        edge = pnl_per_contract / entry_price if entry_price > 0 else 0
        
        # Calculate dynamic edge threshold
        if secs_left >= 300:
            min_edge = DYNAMIC_MIN_EDGE_HIGH
        elif secs_left <= 30:
            min_edge = DYNAMIC_MIN_EDGE_LOW
        else:
            # Linear interpolation
            t_factor = (secs_left - 30) / (300 - 30)
            min_edge = DYNAMIC_MIN_EDGE_LOW + t_factor * (DYNAMIC_MIN_EDGE_HIGH - DYNAMIC_MIN_EDGE_LOW)
        
        # Check if we would have passed the edge threshold
        would_have_traded = edge >= min_edge
        
        if potential_pnl > 5:  # Only significant opportunities
            results['opportunities'].append({
                'secs_left': secs_left,
                'btc_price': btc,
                'entry_price': entry_price,
                'edge': edge,
                'min_edge': min_edge,
                'would_trade': would_have_traded,
                'potential_pnl': potential_pnl,
            })
    
    return results

def main():
    print("=" * 80)
    print("MISSED OPPORTUNITY ANALYSIS - Today's Markets")
    print("=" * 80)
    
    # Get all KXBTC15M files from today
    files = sorted(glob.glob(os.path.join(DATA_DIR, "KXBTC15M-*.csv")))
    print(f"\nFound {len(files)} KXBTC15M markets\n")
    
    total_missed_pnl = 0
    total_would_trade_pnl = 0
    
    for filepath in files:
        result = analyze_market(filepath)
        if result is None:
            continue
        
        print(f"\n{'='*70}")
        print(f"Market: {result['ticker']}")
        print(f"Strike: ${result['strike']:,.0f}")
        print(f"BTC: ${result['first_btc']:,.0f} → ${result['final_btc']:,.0f}")
        print(f"Outcome: {result['outcome']} WON")
        print(f"Duration: {result['duration_secs']}s of data")
        
        if not result['opportunities']:
            print("  No significant opportunities found")
            continue
        
        print(f"\nOpportunities ({len(result['opportunities'])} snapshots with >$5 potential):")
        print("-" * 70)
        print(f"{'Secs Left':>10} | {'Entry':>8} | {'Edge':>8} | {'Min Edge':>8} | {'Would Trade':>12} | {'PnL':>10}")
        print("-" * 70)
        
        best_missed = None
        best_would_trade = None
        
        for opp in result['opportunities']:
            status = "✅ YES" if opp['would_trade'] else "❌ NO"
            print(f"{opp['secs_left']:>10} | ${opp['entry_price']:>6.2f} | {opp['edge']*100:>7.1f}% | {opp['min_edge']*100:>7.1f}% | {status:>12} | ${opp['potential_pnl']:>8.2f}")
            
            if not opp['would_trade']:
                if best_missed is None or opp['potential_pnl'] > best_missed['potential_pnl']:
                    best_missed = opp
            else:
                if best_would_trade is None or opp['potential_pnl'] > best_would_trade['potential_pnl']:
                    best_would_trade = opp
        
        if best_missed:
            total_missed_pnl += best_missed['potential_pnl']
            print(f"\n⚠️  BEST MISSED: ${best_missed['potential_pnl']:.2f} at {best_missed['secs_left']}s (edge {best_missed['edge']*100:.1f}% < min {best_missed['min_edge']*100:.1f}%)")
        
        if best_would_trade:
            total_would_trade_pnl += best_would_trade['potential_pnl']
            print(f"✅ BEST TRADEABLE: ${best_would_trade['potential_pnl']:.2f} at {best_would_trade['secs_left']}s")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total PnL from tradeable opportunities: ${total_would_trade_pnl:.2f}")
    print(f"Total PnL from MISSED opportunities: ${total_missed_pnl:.2f}")
    print(f"\nRecommendation: If missed PnL is high, consider:")
    print("  1. Lowering minimum edge thresholds")
    print("  2. Extending trading window (earlier entry)")
    print("  3. Adding spread compression strategy entries")

if __name__ == "__main__":
    main()
