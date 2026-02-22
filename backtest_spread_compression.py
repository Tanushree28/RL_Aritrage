#!/usr/bin/env python3
"""
Backtest Spread Compression Strategy on Historical Data

Strategy: Buy deep ITM/OTM contracts at mid-price when spreads are wide,
hold until spread compresses or contract expires.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Configuration - More relaxed for backtesting
# ---------------------------------------------------------------------------

CAPITAL = 10_000.0
BET_SIZE = 100.0
MIN_SPREAD_PCT = 15.0        # Entry threshold (relaxed from 25%)
MIN_ITM_OTM_FRAC = 0.01      # 1% distance from strike (relaxed from 2.5%)
MIN_SECS_TO_EXPIRY = 60      # 1 minute (relaxed from 5 min)
MAX_SECS_TO_EXPIRY = 900     # 15 minutes
EXIT_SPREAD_PCT = 5.0        # Exit when spread compresses

DATA_DIR = "data/recordings"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    strike: float
    side: str
    entry_price: float
    entry_spread_pct: float
    contracts: int
    entry_time: datetime
    entry_btc: float

@dataclass
class Trade:
    ticker: str
    side: str
    entry_price: float
    exit_price: float
    contracts: int
    pnl: float
    won: bool
    entry_time: datetime
    exit_time: datetime
    exit_reason: str

# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------

def load_market_data(csv_path: str) -> pd.DataFrame:
    """Load market data from a single CSV file."""
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        df['timestamp'] = pd.to_datetime(df['timestamp_unix_ms'], unit='ms')
        return df
    except Exception as e:
        return pd.DataFrame()

def extract_ticker_from_path(path: str) -> str:
    return os.path.basename(path).replace('.csv', '')

def get_all_market_files(date_dirs: List[str]) -> List[str]:
    files = []
    for date_dir in date_dirs:
        pattern = os.path.join(DATA_DIR, date_dir, "KXBTC15M-*.csv")
        files.extend(glob.glob(pattern))
    return sorted(files)

# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def backtest_market(df: pd.DataFrame, ticker: str) -> List[Trade]:
    """Backtest a single market's historical data."""
    trades = []
    position: Optional[Position] = None
    
    if df.empty or len(df) < 10:
        return trades
    
    # Get strike from the data
    strike_col = df['strike_price']
    non_zero = strike_col[strike_col > 100]
    if non_zero.empty:
        return trades
    strike = float(non_zero.iloc[0])
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        try:
            ts = row['timestamp']
            btc_price = float(row.get('coinbase_price', row.get('synth_index_spot', 0)))
            yes_ask = float(row['kalshi_yes_ask'])
            no_ask = float(row['kalshi_no_ask'])
            secs_left = int(row.get('seconds_to_expiry', 0))
        except:
            continue
        
        if btc_price <= 0 or yes_ask <= 0 or np.isnan(btc_price):
            continue
        
        # Calculate spread: yes_bid = 1 - no_ask
        yes_bid = max(0.01, 1.0 - no_ask)
        spread = yes_ask - yes_bid
        spread_pct = spread * 100
        mid_price = (yes_ask + yes_bid) / 2
        
        # --- Exit checks ---
        if position is not None:
            # Check expiry (settlement)
            if secs_left <= 0:
                won = (btc_price > position.strike) == (position.side == "yes")
                exit_price = 1.0 if won else 0.0
                pnl = (exit_price - position.entry_price) * position.contracts
                
                trades.append(Trade(
                    ticker=position.ticker,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    contracts=position.contracts,
                    pnl=pnl,
                    won=pnl > 0,
                    entry_time=position.entry_time,
                    exit_time=ts,
                    exit_reason="settled"
                ))
                position = None
                continue
            
            # Check spread compression
            if spread_pct < EXIT_SPREAD_PCT:
                exit_price = mid_price if position.side == "yes" else (1 - mid_price)
                pnl = (exit_price - position.entry_price) * position.contracts
                
                trades.append(Trade(
                    ticker=position.ticker,
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    contracts=position.contracts,
                    pnl=pnl,
                    won=pnl > 0,
                    entry_time=position.entry_time,
                    exit_time=ts,
                    exit_reason="spread_compressed"
                ))
                position = None
                continue
        
        # --- Entry checks ---
        if position is None:
            # Time constraints
            if secs_left < MIN_SECS_TO_EXPIRY or secs_left > MAX_SECS_TO_EXPIRY:
                continue
            
            # Spread constraint
            if spread_pct < MIN_SPREAD_PCT:
                continue
            
            # Deep ITM/OTM check
            distance_frac = (btc_price - strike) / strike
            
            if distance_frac > MIN_ITM_OTM_FRAC:
                side = "yes"
            elif distance_frac < -MIN_ITM_OTM_FRAC:
                side = "no"
            else:
                continue
            
            # Calculate entry
            entry_price = mid_price if side == "yes" else (1 - mid_price)
            if entry_price <= 0.01:
                continue
            contracts = max(1, int(BET_SIZE / entry_price))
            
            position = Position(
                ticker=ticker,
                strike=strike,
                side=side,
                entry_price=entry_price,
                entry_spread_pct=spread_pct,
                contracts=contracts,
                entry_time=ts,
                entry_btc=btc_price
            )
    
    # Handle position still open at end
    if position is not None:
        final_btc = float(df.iloc[-1].get('coinbase_price', df.iloc[-1].get('synth_index_spot', 0)))
        won = (final_btc > position.strike) == (position.side == "yes")
        exit_price = 1.0 if won else 0.0
        pnl = (exit_price - position.entry_price) * position.contracts
        
        trades.append(Trade(
            ticker=position.ticker,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            contracts=position.contracts,
            pnl=pnl,
            won=pnl > 0,
            entry_time=position.entry_time,
            exit_time=df.iloc[-1]['timestamp'],
            exit_reason="end_of_data"
        ))
    
    return trades

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SPREAD COMPRESSION STRATEGY BACKTEST")
    print("=" * 70)
    print(f"Capital: ${CAPITAL:,.0f}")
    print(f"Bet Size: ${BET_SIZE:.0f}")
    print(f"Min Spread: {MIN_SPREAD_PCT}%")
    print(f"Min ITM/OTM: {MIN_ITM_OTM_FRAC*100}%")
    print(f"Time Window: {MIN_SECS_TO_EXPIRY}s - {MAX_SECS_TO_EXPIRY}s")
    print(f"Exit Spread: {EXIT_SPREAD_PCT}%")
    print("=" * 70)
    
    date_dirs = sorted([d for d in os.listdir(DATA_DIR) 
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("2026")])
    
    print(f"\nFound data for dates: {date_dirs}")
    
    market_files = get_all_market_files(date_dirs)
    print(f"Found {len(market_files)} KXBTC15M market files")
    
    all_trades: List[Trade] = []
    markets_with_trades = 0
    skipped = 0
    
    for i, filepath in enumerate(market_files):
        ticker = extract_ticker_from_path(filepath)
        df = load_market_data(filepath)
        
        if df.empty:
            skipped += 1
            continue
        
        trades = backtest_market(df, ticker)
        if trades:
            all_trades.extend(trades)
            markets_with_trades += 1
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(market_files)} markets, {len(all_trades)} trades so far")
    
    print(f"\n{'=' * 70}")
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Skipped (empty/corrupt): {skipped}")
    
    if not all_trades:
        print("\nNo trades executed!")
        return
    
    # Calculate statistics
    total_pnl = sum(t.pnl for t in all_trades)
    wins = sum(1 for t in all_trades if t.won)
    losses = len(all_trades) - wins
    win_rate = wins / len(all_trades) * 100
    
    settled = [t for t in all_trades if t.exit_reason == "settled"]
    compressed = [t for t in all_trades if t.exit_reason == "spread_compressed"]
    
    avg_win = np.mean([t.pnl for t in all_trades if t.won]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in all_trades if not t.won]) if losses > 0 else 0
    
    print(f"\nTotal Trades: {len(all_trades)}")
    print(f"Markets with Trades: {markets_with_trades}")
    print(f"\nWins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"\nTotal PnL: ${total_pnl:,.2f}")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    
    print(f"\nExit Breakdown:")
    print(f"  Settled at expiry: {len(settled)}")
    print(f"  Spread compressed: {len(compressed)}")
    
    # Breakdown by side
    yes_trades = [t for t in all_trades if t.side == "yes"]
    no_trades = [t for t in all_trades if t.side == "no"]
    
    print(f"\nBy Side:")
    if yes_trades:
        yes_pnl = sum(t.pnl for t in yes_trades)
        yes_wins = sum(1 for t in yes_trades if t.won)
        print(f"  YES: {len(yes_trades)} trades, {yes_wins} wins ({yes_wins/len(yes_trades)*100:.0f}%), PnL: ${yes_pnl:.2f}")
    if no_trades:
        no_pnl = sum(t.pnl for t in no_trades)
        no_wins = sum(1 for t in no_trades if t.won)
        print(f"  NO: {len(no_trades)} trades, {no_wins} wins ({no_wins/len(no_trades)*100:.0f}%), PnL: ${no_pnl:.2f}")
    
    # ROI calculation
    roi = (total_pnl / CAPITAL) * 100
    print(f"\nROI: {roi:.1f}%")
    
    # Sample trades
    print(f"\n{'=' * 70}")
    print("SAMPLE TRADES")
    print("=" * 70)
    for trade in all_trades[:15]:
        emoji = "✅" if trade.won else "❌"
        print(f"{emoji} {trade.ticker} | {trade.side.upper()} | "
              f"Entry ${trade.entry_price:.2f} → Exit ${trade.exit_price:.2f} | "
              f"PnL: ${trade.pnl:.2f} | {trade.exit_reason}")
    
    if len(all_trades) > 15:
        print(f"... and {len(all_trades) - 15} more trades")

if __name__ == "__main__":
    main()
