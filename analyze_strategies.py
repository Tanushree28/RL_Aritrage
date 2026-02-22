#!/usr/bin/env python3
"""
Strategy Analysis Script - Analyzes recorded Kalshi data for 3 trading strategies.

This script reads CSV files from the bot's recordings and simulates:
1. Spread Compression Strategy
2. Momentum Chase Strategy  
3. Rolling Average Arbitrage Strategy

Run independently from the main bot - does NOT interfere with live trading.

Usage:
    python3 analyze_strategies.py [--date YYYY-MM-DD] [--output report.md]
"""

import os
import csv
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")

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
class Trade:
    strategy: str
    ticker: str
    entry_time: int
    exit_time: int
    side: str  # "yes" or "no"
    entry_price: float
    exit_price: float
    pnl: float
    edge: float

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_recordings(date_str: str) -> Dict[str, List[Tick]]:
    """Load all CSV files for a given date."""
    date_dir = os.path.join(RECORDINGS_DIR, date_str)
    if not os.path.exists(date_dir):
        print(f"No recordings found for {date_str}")
        return {}
    
    data = {}
    for filename in os.listdir(date_dir):
        if not filename.endswith(".csv"):
            continue
        ticker = filename.replace(".csv", "")
        filepath = os.path.join(date_dir, filename)
        
        ticks = []
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
                        spot_price=float(row["synth_index_spot"]),
                        rolling_60s=float(row["synth_index_60s"]),
                        model_prob=float(row["model_probability"]),
                    )
                    ticks.append(tick)
                except (KeyError, ValueError) as e:
                    continue
        
        if ticks:
            data[ticker] = sorted(ticks, key=lambda t: t.timestamp_ms)
    
    return data

# ---------------------------------------------------------------------------
# Strategy 1: Spread Compression
# ---------------------------------------------------------------------------

def analyze_spread_compression(data: Dict[str, List[Tick]]) -> List[Trade]:
    """
    Strategy: Buy deep ITM contracts at mid-price when spreads are wide.
    
    Entry: When spread > 30% and contract is deep ITM (price > strike + 3%)
    Exit: Hold to expiry (simplified: use last tick as exit)
    """
    trades = []
    
    for ticker, ticks in data.items():
        if len(ticks) < 10:
            continue
            
        # Find entry opportunities
        for i, tick in enumerate(ticks):
            # Only enter when > 5 min to expiry
            if tick.seconds_to_expiry < 300 or tick.seconds_to_expiry > 3600:
                continue
            
            # Check if deep ITM (spot > strike + 3%)
            itm_pct = (tick.spot_price - tick.strike) / tick.strike
            is_deep_itm = itm_pct > 0.03
            is_deep_otm = itm_pct < -0.03
            
            if not (is_deep_itm or is_deep_otm):
                continue
            
            # Check spread width
            if tick.spread_pct < 30:
                continue
            
            # Entry at mid-price
            side = "yes" if is_deep_itm else "no"
            entry_price = tick.mid_price if side == "yes" else (1 - tick.mid_price)
            
            # Find exit (last tick or when spread compresses)
            exit_tick = ticks[-1]
            for j in range(i + 1, len(ticks)):
                if ticks[j].spread_pct < 10:
                    exit_tick = ticks[j]
                    break
            
            # Calculate PnL (assume fill at mid)
            if side == "yes":
                # Settled YES if spot > strike
                won = exit_tick.spot_price > tick.strike
                exit_price = 1.0 if won else 0.0
            else:
                won = exit_tick.spot_price < tick.strike
                exit_price = 1.0 if won else 0.0
            
            pnl = exit_price - entry_price
            edge = tick.spread_pct / 2  # Half the spread as theoretical edge
            
            trades.append(Trade(
                strategy="Spread Compression",
                ticker=ticker,
                entry_time=tick.timestamp_ms,
                exit_time=exit_tick.timestamp_ms,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                edge=edge,
            ))
            break  # Only one trade per ticker
    
    return trades

# ---------------------------------------------------------------------------
# Strategy 2: Momentum Chase
# ---------------------------------------------------------------------------

def analyze_momentum_chase(data: Dict[str, List[Tick]]) -> List[Trade]:
    """
    Strategy: Trade when price crosses strike and market hasn't repriced.
    
    Entry: When spot crosses strike decisively (>0.5%) but market prob differs by >15%
    Exit: When market catches up or at expiry
    """
    trades = []
    
    for ticker, ticks in data.items():
        if len(ticks) < 20:
            continue
        
        prev_tick = None
        in_trade = False
        entry_tick = None
        
        for i, tick in enumerate(ticks):
            if prev_tick is None:
                prev_tick = tick
                continue
            
            if in_trade:
                # Check exit conditions
                market_caught_up = abs(tick.mid_price - true_prob) < 0.05
                near_expiry = tick.seconds_to_expiry < 60
                
                if market_caught_up or near_expiry:
                    # Exit trade
                    won = (tick.spot_price > tick.strike) == (entry_side == "yes")
                    exit_price = tick.mid_price if entry_side == "yes" else (1 - tick.mid_price)
                    
                    trades.append(Trade(
                        strategy="Momentum Chase",
                        ticker=ticker,
                        entry_time=entry_tick.timestamp_ms,
                        exit_time=tick.timestamp_ms,
                        side=entry_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=exit_price - entry_price,
                        edge=abs(true_prob - market_prob) * 100,
                    ))
                    in_trade = False
                
                prev_tick = tick
                continue
            
            # Check entry conditions
            if tick.seconds_to_expiry < 120 or tick.seconds_to_expiry > 600:
                prev_tick = tick
                continue
            
            # Detect decisive cross
            prev_above = prev_tick.spot_price > tick.strike
            curr_above = tick.spot_price > tick.strike
            
            if prev_above == curr_above:
                prev_tick = tick
                continue
            
            # Price crossed strike - check if "decisive" (>0.3% move)
            move_pct = abs(tick.spot_price - prev_tick.spot_price) / tick.strike
            if move_pct < 0.003:
                prev_tick = tick
                continue
            
            # Calculate true probability from rolling avg
            if tick.rolling_60s > tick.strike:
                true_prob = min(0.95, 0.5 + (tick.rolling_60s - tick.strike) / tick.strike * 10)
            else:
                true_prob = max(0.05, 0.5 - (tick.strike - tick.rolling_60s) / tick.strike * 10)
            
            market_prob = tick.mid_price
            
            # Check for mispricing
            prob_diff = abs(true_prob - market_prob)
            if prob_diff < 0.10:
                prev_tick = tick
                continue
            
            # Enter trade
            in_trade = True
            entry_tick = tick
            entry_side = "yes" if true_prob > market_prob else "no"
            entry_price = tick.yes_ask if entry_side == "yes" else tick.no_ask
            
            prev_tick = tick
    
    return trades

# ---------------------------------------------------------------------------
# Strategy 3: Rolling Average Arbitrage
# ---------------------------------------------------------------------------

def analyze_rolling_avg_arb(data: Dict[str, List[Tick]]) -> List[Trade]:
    """
    Strategy: Trade when spot differs significantly from rolling average.
    
    Settlement uses 60s rolling avg, not spot. When they diverge, there's mispricing.
    """
    trades = []
    
    for ticker, ticks in data.items():
        if len(ticks) < 10:
            continue
        
        for i, tick in enumerate(ticks):
            # Only trade in final 3 minutes
            if tick.seconds_to_expiry > 180 or tick.seconds_to_expiry < 30:
                continue
            
            # Calculate divergence
            spot_vs_strike = (tick.spot_price - tick.strike) / tick.strike
            avg_vs_strike = (tick.rolling_60s - tick.strike) / tick.strike
            
            # Convert to probabilities (using sigmoid approximation)
            def to_prob(dist):
                k = 20
                import math
                return 1 / (1 + math.exp(-k * dist))
            
            true_prob = to_prob(avg_vs_strike)  # Based on settlement pricing
            spot_prob = to_prob(spot_vs_strike)  # Based on current spot
            
            # Market usually follows spot, not rolling avg
            market_prob = tick.mid_price
            
            # Check for arbitrage: market follows spot but settlement uses avg
            arb_edge = abs(true_prob - market_prob)
            
            if arb_edge < 0.12:
                continue
            
            # Entry
            side = "yes" if true_prob > market_prob else "no"
            entry_price = tick.yes_ask if side == "yes" else tick.no_ask
            
            # Simulate to expiry
            exit_tick = ticks[-1]
            won = (exit_tick.rolling_60s > tick.strike) == (side == "yes")
            exit_price = 1.0 if won else 0.0
            
            trades.append(Trade(
                strategy="Rolling Avg Arb",
                ticker=ticker,
                entry_time=tick.timestamp_ms,
                exit_time=exit_tick.timestamp_ms,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=exit_price - entry_price,
                edge=arb_edge * 100,
            ))
            break  # One trade per ticker
    
    return trades

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_trades: List[Trade], output_path: str, date_str: str):
    """Generate markdown report."""
    
    # Group by strategy
    by_strategy = defaultdict(list)
    for t in all_trades:
        by_strategy[t.strategy].append(t)
    
    lines = [
        f"# Strategy Analysis Report",
        f"",
        f"**Date:** {date_str}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Strategy | Trades | Wins | Win Rate | Total PnL | Avg Edge |",
        f"|----------|--------|------|----------|-----------|----------|",
    ]
    
    for strategy in ["Spread Compression", "Momentum Chase", "Rolling Avg Arb"]:
        trades = by_strategy.get(strategy, [])
        if not trades:
            lines.append(f"| {strategy} | 0 | - | - | - | - |")
            continue
        
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / len(trades) * 100 if trades else 0
        total_pnl = sum(t.pnl for t in trades)
        avg_edge = sum(t.edge for t in trades) / len(trades) if trades else 0
        
        lines.append(
            f"| {strategy} | {len(trades)} | {wins} | {win_rate:.1f}% | "
            f"${total_pnl*100:.2f} | {avg_edge:.1f}% |"
        )
    
    lines.extend([
        f"",
        f"---",
        f"",
    ])
    
    # Detail by strategy
    for strategy in ["Spread Compression", "Momentum Chase", "Rolling Avg Arb"]:
        trades = by_strategy.get(strategy, [])
        lines.extend([
            f"## {strategy}",
            f"",
        ])
        
        if not trades:
            lines.extend([f"No trades triggered.", f""])
            continue
        
        lines.extend([
            f"| Ticker | Side | Entry | Exit | PnL | Edge |",
            f"|--------|------|-------|------|-----|------|",
        ])
        
        for t in trades[:20]:  # Limit to 20 trades per strategy
            ticker_short = t.ticker.split("-")[-1] if "-" in t.ticker else t.ticker
            pnl_str = f"+${t.pnl*100:.0f}" if t.pnl > 0 else f"-${abs(t.pnl)*100:.0f}"
            lines.append(
                f"| {ticker_short[:12]} | {t.side.upper()} | ${t.entry_price:.2f} | "
                f"${t.exit_price:.2f} | {pnl_str} | {t.edge:.1f}% |"
            )
        
        lines.append(f"")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to: {output_path}")
    return lines

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze trading strategies from recorded data")
    parser.add_argument("--date", type=str, help="Date to analyze (YYYY-MM-DD)", default=None)
    parser.add_argument("--output", type=str, help="Output report path", default=None)
    args = parser.parse_args()
    
    # Default to today
    if args.date is None:
        # Try today and yesterday
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        if os.path.exists(os.path.join(RECORDINGS_DIR, today)):
            date_str = today
        elif os.path.exists(os.path.join(RECORDINGS_DIR, yesterday)):
            date_str = yesterday
        else:
            # Find any available date
            if os.path.exists(RECORDINGS_DIR):
                dates = sorted(os.listdir(RECORDINGS_DIR), reverse=True)
                date_str = dates[0] if dates else today
            else:
                date_str = today
    else:
        date_str = args.date
    
    output_path = args.output or os.path.join(PROJECT_ROOT, f"strategy_report_{date_str}.md")
    
    print(f"Loading recordings for {date_str}...")
    data = load_recordings(date_str)
    print(f"Loaded {len(data)} tickers")
    
    if not data:
        print("No data to analyze. Exiting.")
        return
    
    print("\nAnalyzing Strategy 1: Spread Compression...")
    trades_1 = analyze_spread_compression(data)
    print(f"  Found {len(trades_1)} potential trades")
    
    print("\nAnalyzing Strategy 2: Momentum Chase...")
    trades_2 = analyze_momentum_chase(data)
    print(f"  Found {len(trades_2)} potential trades")
    
    print("\nAnalyzing Strategy 3: Rolling Avg Arb...")
    trades_3 = analyze_rolling_avg_arb(data)
    print(f"  Found {len(trades_3)} potential trades")
    
    all_trades = trades_1 + trades_2 + trades_3
    print(f"\nTotal trades: {len(all_trades)}")
    
    print("\nGenerating report...")
    generate_report(all_trades, output_path, date_str)
    
    # Print summary to console
    total_pnl = sum(t.pnl for t in all_trades)
    wins = sum(1 for t in all_trades if t.pnl > 0)
    print(f"\n{'='*50}")
    print(f"SUMMARY: {len(all_trades)} trades, {wins} wins, ${total_pnl*100:.2f} PnL")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
