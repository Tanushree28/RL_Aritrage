#!/usr/bin/env python3
"""
Reinforcement Learning Trading Strategy

This strategy uses Q-learning to learn optimal trading decisions from historical data.
It rewards winning trades and punishes losing trades to maximize profit.

State Space:
  - distance_pct: How far BTC price is from strike (0-5%)
  - time_bucket: Time remaining category (0-5)
  - price_bucket: Entry price category (0-9)
  - momentum: Price movement direction

Action Space:
  - 0: HOLD (no trade)
  - 1: BUY_YES
  - 2: BUY_NO

Reward:
  - WIN: +PnL (positive reinforcement)
  - LOSS: -|PnL| (punishment)
  - HOLD: 0 (neutral)
"""

import os
import glob
import pickle
import json
import time
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional, List

# Q-Learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99  # Faster decay per episode
EPSILON_WARMSTART = 0.15  # Low epsilon when fine-tuning existing model

# State discretization - MUST match Rust code in trade_executor.rs
DISTANCE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5]  # percent (trades only when < 0.5%)
TIME_BUCKETS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]  # seconds (last 5 minutes only)
PRICE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # dollars

# Actions
HOLD = 0
BUY_YES = 1
BUY_NO = 2

# Asset-specific ticker patterns
ASSET_TICKER_PATTERNS = {
    "btc": "KXBTC15M-*.csv",
    "eth": "KXETH15M-*.csv",
    "sol": "KXSOL15M-*.csv",
}


@dataclass
class MarketSnapshot:
    """A single point-in-time market observation."""
    timestamp: float
    btc_price: float
    strike: float
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    secs_left: int
    momentum: float = 0.0  # % price change over last 30 data points
    
    @property
    def distance_pct(self) -> float:
        return abs(self.btc_price - self.strike) / self.strike * 100
    
    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid
    
    @property
    def is_above_strike(self) -> bool:
        return self.btc_price > self.strike


class QLearningAgent:
    """Q-Learning agent for trading decisions."""
    
    def __init__(self, load_from: Optional[str] = None):
        self.q_table = defaultdict(lambda: np.zeros(3))  # 3 actions
        self.epsilon = EPSILON_START
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        
        if load_from and os.path.exists(load_from):
            self.load(load_from)
    
    def discretize_state(self, snapshot: MarketSnapshot) -> Tuple:
        """Convert continuous state to discrete buckets. MUST match Rust code."""
        
        # Distance bucket
        dist = snapshot.distance_pct
        dist_bucket = len([b for b in DISTANCE_BUCKETS if dist >= b])
        
        # Time bucket
        time_bucket = len([b for b in TIME_BUCKETS if snapshot.secs_left <= b])
        
        # Yes price bucket
        yes_bucket = len([b for b in PRICE_BUCKETS if snapshot.yes_ask >= b])
        
        # Above/below strike
        direction = 1 if snapshot.is_above_strike else 0
        
        # Spread bucket (0=tight, 1=wide)
        spread_bucket = 1 if snapshot.spread > 0.05 else 0
        
        # Momentum bucket: 0=down (<-0.1%), 1=neutral, 2=up (>+0.1%)
        if snapshot.momentum < -0.1:
            momentum_bucket = 0
        elif snapshot.momentum > 0.1:
            momentum_bucket = 2
        else:
            momentum_bucket = 1
        
        return (dist_bucket, time_bucket, yes_bucket, direction, spread_bucket, momentum_bucket)
    
    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(3)
        return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Q-learning update rule."""
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def save(self, path: str):
        """Save Q-table and epsilon to file."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Q-table with {len(self.q_table)} states to {path}")

    def load(self, path: str):
        """Load Q-table and epsilon from file."""
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            # Handle both old format (just dict) and new format (dict with q_table and epsilon)
            if isinstance(loaded, dict) and 'q_table' in loaded:
                self.q_table = defaultdict(lambda: np.zeros(3), loaded['q_table'])
                self.epsilon = loaded.get('epsilon', EPSILON_WARMSTART)
            else:
                # Old format - just Q-table dict, use warm-start epsilon
                self.q_table = defaultdict(lambda: np.zeros(3), loaded)
                self.epsilon = EPSILON_WARMSTART
        print(f"Loaded Q-table with {len(self.q_table)} states from {path}")

    def save_json(self, path: str):
        """Save policy as JSON for Rust consumption."""
        # Convert dictionary keys from tuple to string/array for JSON
        # Q-table: { "d,t,p,dir,sp": [q0, q1, q2] }
        policy_data = {}
        for state, q_values in self.q_table.items():
            # state is tuple (dist, time, price, dir, spread)
            key = ",".join(map(str, state))
            # Handle both numpy arrays and lists
            if isinstance(q_values, np.ndarray):
                policy_data[key] = q_values.tolist()
            else:
                policy_data[key] = list(q_values)
            
        # Also save buckets so Rust knows how to discretize
        data = {
            "buckets": {
                "distance": DISTANCE_BUCKETS,
                "time": TIME_BUCKETS,
                "price": PRICE_BUCKETS
            },
            "q_table": policy_data,
            "timestamp": time.time()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved policy JSON to {path}")


def load_market_data(csv_path: str) -> Tuple[List[MarketSnapshot], bool]:
    """Load market data from CSV and determine settlement outcome."""
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if len(df) < 10:
            return [], False
        
        # Get strike and settlement
        strike = df['strike_price'].max()
        if strike < 100:
            return [], False
        
        final_btc = df['coinbase_price'].iloc[-1]
        yes_won = final_btc > strike
        
        # Calculate momentum (% change over rolling 30-point window)
        prices = df['coinbase_price'].values
        momentum_values = []
        for i in range(len(prices)):
            if i >= 30:
                old_price = prices[i - 30]
                new_price = prices[i]
                if old_price > 0:
                    momentum = (new_price - old_price) / old_price * 100.0
                else:
                    momentum = 0.0
            else:
                momentum = 0.0
            momentum_values.append(momentum)
        
        snapshots = []
        for idx, row in df.iterrows():
            try:
                snap = MarketSnapshot(
                    timestamp=row['timestamp_unix_ms'],
                    btc_price=row['coinbase_price'],
                    strike=strike,
                    yes_ask=row['kalshi_yes_ask'],
                    yes_bid=row['kalshi_yes_bid'],
                    no_ask=row['kalshi_no_ask'],
                    no_bid=row['kalshi_no_bid'],
                    secs_left=int(row.get('seconds_to_expiry', 0)),
                    momentum=momentum_values[idx]
                )
                if snap.secs_left > 0 and snap.yes_ask > 0:
                    snapshots.append(snap)
            except:
                continue
        
        return snapshots, yes_won
    except Exception as e:
        return [], False


def simulate_trade(snapshot: MarketSnapshot, action: int, yes_won: bool) -> float:
    """Simulate a trade and return PnL."""
    if action == HOLD:
        return 0.0
    
    bet_size = 100.0
    
    if action == BUY_YES:
        entry_price = snapshot.yes_ask
        contracts = int(bet_size / entry_price) if entry_price > 0.01 else 0
        if contracts == 0:
            return 0.0
        
        if yes_won:
            return (1.0 - entry_price) * contracts  # WIN
        else:
            return -entry_price * contracts  # LOSS
    
    elif action == BUY_NO:
        entry_price = snapshot.no_ask
        contracts = int(bet_size / entry_price) if entry_price > 0.01 else 0
        if contracts == 0:
            return 0.0
        
        if not yes_won:
            return (1.0 - entry_price) * contracts  # WIN
        else:
            return -entry_price * contracts  # LOSS
    
    return 0.0


def train_agent(data_dir: str = "data/btc/recordings", episodes: int = 100, asset: str = "btc"):
    """Train the Q-learning agent on historical data."""

    agent = QLearningAgent()

    # Get all CSV files for the asset
    ticker_pattern = ASSET_TICKER_PATTERNS.get(asset, "KXBTC15M-*.csv")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", ticker_pattern), recursive=True))
    print(f"Found {len(csv_files)} market files")
    
    # Training loop
    total_rewards = []
    
    for episode in range(episodes):
        episode_reward = 0
        trades_taken = 0
        wins = 0
        losses = 0
        
        # Shuffle files each episode
        np.random.shuffle(csv_files)
        
        for csv_file in csv_files:
            snapshots, yes_won = load_market_data(csv_file)
            if not snapshots:
                continue
            
            # Allow full trading window (30-900s) for RL to learn optimal timing
            late_snapshots = [s for s in snapshots if 30 <= s.secs_left <= 900]
            if not late_snapshots:
                continue

            # Take ONE action per market (simplified)
            # Sample across full window to learn when trades are profitable
            target_snapshots = [s for s in late_snapshots if 30 <= s.secs_left <= 900]
            if not target_snapshots:
                continue
            
            snapshot = target_snapshots[len(target_snapshots) // 2]  # Middle snapshot
            state = agent.discretize_state(snapshot)
            action = agent.choose_action(state, training=True)
            
            # Get reward (PnL from settlement)
            reward = simulate_trade(snapshot, action, yes_won)
            
            # Get next state (terminal - use same state)
            next_state = state
            
            # Update Q-table
            agent.update(state, action, reward, next_state)
            
            episode_reward += reward
            if action != HOLD:
                trades_taken += 1
                if reward > 0:
                    wins += 1
                else:
                    losses += 1
        
        agent.decay_epsilon()
        total_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Reward=${episode_reward:+.2f} "
                  f"Avg10=${avg_reward:+.2f} "
                  f"Trades={trades_taken} W/L={wins}/{losses} "
                  f"ε={agent.epsilon:.3f}")
    
    return agent


def evaluate_agent(agent: QLearningAgent, data_dir: str = "data/btc/recordings", asset: str = "btc"):
    """Evaluate the trained agent."""

    ticker_pattern = ASSET_TICKER_PATTERNS.get(asset, "KXBTC15M-*.csv")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", ticker_pattern), recursive=True))
    
    total_pnl = 0
    trades = 0
    wins = 0
    losses = 0
    holds = 0
    
    action_names = {HOLD: "HOLD", BUY_YES: "BUY_YES", BUY_NO: "BUY_NO"}
    
    print("\n" + "=" * 80)
    print("EVALUATION (Learned Policy)")
    print("=" * 80)
    
    for csv_file in csv_files[-20:]:  # Last 20 markets
        snapshots, yes_won = load_market_data(csv_file)
        if not snapshots:
            continue
        
        target_snapshots = [s for s in snapshots if 30 <= s.secs_left <= 180]
        if not target_snapshots:
            continue
        
        snapshot = target_snapshots[len(target_snapshots) // 2]
        state = agent.discretize_state(snapshot)
        action = agent.choose_action(state, training=False)
        
        pnl = simulate_trade(snapshot, action, yes_won)
        total_pnl += pnl
        
        if action == HOLD:
            holds += 1
        else:
            trades += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1
        
        ticker = os.path.basename(csv_file).replace('.csv', '')
        outcome = "YES" if yes_won else "NO"
        result = "✅" if pnl > 0 else "❌" if pnl < 0 else "⏸"
        print(f"{ticker}: {action_names[action]:>8} @ ${snapshot.yes_ask:.2f} | "
              f"{outcome} won | {result} ${pnl:+.2f}")
    
    print("\n" + "-" * 80)
    print(f"Total Trades: {trades} | Wins: {wins} | Losses: {losses} | Holds: {holds}")
    print(f"Win Rate: {wins/trades*100:.1f}%" if trades > 0 else "No trades")
    print(f"Total PnL: ${total_pnl:+.2f}")
    
    return total_pnl


def print_learned_policy(agent: QLearningAgent):
    """Print the learned Q-values for interpretation."""
    
    print("\n" + "=" * 80)
    print("LEARNED POLICY (Top Q-values)")
    print("=" * 80)
    
    action_names = {0: "HOLD", 1: "BUY_YES", 2: "BUY_NO"}
    
    # Sort states by max Q-value
    sorted_states = sorted(
        agent.q_table.items(),
        key=lambda x: np.max(x[1]),
        reverse=True
    )[:30]
    
    print(f"\n{'State':>30} | {'Best Action':>10} | {'Q-values':>30}")
    print("-" * 80)
    
    for state, q_values in sorted_states:
        best_action = np.argmax(q_values)
        dist_b, time_b, price_b, direction, spread_b = state
        
        state_str = f"d={dist_b} t={time_b} p={price_b} dir={direction} sp={spread_b}"
        q_str = f"[{q_values[0]:+.1f}, {q_values[1]:+.1f}, {q_values[2]:+.1f}]"
        
        print(f"{state_str:>30} | {action_names[best_action]:>10} | {q_str:>30}")

def continuous_training_loop(interval_minutes: int = 15, asset: str = "btc"):
    """Run training periodically."""
    print(f"Starting continuous training loop for {asset.upper()} (interval: {interval_minutes}m)...")
    data_dir = f"data/{asset}/recordings"
    model_dir = f"model/{asset}"

    while True:
        print("\n" + "="*60)
        print(f"Training cycle at {time.strftime('%H:%M:%S')}")
        print("="*60)

        # Train
        try:
            agent = train_agent(data_dir=data_dir, episodes=30, asset=asset)

            # Save models
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/rl_q_table.pkl"
            agent.save(model_path)
            agent.save_json(f"{model_dir}/rl_policy.json")

            # Evaluate recent
            evaluate_agent(agent, data_dir=data_dir, asset=asset)

        except Exception as e:
            print(f"Error in training loop: {e}")

        print(f"\nSleeping for {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RL Trading Strategy Training")
    parser.add_argument("--asset", type=str, default="btc", choices=["btc", "eth"],
                        help="Asset to train on (default: btc)")
    parser.add_argument("--loop", action="store_true", help="Run continuous training loop")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    args = parser.parse_args()

    asset = args.asset
    data_dir = f"data/{asset}/recordings"
    model_dir = f"model/{asset}"

    print("=" * 80)
    print(f"REINFORCEMENT LEARNING TRADING STRATEGY ({asset.upper()})")
    print("=" * 80)
    print()

    if args.loop:
        continuous_training_loop(asset=asset)
        return

    # Train the agent
    print(f"Training Q-Learning Agent on {asset.upper()} Historical Data...")
    print("-" * 80)
    agent = train_agent(data_dir=data_dir, episodes=args.episodes, asset=asset)

    # Save the trained model
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/rl_q_table.pkl"
    agent.save(model_path)
    agent.save_json(f"{model_dir}/rl_policy.json")

    # Print learned policy
    print_learned_policy(agent)

    # Evaluate
    evaluate_agent(agent, data_dir=data_dir, asset=asset)

    print("\n" + "=" * 80)
    print("RL Strategy trained and saved!")
    print(f"Model saved to: {model_path}")
    print(f"Policy saved to: {model_dir}/rl_policy.json")
    print("=" * 80)
    print(f"To run continuous training: python3 rl_strategy.py --asset {asset} --loop")


if __name__ == "__main__":
    main()
