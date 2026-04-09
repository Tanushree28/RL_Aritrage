#!/usr/bin/env python3
"""
Optimal RL Policy Training Script

Trains Q-learning policies for Kalshi 15-minute crypto binary options using
experience replay from SQLite databases. Supports:
- Per-asset training (BTC, ETH, SOL, XRP)
- Cross-asset pooled training (universal policy)
- Train/test split evaluation
- Baseline comparisons (random, always-hold, heuristic)
- JSON export for Rust consumption

Usage:
    python3 train_optimal_policy.py                     # Train BTC only
    python3 train_optimal_policy.py --all               # Train all assets + pooled
    python3 train_optimal_policy.py --asset eth          # Train ETH only
    python3 train_optimal_policy.py --episodes 200       # More training episodes
"""

import argparse
import json
import os
import sqlite3
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rl_strategy import (
    QLearningAgent, DISTANCE_BUCKETS, TIME_BUCKETS, PRICE_BUCKETS,
    HOLD, BUY_YES, BUY_NO, LEARNING_RATE
)

# Assets with experience databases
ASSETS = ["btc", "eth", "sol", "xrp"]

# Training hyperparameters
DISCOUNT_FACTOR = 0.0   # Episodic: each market settles independently
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995   # Slower decay for better exploration


def load_experiences_from_db(db_path: str, max_rows: int = 0) -> List[dict]:
    """Load all experiences from a SQLite database."""
    if not os.path.exists(db_path):
        print(f"  [SKIP] {db_path} not found")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT state_json, action, reward, next_state_json, timestamp FROM experiences ORDER BY timestamp"
    if max_rows > 0:
        query += f" LIMIT {max_rows}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    experiences = []
    for state_json, action, reward, next_state_json, ts in rows:
        try:
            state = tuple(json.loads(state_json))
            next_state = tuple(json.loads(next_state_json))
            experiences.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "timestamp": ts,
            })
        except (json.JSONDecodeError, TypeError):
            continue

    return experiences


def migrate_state_to_5d(state: tuple) -> tuple:
    """Convert any state format to current 5D: (dist, time, price, dir, momentum)."""
    if len(state) == 7:
        # Rust 7D: drop spread (idx 4) and time_of_day (idx 6)
        return (state[0], state[1], state[2], state[3], state[5])
    elif len(state) == 6:
        # Old 6D: drop spread (idx 4)
        return (state[0], state[1], state[2], state[3], state[5])
    elif len(state) == 5:
        return state
    return state


def filter_experiences(experiences: List[dict]) -> List[dict]:
    """Filter out invalid experiences."""
    filtered = []
    for exp in experiences:
        state = exp["state"]
        # Must have valid dimensions after migration
        migrated = migrate_state_to_5d(state)
        if len(migrated) != 5:
            continue
        # Skip if any bucket index is unreasonably large
        if any(v > 20 for v in migrated):
            continue
        filtered.append(exp)
    return filtered


def split_train_test(experiences: List[dict], test_fraction: float = 0.2) -> Tuple[List[dict], List[dict]]:
    """Split experiences by time: oldest for training, newest for testing."""
    # Sort by timestamp
    sorted_exps = sorted(experiences, key=lambda x: x["timestamp"])
    split_idx = int(len(sorted_exps) * (1 - test_fraction))
    return sorted_exps[:split_idx], sorted_exps[split_idx:]


def train_q_learning(
    train_data: List[dict],
    episodes: int = 200,
    samples_per_episode: int = 1000,
    learning_rate: float = LEARNING_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    warm_start_agent: Optional[QLearningAgent] = None,
) -> QLearningAgent:
    """Train a Q-learning agent on experience data."""
    agent = warm_start_agent or QLearningAgent()
    agent.learning_rate = learning_rate
    agent.discount_factor = discount_factor

    if warm_start_agent is None:
        agent.epsilon = EPSILON_START

    n_data = len(train_data)
    if n_data == 0:
        print("  [WARN] No training data!")
        return agent

    print(f"  Training on {n_data} experiences for {episodes} episodes...")

    for episode in range(episodes):
        # Sample with replacement
        indices = np.random.randint(0, n_data, size=min(samples_per_episode, n_data))

        episode_reward = 0.0
        trades = 0

        for idx in indices:
            exp = train_data[idx]
            state = migrate_state_to_5d(exp["state"])
            next_state = migrate_state_to_5d(exp["next_state"])
            action = exp["action"]
            reward = exp["reward"]

            # Q-learning update: Q(s,a) <- Q(s,a) + α[r + γ*max(Q(s')) - Q(s,a)]
            # With γ=0: Q(s,a) <- Q(s,a) + α[r - Q(s,a)]
            current_q = agent.q_table[state][action]
            next_max_q = np.max(agent.q_table[next_state])
            new_q = current_q + agent.learning_rate * (
                reward + agent.discount_factor * next_max_q - current_q
            )
            agent.q_table[state][action] = new_q

            episode_reward += reward
            if action != HOLD:
                trades += 1

        # Decay epsilon
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        # Decay learning rate slightly over time
        agent.learning_rate = max(0.01, agent.learning_rate * 0.999)

        if (episode + 1) % 25 == 0:
            unique_states = len(agent.q_table)
            print(f"    Episode {episode+1}/{episodes}: "
                  f"states={unique_states}, "
                  f"eps={agent.epsilon:.3f}, "
                  f"lr={agent.learning_rate:.4f}")

    return agent


def evaluate_policy(agent: QLearningAgent, test_data: List[dict], label: str = "Policy") -> Dict:
    """Evaluate a trained policy on test data."""
    total_pnl = 0.0
    wins = 0
    losses = 0
    trades = 0
    holds = 0
    pnl_list = []

    for exp in test_data:
        state = migrate_state_to_5d(exp["state"])
        true_action = exp["action"]
        true_reward = exp["reward"]

        # Skip HOLD experiences (can't determine counterfactual outcome)
        if true_action == HOLD:
            continue

        # Determine market outcome from the actual trade
        if true_action == BUY_YES:
            yes_won = true_reward > 0
        else:  # BUY_NO
            yes_won = true_reward < 0  # NO won means YES lost

        # Get agent's decision
        agent_action = agent.choose_action(state, training=False)

        if agent_action == HOLD:
            holds += 1
            continue

        # Estimate PnL from agent's action
        price_bucket = state[2]
        # Map price bucket to approximate entry price
        price_map = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        est_price = price_map[min(price_bucket, len(price_map) - 1)]

        bet_size = 100.0
        contracts = int(bet_size / est_price) if est_price > 0.01 else 0
        if contracts == 0:
            continue

        if agent_action == BUY_YES:
            if yes_won:
                pnl = (1.0 - est_price) * contracts
            else:
                pnl = -est_price * contracts
        else:  # BUY_NO
            no_price = 1.0 - est_price
            no_contracts = int(bet_size / no_price) if no_price > 0.01 else 0
            if no_contracts == 0:
                continue
            if not yes_won:
                pnl = (1.0 - no_price) * no_contracts
            else:
                pnl = -no_price * no_contracts

        total_pnl += pnl
        pnl_list.append(pnl)
        trades += 1
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    # Metrics
    win_rate = wins / trades if trades > 0 else 0.0
    avg_pnl = np.mean(pnl_list) if pnl_list else 0.0
    std_pnl = np.std(pnl_list) if len(pnl_list) > 1 else 1.0
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0

    return {
        "label": label,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "holds": holds,
        "sharpe": sharpe,
    }


def evaluate_baseline_random(test_data: List[dict]) -> Dict:
    """Baseline: random action selection."""
    total_pnl = 0.0
    wins = 0
    losses = 0
    trades = 0
    pnl_list = []

    np.random.seed(42)  # Reproducible

    for exp in test_data:
        true_action = exp["action"]
        true_reward = exp["reward"]
        if true_action == HOLD:
            continue

        if true_action == BUY_YES:
            yes_won = true_reward > 0
        else:
            yes_won = true_reward < 0

        random_action = np.random.randint(0, 3)
        if random_action == HOLD:
            continue

        price_bucket = migrate_state_to_5d(exp["state"])[2]
        price_map = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        est_price = price_map[min(price_bucket, len(price_map) - 1)]

        bet_size = 100.0
        contracts = int(bet_size / est_price) if est_price > 0.01 else 0
        if contracts == 0:
            continue

        if random_action == BUY_YES:
            pnl = ((1.0 - est_price) * contracts) if yes_won else (-est_price * contracts)
        else:
            no_price = 1.0 - est_price
            no_contracts = int(bet_size / no_price) if no_price > 0.01 else 0
            if no_contracts == 0:
                continue
            pnl = ((1.0 - no_price) * no_contracts) if not yes_won else (-no_price * no_contracts)

        total_pnl += pnl
        pnl_list.append(pnl)
        trades += 1
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    win_rate = wins / trades if trades > 0 else 0.0
    avg_pnl = np.mean(pnl_list) if pnl_list else 0.0
    std_pnl = np.std(pnl_list) if len(pnl_list) > 1 else 1.0

    return {
        "label": "Random Baseline",
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "holds": 0,
        "sharpe": avg_pnl / std_pnl if std_pnl > 0 else 0.0,
    }


def print_results(results: List[Dict]):
    """Print comparison table of evaluation results."""
    print("\n" + "=" * 90)
    print(f"{'Policy':>25} | {'PnL':>10} | {'Avg PnL':>8} | {'Win%':>6} | {'Trades':>6} | {'W/L':>7} | {'Sharpe':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:>25} | ${r['total_pnl']:>+9.2f} | ${r['avg_pnl']:>+7.2f} | "
              f"{r['win_rate']:>5.1%} | {r['trades']:>6} | "
              f"{r['wins']}/{r['losses']:>3} | {r['sharpe']:>+7.3f}")
    print("=" * 90)


def print_top_states(agent: QLearningAgent, n: int = 20):
    """Print the most confident Q-value states."""
    action_names = {0: "HOLD", 1: "BUY_YES", 2: "BUY_NO"}

    sorted_states = sorted(
        agent.q_table.items(),
        key=lambda x: np.max(x[1]),
        reverse=True,
    )[:n]

    print(f"\nTop {n} Most Confident States:")
    print(f"{'State (d,t,p,dir,mom)':>25} | {'Best':>8} | {'Q[HOLD]':>8} | {'Q[YES]':>8} | {'Q[NO]':>8}")
    print("-" * 70)

    for state, q_vals in sorted_states:
        best = action_names[np.argmax(q_vals)]
        if len(state) == 5:
            d, t, p, dr, m = state
            state_str = f"d={d} t={t} p={p} dir={dr} m={m}"
        else:
            state_str = str(state)
        print(f"{state_str:>25} | {best:>8} | {q_vals[0]:>+8.1f} | {q_vals[1]:>+8.1f} | {q_vals[2]:>+8.1f}")


def print_coverage_stats(agent: QLearningAgent):
    """Print state space coverage statistics."""
    total_possible = 6 * 11 * 10 * 2 * 3  # 3,960
    explored = len(agent.q_table)
    non_zero = sum(1 for q in agent.q_table.values() if np.any(q != 0))

    # Action distribution
    action_counts = {0: 0, 1: 0, 2: 0}
    for q_vals in agent.q_table.values():
        best = np.argmax(q_vals)
        action_counts[best] += 1

    print(f"\nState Space Coverage:")
    print(f"  Total possible states: {total_possible}")
    print(f"  States explored:       {explored} ({explored/total_possible:.1%})")
    print(f"  Non-zero Q-values:     {non_zero} ({non_zero/total_possible:.1%})")
    print(f"\nAction Distribution (best action per state):")
    print(f"  HOLD:    {action_counts[0]:>5} ({action_counts[0]/max(explored,1):.1%})")
    print(f"  BUY_YES: {action_counts[1]:>5} ({action_counts[1]/max(explored,1):.1%})")
    print(f"  BUY_NO:  {action_counts[2]:>5} ({action_counts[2]/max(explored,1):.1%})")


def train_asset(asset: str, episodes: int = 200, save: bool = True) -> Tuple[QLearningAgent, Dict]:
    """Train and evaluate a policy for a single asset."""
    print(f"\n{'#' * 80}")
    print(f"  Training {asset.upper()} Policy")
    print(f"{'#' * 80}")

    db_path = f"data/{asset}/experiences.db"
    experiences = load_experiences_from_db(db_path)
    print(f"  Loaded {len(experiences)} experiences from {db_path}")

    if len(experiences) < 100:
        print(f"  [SKIP] Not enough experiences for {asset}")
        return QLearningAgent(), {}

    experiences = filter_experiences(experiences)
    print(f"  After filtering: {len(experiences)} valid experiences")

    train_data, test_data = split_train_test(experiences, test_fraction=0.2)
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    # Train
    agent = train_q_learning(train_data, episodes=episodes)

    # Evaluate
    results = []
    results.append(evaluate_policy(agent, test_data, label=f"{asset.upper()} Q-Learning"))
    results.append(evaluate_baseline_random(test_data))

    print_results(results)
    print_coverage_stats(agent)
    print_top_states(agent)

    # Save
    if save:
        model_dir = f"model/{asset}"
        os.makedirs(model_dir, exist_ok=True)
        agent.save_json(f"{model_dir}/rl_policy.json")
        agent.save(f"{model_dir}/rl_q_table.pkl")

    return agent, results[0] if results else {}


def train_pooled(episodes: int = 200, save: bool = True) -> Tuple[QLearningAgent, Dict]:
    """Train a universal pooled policy from all assets."""
    print(f"\n{'#' * 80}")
    print(f"  Training POOLED Universal Policy")
    print(f"{'#' * 80}")

    all_experiences = []
    for asset in ASSETS:
        db_path = f"data/{asset}/experiences.db"
        exps = load_experiences_from_db(db_path)
        print(f"  {asset.upper()}: {len(exps)} experiences")
        all_experiences.extend(exps)

    print(f"  Total pooled: {len(all_experiences)} experiences")

    if len(all_experiences) < 100:
        print("  [SKIP] Not enough pooled experiences")
        return QLearningAgent(), {}

    all_experiences = filter_experiences(all_experiences)
    print(f"  After filtering: {len(all_experiences)} valid experiences")

    train_data, test_data = split_train_test(all_experiences, test_fraction=0.2)
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

    # Train with more samples per episode since we have more data
    agent = train_q_learning(
        train_data,
        episodes=episodes,
        samples_per_episode=2000,
    )

    # Evaluate
    results = []
    results.append(evaluate_policy(agent, test_data, label="POOLED Q-Learning"))
    results.append(evaluate_baseline_random(test_data))

    print_results(results)
    print_coverage_stats(agent)
    print_top_states(agent)

    # Save
    if save:
        os.makedirs("model/universal", exist_ok=True)
        agent.save_json("model/universal/rl_policy_universal.json")
        agent.save("model/universal/rl_q_table_universal.pkl")

        # Also save a copy to each asset dir as fallback
        for asset in ASSETS:
            model_dir = f"model/{asset}"
            os.makedirs(model_dir, exist_ok=True)
            agent.save_json(f"{model_dir}/rl_policy_universal.json")

    return agent, results[0] if results else {}


def main():
    parser = argparse.ArgumentParser(description="Train Optimal RL Policy")
    parser.add_argument("--asset", type=str, default="btc",
                        choices=ASSETS,
                        help="Asset to train (default: btc)")
    parser.add_argument("--all", action="store_true",
                        help="Train all assets + pooled universal policy")
    parser.add_argument("--pooled-only", action="store_true",
                        help="Train only the pooled universal policy")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes (default: 200)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save trained models")
    args = parser.parse_args()

    print("=" * 80)
    print("OPTIMAL RL POLICY TRAINING")
    print(f"State space: 6x11x10x2x3 = 3,960 states | Actions: HOLD, BUY_YES, BUY_NO")
    print(f"Discount factor (gamma): {DISCOUNT_FACTOR} (episodic)")
    print(f"Episodes: {args.episodes}")
    print("=" * 80)

    save = not args.no_save

    if args.pooled_only:
        train_pooled(episodes=args.episodes, save=save)
    elif args.all:
        # Train per-asset policies
        for asset in ASSETS:
            train_asset(asset, episodes=args.episodes, save=save)
        # Train pooled
        train_pooled(episodes=args.episodes, save=save)
    else:
        train_asset(args.asset, episodes=args.episodes, save=save)

    print("\nDone!")


if __name__ == "__main__":
    main()
