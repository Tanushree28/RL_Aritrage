#!/usr/bin/env python3
"""
Continuous Learning Pipeline for RL Trading Agent

Orchestrates the full training cycle:
1. Load current production model (warm-start)
2. Sample experiences from replay buffer (prioritized + recent)
3. Train using incremental Q-learning updates
4. Evaluate on held-out recent markets
5. Promote to production if passes safety checks

Usage:
    python3 continuous_learner.py                    # Single training run
    python3 continuous_learner.py --loop              # Continuous learning loop
    python3 continuous_learner.py --loop --interval 120  # 2-hour intervals
"""

import argparse
import glob
import json
import pathlib
import shutil
import sqlite3
import sys
import time
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from model_versioning import ModelRegistry
from experience_buffer import ExperienceBuffer
from rl_strategy import QLearningAgent, LEARNING_RATE, DISCOUNT_FACTOR

# Optional DQN support
try:
    from dqn_agent import DQNAgent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("[ContinuousLearner] DQN not available, using Q-learning only")


def migrate_state(state_tuple):
    """
    Migrate old 5D states to new 6D format by adding momentum_bucket=1 (neutral).
    New 6D format: (dist, time, price, direction, spread, momentum)
    """
    if len(state_tuple) == 5:
        # Old format: add neutral momentum (1)
        return state_tuple + (1,)
    return state_tuple  # Already 6D or different format


class ContinuousLearner:
    """
    Continuous learning pipeline with:
    - Warm-start from current production model
    - Prioritized experience replay
    - Automated evaluation and promotion
    - Safe rollback capability
    - Support for both Q-learning and DQN
    """

    def __init__(
        self,
        asset: str = "btc",
        registry_path: str = "model_registry/btc",
        db_path: str = "data/btc/experiences.db",
        data_dir: str = "data/btc/recordings",
        use_dqn: bool = False
    ):
        self.asset = asset
        self.registry = ModelRegistry(registry_path)
        self.buffer = ExperienceBuffer(db_path)
        self.data_dir = data_dir
        self.agent = None
        self.use_dqn = use_dqn and DQN_AVAILABLE

        if self.use_dqn:
            print(f"[ContinuousLearner] Using DQN mode for {asset}")
        else:
            print(f"[ContinuousLearner] Using Q-learning mode for {asset}")

    def load_current_model(self) -> bool:
        """
        Load current production model for warm-start.

        Returns:
            True if model loaded successfully, False if starting from scratch
        """
        current_version = self.registry.get_current_version()

        if self.use_dqn:
            # DQN model loading
            if current_version:
                model_path = f"model_registry/{self.asset}/versions/{current_version}/dqn_checkpoint.pt"
                try:
                    self.agent = DQNAgent(load_from=model_path)
                    print(f"[ContinuousLearner] Loaded DQN model: {current_version}")
                    print(f"  Epsilon: {self.agent.epsilon:.4f}")
                    print(f"  Train steps: {self.agent.train_steps}")
                    return True
                except Exception as e:
                    print(f"[ContinuousLearner] Failed to load DQN model {current_version}: {e}")
                    print("[ContinuousLearner] Starting with fresh DQN model")
                    self.agent = DQNAgent()
                    return False
            else:
                print("[ContinuousLearner] No current DQN model found, starting fresh")
                self.agent = DQNAgent()
                return False
        else:
            # Q-learning model loading
            if current_version:
                model_path = f"model_registry/{self.asset}/versions/{current_version}/rl_q_table.pkl"
                try:
                    self.agent = QLearningAgent(load_from=model_path)
                    print(f"[ContinuousLearner] Loaded current model: {current_version}")
                    print(f"  Q-table states: {len(self.agent.q_table)}")
                    print(f"  Epsilon: {self.agent.epsilon:.4f}")
                    return True
                except Exception as e:
                    print(f"[ContinuousLearner] Failed to load model {current_version}: {e}")
                    print("[ContinuousLearner] Starting with fresh model")
                    self.agent = QLearningAgent()
                    return False
            else:
                print("[ContinuousLearner] No current model found, starting fresh")
                self.agent = QLearningAgent()
                return False

    def train_episode(self, n_samples: int = 500) -> float:
        """
        Train one episode using prioritized experience replay.

        Args:
            n_samples: Number of experiences to sample

        Returns:
            Average TD error for the episode
        """
        # Sample experiences with prioritization
        samples = self.buffer.sample_prioritized(n_samples=n_samples)

        if len(samples) == 0:
            print("[ContinuousLearner] No experiences available for training")
            return 0.0

        total_td_error = 0.0
        valid_samples = 0

        for state_json, action, reward, next_state_json in samples:
            try:
                # Deserialize states and migrate old 5D -> 6D format
                state = migrate_state(tuple(json.loads(state_json)))
                next_state = migrate_state(tuple(json.loads(next_state_json)))

                # Q-learning update
                self.agent.update(state, action, reward, next_state)

                # Track TD error for monitoring
                current_q = self.agent.q_table[state][action]
                next_max_q = np.max(self.agent.q_table[next_state])
                td_error = abs(reward + DISCOUNT_FACTOR * next_max_q - current_q)
                total_td_error += td_error
                valid_samples += 1

            except Exception as e:
                print(f"[ContinuousLearner] Error processing experience: {e}")
                continue

        avg_td_error = total_td_error / valid_samples if valid_samples > 0 else 0.0
        return avg_td_error

    def train_episode_dqn(self, batch_size: int = 64) -> float:
        """
        Train one episode using DQN with batch updates.

        Args:
            batch_size: Batch size for training

        Returns:
            Average loss for the episode
        """
        # Sample experiences with prioritization
        samples = self.buffer.sample_prioritized(n_samples=batch_size * 8)  # Get more samples for batching

        if len(samples) < batch_size:
            print(f"[ContinuousLearner] Not enough experiences for DQN batch: {len(samples)} < {batch_size}")
            return 0.0

        # Prepare batch data
        states = []
        actions = []
        rewards = []
        next_states = []

        for state_json, action, reward, next_state_json in samples:
            try:
                state = migrate_state(tuple(json.loads(state_json)))
                next_state = migrate_state(tuple(json.loads(next_state_json)))

                # Convert discrete state tuples to continuous features
                state_cont = self.agent.state_tuple_to_continuous(state)
                next_state_cont = self.agent.state_tuple_to_continuous(next_state)

                states.append(state_cont)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state_cont)

            except Exception as e:
                continue

        if len(states) < batch_size:
            return 0.0

        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        # Normalize rewards to [-1, 1] range for stable DQN training
        # Using tanh-like normalization: reward / (|reward| + 100)
        rewards = rewards / (np.abs(rewards) + 100.0)

        # Train on multiple batches
        n_batches = len(states) // batch_size
        total_loss = 0.0

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_loss = self.agent.train_batch(
                states[start_idx:end_idx],
                actions[start_idx:end_idx],
                rewards[start_idx:end_idx],
                next_states[start_idx:end_idx]
            )
            total_loss += batch_loss

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss

    def train_episode_from_csv(self) -> float:
        """
        Train one episode using historical CSV data (fallback when insufficient live experiences).

        Returns:
            Average TD error for the episode
        """
        from rl_strategy import load_market_data, simulate_trade

        # Get all CSV files
        csv_files = glob.glob(f"{self.data_dir}/**/*.csv", recursive=True)

        if len(csv_files) == 0:
            print("[ContinuousLearner] No CSV files available for training")
            return 0.0

        # Shuffle and sample subset for this episode
        np.random.shuffle(csv_files)
        episode_files = csv_files[:min(50, len(csv_files))]  # Sample up to 50 markets per episode

        total_td_error = 0.0
        valid_updates = 0

        for csv_file in episode_files:
            snapshots, yes_won = load_market_data(csv_file)
            if not snapshots:
                continue

            # Sample from trading window
            target_snapshots = [s for s in snapshots if 30 <= s.secs_left <= 900]
            if not target_snapshots:
                continue

            # Take middle snapshot
            snapshot = target_snapshots[len(target_snapshots) // 2]
            state = self.agent.discretize_state(snapshot)
            action = self.agent.choose_action(state, training=True)

            # Get reward from settlement
            reward = simulate_trade(snapshot, action, yes_won)

            # Terminal state (use same state as next_state)
            next_state = state

            # Q-learning update
            self.agent.update(state, action, reward, next_state)

            # Track TD error
            current_q = self.agent.q_table[state][action]
            next_max_q = np.max(self.agent.q_table[next_state])
            td_error = abs(reward + DISCOUNT_FACTOR * next_max_q - current_q)
            total_td_error += td_error
            valid_updates += 1

        avg_td_error = total_td_error / valid_updates if valid_updates > 0 else 0.0
        return avg_td_error

    def evaluate_model(
        self,
        agent: QLearningAgent,
        n_markets: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate agent on held-out recent markets.

        Args:
            agent: Agent to evaluate
            n_markets: Number of recent markets to evaluate on

        Returns:
            Dictionary with evaluation metrics
        """
        from rl_strategy import load_market_data, simulate_trade

        # Get recent market files
        csv_files = sorted(
            glob.glob(f"{self.data_dir}/**/*.csv", recursive=True)
        )[-n_markets:]

        if len(csv_files) == 0:
            print("[ContinuousLearner] No market data available for evaluation")
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "wins": 0,
                "losses": 0,
                "sharpe_ratio": 0.0,
                "num_markets": 0
            }

        total_pnl = 0.0
        wins = 0
        losses = 0
        trades = 0

        for csv_file in csv_files:
            snapshots, yes_won = load_market_data(csv_file)
            if not snapshots:
                continue

            # Evaluate on late-game snapshot (60-120s before expiry)
            target_snapshots = [s for s in snapshots if 30 <= s.secs_left <= 180]
            if not target_snapshots:
                continue

            # Take middle snapshot for stability
            snapshot = target_snapshots[len(target_snapshots) // 2]
            state = agent.discretize_state(snapshot)
            action = agent.choose_action(state, training=False)

            # Simulate trade outcome
            pnl = simulate_trade(snapshot, action, yes_won)
            total_pnl += pnl

            if action != 0:  # Not HOLD
                trades += 1
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        # Calculate metrics
        win_rate = wins / trades if trades > 0 else 0.0
        sharpe = self._calculate_sharpe(total_pnl, trades)

        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "num_trades": trades,
            "wins": wins,
            "losses": losses,
            "sharpe_ratio": sharpe,
            "num_markets": len(csv_files)
        }

    def _calculate_sharpe(self, total_pnl: float, num_trades: int) -> float:
        """
        Calculate approximate Sharpe ratio.

        Args:
            total_pnl: Total profit/loss
            num_trades: Number of trades

        Returns:
            Sharpe ratio (simplified calculation)
        """
        if num_trades == 0:
            return 0.0

        avg_return = total_pnl / num_trades
        # Simplified: assume std dev of ~30% of avg return magnitude
        std_dev = abs(avg_return) * 0.3 + 0.01  # Small constant to avoid division by zero

        return avg_return / std_dev

    def evaluate_model_on_live_data(self, agent: QLearningAgent) -> Dict[str, Any]:
        """
        Evaluate agent on hold-out live experiences (most recent 20%).

        This provides more realistic evaluation than CSV files by testing
        on actual live trading decisions and outcomes.

        Args:
            agent: Agent to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Get hold-out experiences (last 20% of experiences, excluding very recent)
        test_experiences = self.buffer.get_recent_holdout(fraction=0.2)

        if len(test_experiences) < 10:
            print("[ContinuousLearner] Insufficient live experiences for evaluation, falling back to CSV")
            return self.evaluate_model(agent)

        print(f"[ContinuousLearner] Evaluating on {len(test_experiences)} hold-out live experiences")

        total_pnl = 0.0
        wins = 0
        losses = 0
        trades = 0
        pnl_list = []

        for exp in test_experiences:
            try:
                state = migrate_state(tuple(json.loads(exp['state_json'])))
                true_action = exp['action']
                true_reward = exp['reward']

                # Skip HOLD experiences (can't determine market outcome)
                if true_action == 0:
                    continue

                # Infer market outcome from historical action and reward
                if true_action == 1:  # BUY_YES
                    yes_won = true_reward > 0
                else:  # BUY_NO
                    yes_won = true_reward < 0  # NO won means YES lost

                # Get agent's decision (exploitation mode)
                agent_action = agent.choose_action(state, training=False)

                # Calculate agent's PnL based on its chosen action (counterfactual)
                if agent_action == 0:  # HOLD
                    agent_pnl = 0.0
                else:
                    # Estimate entry price from state's price bucket
                    price_bucket = state[2]  # price is 3rd element in state tuple
                    price_midpoints = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
                    est_price = price_midpoints[min(price_bucket, len(price_midpoints) - 1)]

                    bet_size = 100.0
                    contracts = int(bet_size / est_price) if est_price > 0.01 else 0

                    # Kalshi fee: ceil(0.07 * price * (1-price) * contracts * 100) / 100
                    entry_fee = np.ceil(0.07 * est_price * (1.0 - est_price) * contracts * 100) / 100

                    if agent_action == 1:  # BUY_YES
                        if yes_won:
                            agent_pnl = (1.0 - est_price) * contracts - entry_fee
                        else:
                            agent_pnl = -est_price * contracts - entry_fee
                    else:  # BUY_NO
                        no_price = 1.0 - est_price
                        no_contracts = int(bet_size / no_price) if no_price > 0.01 else 0
                        no_fee = np.ceil(0.07 * no_price * (1.0 - no_price) * no_contracts * 100) / 100
                        if not yes_won:
                            agent_pnl = (1.0 - no_price) * no_contracts - no_fee
                        else:
                            agent_pnl = -no_price * no_contracts - no_fee

                total_pnl += agent_pnl
                if agent_action != 0:
                    trades += 1
                    pnl_list.append(agent_pnl)
                    if agent_pnl > 0:
                        wins += 1
                    else:
                        losses += 1

            except Exception as e:
                print(f"[ContinuousLearner] Error evaluating experience: {e}")
                continue

        # Calculate metrics
        win_rate = wins / trades if trades > 0 else 0.0

        # Calculate Sharpe ratio using actual PnL variance
        if len(pnl_list) > 1:
            avg_pnl = np.mean(pnl_list)
            std_pnl = np.std(pnl_list)
            sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
        else:
            sharpe = self._calculate_sharpe(total_pnl, trades)

        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "num_trades": trades,
            "wins": wins,
            "losses": losses,
            "sharpe_ratio": sharpe,
            "num_markets": len(test_experiences)
        }

    def should_promote(
        self,
        new_metrics: Dict[str, Any],
        current_metrics: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Decide if new model should be promoted to production.

        Safety thresholds:
        - Minimum win rate: 50%
        - Minimum Sharpe ratio: 1.0
        - Must improve PnL by at least 5% over current model

        Args:
            new_metrics: Metrics for newly trained model
            current_metrics: Metrics for current production model (None if no current model)

        Returns:
            True if model should be promoted
        """
        # Safety thresholds
        MIN_WIN_RATE = 0.50
        MIN_SHARPE = 1.0
        MIN_IMPROVEMENT = 1.05  # 5% better PnL

        # Must have at least some trades to evaluate
        if new_metrics["num_trades"] < 5:
            print(f"[ContinuousLearner] ❌ Too few trades: {new_metrics['num_trades']} < 5")
            return False

        # Must meet minimum quality bar
        if new_metrics["win_rate"] < MIN_WIN_RATE:
            print(f"[ContinuousLearner] ❌ Win rate too low: {new_metrics['win_rate']:.2%} < {MIN_WIN_RATE:.2%}")
            return False

        if new_metrics["sharpe_ratio"] < MIN_SHARPE:
            print(f"[ContinuousLearner] ❌ Sharpe ratio too low: {new_metrics['sharpe_ratio']:.2f} < {MIN_SHARPE:.2f}")
            return False

        # Must improve over current model (if exists)
        if current_metrics and current_metrics["num_trades"] >= 5:
            required_pnl = current_metrics["total_pnl"] * MIN_IMPROVEMENT
            if new_metrics["total_pnl"] < required_pnl:
                print(f"[ContinuousLearner] ❌ PnL improvement insufficient:")
                print(f"  New: ${new_metrics['total_pnl']:.2f}")
                print(f"  Required: ${required_pnl:.2f} (5% better than current ${current_metrics['total_pnl']:.2f})")
                return False

        # All checks passed
        print(f"[ContinuousLearner] ✅ Model passes promotion criteria:")
        print(f"  Win rate: {new_metrics['win_rate']:.2%}")
        print(f"  Sharpe: {new_metrics['sharpe_ratio']:.2f}")
        print(f"  PnL: ${new_metrics['total_pnl']:.2f}")
        print(f"  Trades: {new_metrics['num_trades']}")

        return True

    def train_and_evaluate(
        self,
        n_episodes: int = 30,
        n_samples_per_episode: int = 500
    ) -> bool:
        """
        Full training cycle: train → evaluate → promote.

        Args:
            n_episodes: Number of training episodes
            n_samples_per_episode: Experiences to sample per episode

        Returns:
            True if model was promoted, False otherwise
        """
        print("=" * 80)
        print("CONTINUOUS LEARNING CYCLE")
        print("=" * 80)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        print()

        # Report data sources
        exp_count = self.buffer.get_count()
        csv_count = len(glob.glob(f"{self.data_dir}/**/*.csv", recursive=True))

        print("Data Sources:")
        print(f"  Live Experiences: {exp_count}")
        print(f"  CSV Market Files: {csv_count}")

        # Determine training mode (use experience replay if we have enough data)
        use_experience_replay = exp_count >= 100
        model_type = "DQN" if self.use_dqn else "Q-LEARNING"
        training_mode = "EXPERIENCE REPLAY" if use_experience_replay else "CSV FALLBACK"
        print(f"  Model Type: {model_type}")
        print(f"  Training Mode: {training_mode}")
        print()

        # Load current model for warm-start
        has_current = self.load_current_model()

        # Evaluate current model (if exists)
        current_metrics = None
        if has_current and self.registry.get_current_version():
            print("\n" + "-" * 80)
            print("Evaluating current production model...")
            print("-" * 80)

            # Use live data evaluation if we have enough experiences
            if exp_count >= 50:
                current_metrics = self.evaluate_model_on_live_data(self.agent)
            else:
                current_metrics = self.evaluate_model(self.agent)

            self._print_metrics("CURRENT MODEL", current_metrics)

        # Train new model
        print("\n" + "-" * 80)
        print(f"Training for {n_episodes} episodes ({model_type})...")
        print("-" * 80)

        for episode in range(n_episodes):
            if self.use_dqn:
                # DQN batch training
                avg_metric = self.train_episode_dqn(batch_size=64)
                metric_name = "Loss"
            elif use_experience_replay:
                avg_metric = self.train_episode(n_samples=n_samples_per_episode)
                metric_name = "Avg TD Error"
            else:
                avg_metric = self.train_episode_from_csv()
                metric_name = "Avg TD Error"

            # Decay epsilon after each episode
            self.agent.decay_epsilon()

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode+1}/{n_episodes}: {metric_name} = {avg_metric:.4f}, ε={self.agent.epsilon:.3f}")

        # Evaluate new model
        print("\n" + "-" * 80)
        print("Evaluating new model...")
        print("-" * 80)

        # Use live data evaluation if we have enough experiences
        if exp_count >= 50:
            new_metrics = self.evaluate_model_on_live_data(self.agent)
        else:
            new_metrics = self.evaluate_model(self.agent)

        self._print_metrics("NEW MODEL", new_metrics)

        # Save to staging
        metadata = {
            "evaluation": new_metrics,
            "current_model_eval": current_metrics,
            "n_episodes": n_episodes,
            "n_samples_per_episode": n_samples_per_episode,
            "training_timestamp": time.time(),
            "model_type": "dqn" if self.use_dqn else "q_learning"
        }

        print("\n" + "-" * 80)
        print("Saving model to staging...")
        print("-" * 80)
        version_name = self.registry.save_model(self.agent, metadata, stage="staging")

        # Promotion decision
        print("\n" + "-" * 80)
        print("Promotion Decision")
        print("-" * 80)

        if self.should_promote(new_metrics, current_metrics):
            # Move from staging to versions
            staging_path = f"model_registry/{self.asset}/staging/{version_name}"
            version_path = f"model_registry/{self.asset}/versions/{version_name}"

            if pathlib.Path(staging_path).exists():
                shutil.move(staging_path, version_path)

            # Promote to production
            self.registry.promote_to_production(version_name)

            print(f"\n{'=' * 80}")
            print(f"✅ PROMOTED {version_name} TO PRODUCTION")
            print(f"{'=' * 80}")

            return True
        else:
            print(f"\n{'=' * 80}")
            print(f"❌ MODEL NOT PROMOTED - Keeping current version")
            print(f"{'=' * 80}")

            return False

    def _print_metrics(self, label: str, metrics: Dict[str, Any]):
        """Pretty-print evaluation metrics"""
        print(f"\n{label}:")
        print(f"  Total PnL:     ${metrics['total_pnl']:+.2f}")
        print(f"  Win Rate:      {metrics['win_rate']:.1%}")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
        print(f"  Trades:        {metrics['num_trades']} ({metrics['wins']}W / {metrics['losses']}L)")
        print(f"  Markets:       {metrics['num_markets']}")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Learning Pipeline for RL Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 continuous_learner.py --asset btc           # Single training run for BTC
  python3 continuous_learner.py --asset eth --loop    # Continuous loop for ETH
  python3 continuous_learner.py --asset btc --loop --interval 60  # 1-hour intervals
        """
    )

    parser.add_argument(
        "--asset",
        type=str,
        default="btc",
        choices=["btc", "eth", "sol"],
        help="Asset to train on (default: btc)"
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuous training loop"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Training interval in minutes (default: 120)"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of training episodes per cycle (default: 30)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Experiences to sample per episode (default: 500)"
    )

    parser.add_argument(
        "--dqn",
        action="store_true",
        help="Use Deep Q-Network instead of tabular Q-learning"
    )

    args = parser.parse_args()

    # Import pathlib here to avoid circular import
    import pathlib

    # Create asset-specific learner
    learner = ContinuousLearner(
        asset=args.asset,
        registry_path=f"model_registry/{args.asset}",
        db_path=f"data/{args.asset}/experiences.db",
        data_dir=f"data/{args.asset}/recordings",
        use_dqn=args.dqn
    )

    if args.loop:
        print(f"Starting continuous training loop (interval: {args.interval} minutes)")
        print("Press Ctrl+C to stop")
        print()

        cycle_count = 0
        while True:
            cycle_count += 1
            print(f"\n{'#' * 80}")
            print(f"TRAINING CYCLE #{cycle_count}")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
            print(f"{'#' * 80}\n")

            try:
                learner.train_and_evaluate(
                    n_episodes=args.episodes,
                    n_samples_per_episode=args.samples
                )
            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal, shutting down...")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Error in training cycle: {e}")
                import traceback
                traceback.print_exc()

            print(f"\n{'=' * 80}")
            print(f"Sleeping for {args.interval} minutes...")
            print(f"Next cycle at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(time.time() + args.interval * 60))}")
            print(f"{'=' * 80}\n")

            try:
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                print("\n\nReceived interrupt signal, shutting down...")
                sys.exit(0)
    else:
        # Single training run
        success = learner.train_and_evaluate(
            n_episodes=args.episodes,
            n_samples_per_episode=args.samples
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
