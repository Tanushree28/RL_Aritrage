#!/usr/bin/env python3
"""
Experience Replay Buffer for RL Trading Agent

Stores experiences (state, action, reward, next_state) in SQLite for:
- Prioritized experience replay during training
- Historical analysis and debugging
- Continuous learning from recent trading outcomes
"""

import sqlite3
import json
import time
from typing import Tuple, List, Optional


class ExperienceBuffer:
    """
    SQLite-backed experience replay buffer with prioritized sampling.

    Features:
    - Prioritized replay: high TD-error experiences sampled more frequently
    - Automatic pruning: keeps last 30 days + top high-priority experiences
    - Balanced sampling: mix of recent and high-value historical experiences
    """

    def __init__(self, db_path: str = "data/experiences.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main experiences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                ticker TEXT NOT NULL,
                state_json TEXT NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,
                next_state_json TEXT NOT NULL,
                td_error REAL DEFAULT 0,
                priority REAL DEFAULT 1,
                used_count INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON experiences(priority DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON experiences(created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON experiences(ticker)")

        conn.commit()
        conn.close()
        print(f"[ExperienceBuffer] Initialized database at {self.db_path}")

    def add_experience(
        self,
        ticker: str,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple
    ):
        """
        Add a single experience to the buffer.

        Args:
            ticker: Market ticker (e.g., "KXBTC15M-260206-1500-96000")
            state: Discretized state tuple (dist_bucket, time_bucket, price_bucket, direction, spread_bucket)
            action: Action taken (0=HOLD, 1=BUY_YES, 2=BUY_NO)
            reward: PnL from settlement
            next_state: Next state (terminal in most cases)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Initial priority based on absolute reward (high-impact experiences prioritized)
        initial_priority = abs(reward) + 0.01  # Small constant to avoid zero priority

        cursor.execute("""
            INSERT INTO experiences (timestamp, ticker, state_json, action, reward, next_state_json, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            ticker,
            json.dumps(state),
            action,
            reward,
            json.dumps(next_state),
            initial_priority
        ))

        conn.commit()
        conn.close()

    def add_batch(self, experiences: List[Tuple]):
        """
        Add multiple experiences in a single transaction for efficiency.

        Args:
            experiences: List of (ticker, state, action, reward, next_state) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = []
        for ticker, state, action, reward, next_state in experiences:
            priority = abs(reward) + 0.01
            data.append((
                time.time(),
                ticker,
                json.dumps(state),
                action,
                reward,
                json.dumps(next_state),
                priority
            ))

        cursor.executemany("""
            INSERT INTO experiences (timestamp, ticker, state_json, action, reward, next_state_json, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)

        conn.commit()
        conn.close()
        print(f"[ExperienceBuffer] Added {len(experiences)} experiences in batch")

    def sample_prioritized(
        self,
        n_samples: int = 500,
        priority_weight: float = 0.9,
        recent_hours: int = 48
    ) -> List[Tuple]:
        """
        Sample experiences with prioritized replay, heavily biased toward RECENT data.

        Args:
            n_samples: Total number of experiences to sample
            priority_weight: Fraction of samples from recent window (0.0-1.0)
            recent_hours: Number of recent hours to consider (default 48 = 2 days)

        Returns:
            List of (state_json, action, reward, next_state_json) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 90% from last 48 hours, 10% from historical for diversity
        cutoff_time = time.time() - (recent_hours * 3600)

        # Recent experiences (last 48 hours) - 90% of samples
        n_recent = int(n_samples * priority_weight)

        # Recent winners (40% of total)
        n_recent_winners = int(n_samples * 0.4)
        recent_winner_samples = cursor.execute("""
            SELECT state_json, action, reward, next_state_json
            FROM experiences
            WHERE reward > 0 AND action != 0 AND timestamp > ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (cutoff_time, n_recent_winners)).fetchall()

        # Recent losers (25% of total) - need to learn from mistakes too
        n_recent_losers = int(n_samples * 0.25)
        recent_loser_samples = cursor.execute("""
            SELECT state_json, action, reward, next_state_json
            FROM experiences
            WHERE reward < 0 AND action != 0 AND timestamp > ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (cutoff_time, n_recent_losers)).fetchall()

        # Recent HOLDs (25% of total) - learn when NOT to trade
        n_recent_holds = int(n_samples * 0.25)
        recent_hold_samples = cursor.execute("""
            SELECT state_json, action, reward, next_state_json
            FROM experiences
            WHERE action = 0 AND timestamp > ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (cutoff_time, n_recent_holds)).fetchall()

        # Historical diversity (10% of total) - top performers from older data
        n_historical = n_samples - len(recent_winner_samples) - len(recent_loser_samples) - len(recent_hold_samples)
        historical_samples = cursor.execute("""
            SELECT state_json, action, reward, next_state_json
            FROM experiences
            WHERE timestamp < ? AND action != 0
            ORDER BY ABS(reward) DESC
            LIMIT ?
        """, (cutoff_time, max(0, n_historical))).fetchall()

        conn.close()

        total_sampled = len(recent_winner_samples) + len(recent_loser_samples) + len(recent_hold_samples) + len(historical_samples)
        print(f"[ExperienceBuffer] Sampled {total_sampled} experiences from last {recent_hours}h: "
              f"{len(recent_winner_samples)} winners, {len(recent_loser_samples)} losers, "
              f"{len(recent_hold_samples)} holds, {len(historical_samples)} historical)")

        return recent_winner_samples + recent_loser_samples + recent_hold_samples + historical_samples

    def update_priorities(self, td_errors: List[Tuple[int, float]]):
        """
        Update experience priorities based on TD errors from training.

        Args:
            td_errors: List of (experience_id, td_error) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for exp_id, td_error in td_errors:
            # Priority = |TD error| + small constant (to avoid zero priority)
            priority = abs(td_error) + 0.01
            cursor.execute("""
                UPDATE experiences
                SET td_error = ?, priority = ?, used_count = used_count + 1
                WHERE id = ?
            """, (td_error, priority, exp_id))

        conn.commit()
        conn.close()
        print(f"[ExperienceBuffer] Updated priorities for {len(td_errors)} experiences")

    def prune_old_experiences(
        self,
        keep_days: int = 30,
        keep_top_n: int = 500
    ) -> int:
        """
        Remove old low-priority experiences to manage database size.

        Args:
            keep_days: Keep all experiences from last N days
            keep_top_n: Always keep top N high-priority experiences regardless of age

        Returns:
            Number of experiences deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = time.time() - (keep_days * 24 * 3600)

        # Delete old experiences, but preserve high-priority ones
        cursor.execute("""
            DELETE FROM experiences
            WHERE created_at < ?
            AND id NOT IN (
                SELECT id FROM experiences
                ORDER BY priority DESC
                LIMIT ?
            )
        """, (cutoff_time, keep_top_n))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted > 0:
            print(f"[ExperienceBuffer] Pruned {deleted} old low-priority experiences")

        return deleted

    def get_stats(self) -> dict:
        """Get buffer statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        total = cursor.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

        avg_reward = cursor.execute("SELECT AVG(reward) FROM experiences").fetchone()[0]

        action_dist = cursor.execute("""
            SELECT action, COUNT(*) FROM experiences
            GROUP BY action
        """).fetchall()

        recent_count = cursor.execute("""
            SELECT COUNT(*) FROM experiences
            WHERE timestamp > ?
        """, (time.time() - 24*3600,)).fetchone()[0]

        conn.close()

        return {
            "total_experiences": total,
            "avg_reward": avg_reward or 0.0,
            "action_distribution": dict(action_dist),
            "last_24h_count": recent_count
        }

    def get_count(self) -> int:
        """Get total number of experiences in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        conn.close()
        return count

    def get_recent_holdout(
        self,
        fraction: float = 0.2,
        min_age_seconds: int = 300
    ) -> List[dict]:
        """
        Get recent experiences for hold-out evaluation.

        Args:
            fraction: Fraction of recent experiences to return (0-1)
            min_age_seconds: Minimum age to avoid training/test contamination

        Returns:
            List of experience dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent experiences excluding the very latest
        cutoff_time = time.time() - min_age_seconds

        rows = cursor.execute("""
            SELECT state_json, action, reward, next_state_json
            FROM experiences
            WHERE timestamp < ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (cutoff_time, int(1000 * fraction))).fetchall()

        conn.close()

        return [
            {
                'state_json': row[0],
                'action': row[1],
                'reward': row[2],
                'next_state_json': row[3]
            }
            for row in rows
        ]


def main():
    """Test the experience buffer"""
    buffer = ExperienceBuffer()

    # Add some test experiences
    print("\nAdding test experiences...")
    buffer.add_experience("KXBTC-TEST-1", (1, 2, 3, 0, 0), 1, 50.0, (1, 2, 3, 0, 0))
    buffer.add_experience("KXBTC-TEST-2", (2, 3, 4, 1, 1), 2, -30.0, (2, 3, 4, 1, 1))
    buffer.add_experience("KXBTC-TEST-3", (1, 1, 5, 0, 0), 1, 80.0, (1, 1, 5, 0, 0))

    # Get stats
    stats = buffer.get_stats()
    print(f"\nBuffer Stats:")
    print(f"  Total Experiences: {stats['total_experiences']}")
    print(f"  Avg Reward: ${stats['avg_reward']:.2f}")
    print(f"  Action Distribution: {stats['action_distribution']}")
    print(f"  Last 24h: {stats['last_24h_count']}")

    # Test sampling
    if stats['total_experiences'] > 0:
        print(f"\nSampling 2 experiences...")
        samples = buffer.sample_prioritized(n_samples=min(2, stats['total_experiences']))
        for i, (state_json, action, reward, _) in enumerate(samples):
            print(f"  Sample {i+1}: action={action}, reward=${reward:.2f}, state={state_json}")


if __name__ == "__main__":
    main()
