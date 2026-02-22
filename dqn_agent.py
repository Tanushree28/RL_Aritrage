#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Agent for Trading

Replaces tabular Q-learning with neural network for better generalization.
Uses Double DQN to reduce overestimation bias.

State: 6D continuous (dist_pct, time_frac, price, direction, spread, momentum)
Actions: 0=HOLD, 1=BUY_YES, 2=BUY_NO
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from collections import deque
import random


# Hyperparameters
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.95
EPSILON_START = 0.15
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100  # Update target network every N training steps
BATCH_SIZE = 64

# State normalization bounds (for scaling to [-1, 1])
STATE_BOUNDS = {
    'dist_pct': (0.0, 0.5),      # Distance buckets 0-5 map to 0-0.5%
    'time_frac': (0.0, 1.0),     # Time as fraction of 300s
    'price': (0.0, 1.0),         # Yes price 0-1
    'direction': (0.0, 1.0),     # 0 or 1
    'spread': (0.0, 0.1),        # Spread typically 0-10 cents
    'momentum': (-0.5, 0.5),     # Momentum -0.5% to +0.5%
}


class DQNetwork(nn.Module):
    """
    Deep Q-Network with configurable hidden layers.

    Architecture: Input → Hidden layers with ReLU → Output (Q-values per action)
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dims: List[int] = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values for each action."""
        return self.network(x)


class DQNAgent:
    """
    Double DQN Agent with experience replay.

    Features:
    - Double DQN: Uses policy network to select actions, target network to evaluate
    - Soft target updates for stability
    - State normalization for consistent learning
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: List[int] = [64, 32],
        learning_rate: float = LEARNING_RATE,
        load_from: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Exploration
        self.epsilon = EPSILON_START

        # State normalization stats (updated during training)
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self._state_samples = []

        # Training counter
        self.train_steps = 0

        if load_from and os.path.exists(load_from):
            self.load(load_from)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def update_normalization_stats(self, states: np.ndarray):
        """Update running mean/std from batch of states."""
        self._state_samples.extend(states.tolist())

        # Keep last 10000 samples
        if len(self._state_samples) > 10000:
            self._state_samples = self._state_samples[-10000:]

        if len(self._state_samples) >= 100:
            samples = np.array(self._state_samples)
            self.state_mean = samples.mean(axis=0)
            self.state_std = samples.std(axis=0)
            self.state_std = np.maximum(self.state_std, 1e-8)  # Avoid division by zero

    def state_tuple_to_continuous(self, state_tuple: Tuple) -> np.ndarray:
        """
        Convert discrete state tuple to continuous features.

        Input: (dist_bucket, time_bucket, price_bucket, direction, spread_bucket, momentum_bucket)
        Output: Continuous [6] array
        """
        dist_bucket, time_bucket, price_bucket, direction, spread_bucket, momentum_bucket = state_tuple

        # Convert buckets to continuous values (midpoints)
        dist_pct = dist_bucket * 0.1  # Each bucket is ~0.1%
        time_frac = time_bucket / 10.0  # 10 time buckets
        price = price_bucket * 0.1 + 0.05  # Price buckets 0-9 → 0.05-0.95
        direction = float(direction)
        spread = spread_bucket * 0.05  # 0 or 1 → 0 or 0.05
        momentum = (momentum_bucket - 1) * 0.15  # 0,1,2 → -0.15, 0, 0.15

        return np.array([dist_pct, time_frac, price, direction, spread, momentum], dtype=np.float32)

    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: State tuple (discrete buckets)
            training: If True, use epsilon-greedy; if False, pure exploitation

        Returns:
            Action index (0=HOLD, 1=BUY_YES, 2=BUY_NO)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Convert to continuous and normalize
        state_cont = self.state_tuple_to_continuous(state)
        state_norm = self.normalize_state(state_cont)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: Optional[np.ndarray] = None
    ) -> float:
        """
        Train on a batch of experiences using Double DQN.

        Returns:
            Average loss for the batch
        """
        if dones is None:
            dones = np.ones(len(states))  # All terminal for our single-step setting

        # Update normalization stats
        self.update_normalization_stats(states)

        # Normalize states
        states_norm = np.array([self.normalize_state(s) for s in states])
        next_states_norm = np.array([self.normalize_state(s) for s in next_states])

        # Convert to tensors
        states_t = torch.FloatTensor(states_norm).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states_norm).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Double DQN: Use policy net to select actions, target net to evaluate
        with torch.no_grad():
            # Policy network selects best actions
            next_actions = self.policy_net(next_states_t).argmax(1)
            # Target network evaluates those actions
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1))
            # Target Q values
            target_q = rewards_t.unsqueeze(1) + DISCOUNT_FACTOR * next_q * (1 - dones_t.unsqueeze(1))

        # Huber loss for stability
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_mean': self.state_mean.tolist(),
            'state_std': self.state_std.tolist(),
            'train_steps': self.train_steps,
        }

        torch.save(checkpoint, path)
        print(f"[DQNAgent] Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', EPSILON_START)
        self.state_mean = np.array(checkpoint.get('state_mean', np.zeros(self.state_dim)))
        self.state_std = np.array(checkpoint.get('state_std', np.ones(self.state_dim)))
        self.train_steps = checkpoint.get('train_steps', 0)

        print(f"[DQNAgent] Loaded checkpoint from {path}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        print(f"  Train steps: {self.train_steps}")

    def export_onnx(self, path: str):
        """Export policy network to ONNX for Rust inference."""
        self.policy_net.eval()

        # Dummy input for tracing
        dummy_input = torch.randn(1, self.state_dim).to(self.device)

        torch.onnx.export(
            self.policy_net,
            dummy_input,
            path,
            input_names=['state'],
            output_names=['q_values'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'q_values': {0: 'batch_size'}
            },
            opset_version=18
        )

        print(f"[DQNAgent] Exported ONNX model to {path}")

    def save_scaler(self, path: str):
        """Save normalization parameters for Rust."""
        scaler_data = {
            'mean': self.state_mean.tolist(),
            'std': self.state_std.tolist()
        }

        with open(path, 'w') as f:
            json.dump(scaler_data, f, indent=2)

        print(f"[DQNAgent] Saved scaler to {path}")


def test_agent():
    """Test DQN agent functionality."""
    print("Testing DQN Agent...")

    agent = DQNAgent()

    # Test action selection
    test_state = (2, 3, 5, 1, 0, 1)  # Example state tuple
    action = agent.choose_action(test_state, training=False)
    print(f"Test state {test_state} → Action {action}")

    # Test batch training
    batch_size = 32
    states = np.random.randn(batch_size, 6).astype(np.float32)
    actions = np.random.randint(0, 3, batch_size)
    rewards = np.random.randn(batch_size).astype(np.float32) * 50
    next_states = states.copy()

    loss = agent.train_batch(states, actions, rewards, next_states)
    print(f"Training loss: {loss:.4f}")

    # Test save/load
    test_path = "/tmp/test_dqn.pt"
    agent.save(test_path)

    agent2 = DQNAgent()
    agent2.load(test_path)

    # Test ONNX export
    onnx_path = "/tmp/test_dqn.onnx"
    agent.export_onnx(onnx_path)

    # Verify ONNX
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model verified!")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_agent()
