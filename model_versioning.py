#!/usr/bin/env python3
"""
Model Versioning and Registry for RL Trading Agent

Provides safe model deployments with:
- Versioned model storage
- Atomic promotion to production
- Rollback capability
- Metadata tracking (performance metrics, training info)
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ModelRegistry:
    """
    Version-controlled model registry with atomic updates and rollback.

    Directory Structure:
        model_registry/
        ├── versions/
        │   ├── v20260206_140000/
        │   │   ├── rl_q_table.pkl
        │   │   ├── rl_policy.json
        │   │   └── metadata.json
        │   └── v20260206_153000/
        │       └── ...
        ├── current -> versions/v20260206_153000/  (symlink)
        ├── staging/
        └── archive/
    """

    def __init__(self, base_path: str = "model_registry"):
        self.base_path = Path(base_path)
        self.versions_dir = self.base_path / "versions"
        self.staging_dir = self.base_path / "staging"
        self.archive_dir = self.base_path / "archive"
        self.current_link = self.base_path / "current"

        # Create directory structure
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)

        print(f"[ModelRegistry] Initialized at {self.base_path}")

    def save_model(
        self,
        agent,
        metadata: Dict[str, Any],
        stage: str = "staging"
    ) -> str:
        """
        Save model to staging or versioned directory.

        Supports both Q-learning and DQN agents:
        - Q-learning: saves rl_q_table.pkl and rl_policy.json
        - DQN: saves dqn_checkpoint.pt, dqn_model.onnx, and scaler.json

        Args:
            agent: QLearningAgent or DQNAgent instance
            metadata: Dict with evaluation metrics, training info, etc.
            stage: "staging" or "production" (where to save)

        Returns:
            Version name (e.g., "v20260206_140000")
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version_name = f"v{timestamp}"

        if stage == "staging":
            model_dir = self.staging_dir / version_name
        else:
            model_dir = self.versions_dir / version_name

        model_dir.mkdir(parents=True, exist_ok=True)

        # Detect model type
        is_dqn = hasattr(agent, 'policy_net')

        if is_dqn:
            # Save DQN checkpoint (PyTorch format)
            checkpoint_path = model_dir / "dqn_checkpoint.pt"
            agent.save(str(checkpoint_path))

            # Export ONNX for Rust inference
            onnx_path = model_dir / "dqn_model.onnx"
            agent.export_onnx(str(onnx_path))

            # Save scaler params for Rust normalization
            scaler_path = model_dir / "scaler.json"
            agent.save_scaler(str(scaler_path))

            # Also copy to model/{asset}/ for live bot to pick up
            asset = self.base_path.name
            live_model_dir = Path(f"model/{asset}")
            live_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(str(onnx_path), str(live_model_dir / "dqn_model.onnx"))
            shutil.copy(str(scaler_path), str(live_model_dir / "scaler.json"))

            # Copy ONNX data file if it exists (external data storage)
            onnx_data_path = model_dir / "dqn_model.onnx.data"
            if onnx_data_path.exists():
                shutil.copy(str(onnx_data_path), str(live_model_dir / "dqn_model.onnx.data"))

            # Metadata for DQN
            metadata["version"] = version_name
            metadata["timestamp"] = timestamp
            metadata["created_at"] = datetime.utcnow().isoformat()
            metadata["model_type"] = "dqn"
            metadata["train_steps"] = agent.train_steps
            metadata["epsilon"] = agent.epsilon

            print(f"[ModelRegistry] Saved DQN model {version_name} to {stage}/")
            print(f"  Train steps: {metadata['train_steps']}")
            print(f"  Epsilon: {metadata['epsilon']:.4f}")

        else:
            # Save Q-table (pickle format)
            q_table_path = model_dir / "rl_q_table.pkl"
            agent.save(str(q_table_path))

            # Save policy (JSON format for Rust)
            policy_path = model_dir / "rl_policy.json"
            agent.save_json(str(policy_path))

            # Metadata for Q-learning
            metadata["version"] = version_name
            metadata["timestamp"] = timestamp
            metadata["created_at"] = datetime.utcnow().isoformat()
            metadata["model_type"] = "q_learning"
            metadata["q_states"] = len(agent.q_table)
            metadata["epsilon"] = agent.epsilon

            print(f"[ModelRegistry] Saved model {version_name} to {stage}/")
            print(f"  Q-table states: {metadata['q_states']}")
            print(f"  Epsilon: {metadata['epsilon']:.4f}")

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return version_name

    def promote_to_production(self, version_name: str):
        """
        Atomically promote a model from staging to production.

        Args:
            version_name: Version to promote (must exist in staging/)

        Raises:
            ValueError: If version not found
        """
        # Check if in staging
        staging_path = self.staging_dir / version_name
        production_path = self.versions_dir / version_name

        if not staging_path.exists():
            # Maybe it's already in production
            if not production_path.exists():
                raise ValueError(f"Version {version_name} not found in staging/ or versions/")
            print(f"[ModelRegistry] Version {version_name} already in production")
        else:
            # Move from staging to versions
            shutil.move(str(staging_path), str(production_path))
            print(f"[ModelRegistry] Moved {version_name} from staging to versions/")

        # Atomic symlink update
        temp_link = self.current_link.with_suffix('.tmp')

        # Create new symlink pointing to the target
        if temp_link.exists():
            temp_link.unlink()

        # Use relative path for symlink
        relative_path = production_path.relative_to(self.base_path)
        temp_link.symlink_to(relative_path)

        # Atomic replace
        if self.current_link.exists():
            self.current_link.unlink()
        temp_link.rename(self.current_link)

        print(f"[ModelRegistry] ✅ Promoted {version_name} to production (current)")

    def rollback(self) -> Optional[str]:
        """
        Rollback to the previous version.

        Returns:
            Previous version name if successful, None otherwise
        """
        current = self.get_current_version()
        if not current:
            print("[ModelRegistry] No current version to rollback from")
            return None

        # Get all versions sorted by name (timestamp-based)
        versions = sorted([d.name for d in self.versions_dir.iterdir() if d.is_dir()])

        if current not in versions:
            print(f"[ModelRegistry] Current version {current} not found in versions/")
            return None

        current_idx = versions.index(current)
        if current_idx == 0:
            print("[ModelRegistry] Already at oldest version, cannot rollback further")
            return None

        # Rollback to previous
        previous = versions[current_idx - 1]
        print(f"[ModelRegistry] Rolling back from {current} to {previous}")

        self.promote_to_production(previous)
        return previous

    def get_current_version(self) -> Optional[str]:
        """
        Get the currently active version name.

        Returns:
            Version name (e.g., "v20260206_140000") or None if no model deployed
        """
        if not self.current_link.exists():
            return None

        # Resolve symlink to get target directory name
        target = self.current_link.resolve()
        return target.name

    def get_metadata(self, version_name: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a specific version"""
        metadata_path = self.versions_dir / version_name / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_versions(self) -> list:
        """List all versioned models with their metadata"""
        versions = []
        for version_dir in sorted(self.versions_dir.iterdir(), reverse=True):
            if version_dir.is_dir():
                metadata = self.get_metadata(version_dir.name)
                versions.append({
                    "version": version_dir.name,
                    "metadata": metadata
                })
        return versions

    def archive_old_versions(self, keep_latest_n: int = 10):
        """Archive old versions to save disk space"""
        versions = sorted(
            [d for d in self.versions_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True
        )

        current = self.get_current_version()

        archived = 0
        for version_dir in versions[keep_latest_n:]:
            # Never archive current version
            if version_dir.name == current:
                continue

            archive_path = self.archive_dir / version_dir.name
            shutil.move(str(version_dir), str(archive_path))
            archived += 1

        if archived > 0:
            print(f"[ModelRegistry] Archived {archived} old versions")

        return archived


def main():
    """Test the model registry"""
    from rl_strategy import QLearningAgent

    registry = ModelRegistry()

    # Create a test agent
    print("\nCreating test agent...")
    agent = QLearningAgent()

    # Simulate some training
    agent.q_table[(1, 2, 3, 0, 0)] = [0.0, 10.0, -5.0]
    agent.q_table[(2, 3, 4, 1, 1)] = [-2.0, 5.0, 8.0]

    # Save to staging
    print("\nSaving model to staging...")
    metadata = {
        "evaluation": {
            "total_pnl": 250.50,
            "win_rate": 0.62,
            "num_trades": 15,
            "sharpe_ratio": 1.8
        },
        "n_episodes": 50,
        "training_duration_sec": 120
    }

    version = registry.save_model(agent, metadata, stage="staging")

    # Promote to production
    print(f"\nPromoting {version} to production...")
    registry.promote_to_production(version)

    # Check current version
    current = registry.get_current_version()
    print(f"\nCurrent production version: {current}")

    # List all versions
    print("\nAll versions:")
    for v in registry.list_versions():
        print(f"  {v['version']}: {v['metadata'].get('evaluation', {}).get('win_rate', 'N/A')}")

    # Test rollback
    print("\nTesting rollback...")
    previous = registry.rollback()
    if previous:
        print(f"Rolled back to: {previous}")
    else:
        print("Rollback not available (only one version)")


if __name__ == "__main__":
    main()
