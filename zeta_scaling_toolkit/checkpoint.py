"""Trial checkpoint serialization and recovery for mid-trial resumption.

This module provides atomic, versioned checkpoint management with full RNG
state preservation (Python, PyTorch CPU, and CUDA where available). Checkpoints
enable safe recovery from interruptions without losing training progress or
replicating records.
"""

from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.optim
import torch.optim.lr_scheduler


@dataclass
class TrialCheckpoint:
    """Immutable trial checkpoint preserving exact training state.

    A checkpoint captures everything needed to resume training from the exact
    next step without duplication or divergence:
    - Model weights and buffers (state_dict)
    - Optimizer state (momentum, adaptive learning rates, etc.)
    - LR scheduler state (step counter, running averages)
    - Current training step number
    - Complete RNG state for reproducibility (Python, PyTorch CPU, CUDA)
    - Checkpoint format/protocol versioning
    - Experiment identifier for validation

    All RNG states are pickled directly to preserve exact internal structure,
    not reconstructed from seeds, ensuring bit-identical resumption.
    """

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    current_step: int
    python_rng_state: tuple[int, tuple[int, ...], float | None]
    pytorch_cpu_rng_state: torch.Tensor
    pytorch_cuda_rng_state: torch.Tensor | None = None
    experiment_id: str = ""
    checkpoint_format_version: int = 1

    # Class constant for format version
    _FORMAT_VERSION = 1

    @classmethod
    def capture(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        current_step: int,
        experiment_id: str = "",
    ) -> TrialCheckpoint:
        """Capture current training state into a checkpoint.

        Args:
            model: Trainable model module
            optimizer: Optimizer managing model parameters
            scheduler: LR scheduler, or None if not used
            current_step: Current training step (0-indexed)
            experiment_id: Experiment identifier for validation on resume

        Returns:
            Checkpoint capturing exact current state including RNG state.
        """
        pytorch_cuda_rng_state = None
        if torch.cuda.is_available():
            pytorch_cuda_rng_state = torch.cuda.get_rng_state()

        return cls(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else {},
            current_step=current_step,
            python_rng_state=random.getstate(),
            pytorch_cpu_rng_state=torch.get_rng_state(),
            pytorch_cuda_rng_state=pytorch_cuda_rng_state,
            experiment_id=experiment_id,
            checkpoint_format_version=cls._FORMAT_VERSION,
        )

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ) -> int:
        """Restore training state from checkpoint.

        Restores model weights, optimizer state, scheduler state, and all RNG
        state to their checkpoint values. Resume training from returned step + 1.

        Args:
            model: Trainable model module (will be modified in-place)
            optimizer: Optimizer managing model parameters (will be modified)
            scheduler: LR scheduler to restore, or None if not used

        Returns:
            The next training step to execute (current_step + 1).

        Raises:
            RuntimeError: If checkpoint format version is unsupported.
        """
        if self.checkpoint_format_version != self._FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported checkpoint format version {self.checkpoint_format_version}; "
                f"expected {self._FORMAT_VERSION}"
            )

        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)
        if scheduler is not None and self.scheduler_state_dict:
            scheduler.load_state_dict(self.scheduler_state_dict)

        random.setstate(self.python_rng_state)
        torch.set_rng_state(self.pytorch_cpu_rng_state)
        if self.pytorch_cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(self.pytorch_cuda_rng_state)

        return self.current_step + 1

    def compute_digest(self) -> str:
        """Compute SHA256 digest of checkpoint contents for integrity verification.

        Returns:
            Hex SHA256 digest of pickled checkpoint data.
        """
        return hashlib.sha256(pickle.dumps(self)).hexdigest()


def save_checkpoint_atomic(
    checkpoint: TrialCheckpoint,
    path: Path,
) -> None:
    """Write checkpoint atomically to disk.

    Writes checkpoint to a temporary file first, then renames it into place.
    This prevents corruption if the process is interrupted mid-write.

    Args:
        checkpoint: Checkpoint to save
        path: Destination file path

    Raises:
        OSError: If write or rename fails
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    try:
        with open(temp_path, "wb") as f:
            pickle.dump(checkpoint, f)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_checkpoint(path: Path) -> TrialCheckpoint:
    """Load checkpoint from disk.

    Args:
        path: Path to checkpoint file

    Returns:
        Loaded checkpoint

    Raises:
        FileNotFoundError: If checkpoint does not exist
        pickle.UnpicklingError: If checkpoint is corrupted
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
        if not isinstance(checkpoint, TrialCheckpoint):
            raise TypeError(f"Expected TrialCheckpoint, got {type(checkpoint).__name__}")
        return checkpoint
