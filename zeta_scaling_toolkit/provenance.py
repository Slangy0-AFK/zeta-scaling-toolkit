"""Experiment provenance tracking for reproducibility and verification.

Records immutable experiment manifests containing Git commit, source
fingerprints, complete dependency versions, environment details, and
experiment configuration. Manifests enable detection of incompatible
state changes and verification of exact reproducibility conditions.
"""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExperimentManifest:
    """Immutable experiment provenance and configuration.

    Captures everything required to verify reproducibility and detect
    incompatible changes:
    - Source code fingerprints (full toolkit + benchmarks)
    - Git commit SHA
    - Python, PyTorch, and OS versions
    - CPU and GPU details (when available)
    - Precision and optimization settings
    - Random seeds for determinism
    - Experiment configuration (steps, batch size, learning rate, etc.)
    - Training budget and hardware constraints

    A resumed experiment must have a matching manifest or explicitly
    override with strict=False.
    """

    # Manifest version: increment when schema changes incompatibly
    _MANIFEST_VERSION = 1

    # Source and provenance
    protocol_version: int = 1
    git_commit_sha: str = ""
    toolkit_source_digest: str = ""
    benchmark_source_digest: str = ""
    source_digest_algorithm: str = "sha256"

    # Python and dependencies
    python_version: str = ""
    python_implementation: str = ""
    pytorch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""

    # Operating system and hardware
    os_name: str = ""
    os_version: str = ""
    platform_string: str = ""
    cpu_count: int = 0
    cpu_model: str = ""
    gpu_name: str = ""
    gpu_compute_capability: tuple[int, int] | None = None

    # Precision and optimization
    pytorch_dtype_default: str = ""
    cuda_enabled: bool = False
    cudnn_enabled: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False

    # Random seeds and determinism
    seed: int = 0
    torch_seed: int = 0
    random_seed: int = 0

    # Experiment configuration
    experiment_config: dict[str, Any] = field(default_factory=dict)
    training_budget_steps: int = 0
    training_batch_size: int = 0

    @classmethod
    def from_environment(
        cls,
        git_commit_sha: str = "",
        toolkit_source_digest: str = "",
        benchmark_source_digest: str = "",
        seed: int = 0,
        experiment_config: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        """Create manifest from current environment and configuration.

        Args:
            git_commit_sha: Git commit SHA (empty if not in repo)
            toolkit_source_digest: SHA256 of toolkit source files
            benchmark_source_digest: SHA256 of benchmark source files
            seed: Random seed used for experiment
            experiment_config: Experiment configuration dict

        Returns:
            Manifest capturing current environment and provided config.
        """
        import subprocess

        if not git_commit_sha:
            try:
                git_commit_sha = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=Path(__file__).parent.parent,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                git_commit_sha = ""

        cuda_version = ""
        gpu_name = ""
        gpu_compute_capability = None
        if torch.cuda.is_available():
            try:
                cuda_version = torch.version.cuda or ""
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_compute_capability = (
                    props.major,
                    props.minor,
                )
            except Exception:
                pass

        torch_dtype_default = str(torch.get_default_dtype())

        return cls(
            git_commit_sha=git_commit_sha,
            toolkit_source_digest=toolkit_source_digest,
            benchmark_source_digest=benchmark_source_digest,
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            pytorch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            os_name=platform.system(),
            os_version=platform.release(),
            platform_string=platform.platform(),
            cpu_count=os.cpu_count() or 0,
            cpu_model=_get_cpu_model(),
            gpu_name=gpu_name,
            gpu_compute_capability=gpu_compute_capability,
            pytorch_dtype_default=torch_dtype_default,
            cuda_enabled=torch.cuda.is_available(),
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            seed=seed,
            torch_seed=seed,
            random_seed=seed,
            experiment_config=experiment_config or {},
        )

    def matches(self, other: ExperimentManifest, strict: bool = True) -> bool:
        """Check if another manifest is compatible.

        Args:
            other: Manifest to compare against
            strict: If True, all fields must match exactly. If False,
                only critical fields (source digests, versions) are checked.

        Returns:
            True if manifests are compatible, False otherwise.
        """
        if self.protocol_version != other.protocol_version:
            return False

        # Critical: source code and version compatibility
        if self.toolkit_source_digest != other.toolkit_source_digest:
            return False
        if self.benchmark_source_digest != other.benchmark_source_digest:
            return False
        if self.python_version != other.python_version:
            return False
        if self.pytorch_version != other.pytorch_version:
            return False

        if strict:
            # Strict: environment must be identical
            return (
                self.git_commit_sha == other.git_commit_sha
                and self.os_name == other.os_name
                and self.cuda_available == other.cuda_available
                and self.seed == other.seed
                and self.experiment_config == other.experiment_config
            )

        return True

    def as_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary (for JSON serialization, etc.)."""
        result = {
            "protocol_version": self.protocol_version,
            "git_commit_sha": self.git_commit_sha,
            "toolkit_source_digest": self.toolkit_source_digest,
            "benchmark_source_digest": self.benchmark_source_digest,
            "source_digest_algorithm": self.source_digest_algorithm,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "pytorch_version": self.pytorch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "platform_string": self.platform_string,
            "cpu_count": self.cpu_count,
            "cpu_model": self.cpu_model,
            "gpu_name": self.gpu_name,
            "pytorch_dtype_default": self.pytorch_dtype_default,
            "cuda_enabled": self.cuda_enabled,
            "cudnn_enabled": self.cudnn_enabled,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "seed": self.seed,
            "torch_seed": self.torch_seed,
            "random_seed": self.random_seed,
            "experiment_config": self.experiment_config,
            "training_budget_steps": self.training_budget_steps,
            "training_batch_size": self.training_batch_size,
        }
        if self.gpu_compute_capability is not None:
            result["gpu_compute_capability"] = list(self.gpu_compute_capability)
        return result


def compute_source_digest(paths: list[Path]) -> str:
    """Compute SHA256 digest of source files.

    Args:
        paths: List of file paths to include in digest

    Returns:
        Hex SHA256 digest of concatenated file contents
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def compute_toolkit_source_digest() -> str:
    """Compute digest of toolkit source files."""
    toolkit_dir = Path(__file__).parent
    files = [
        toolkit_dir / "core.py",
        toolkit_dir / "positions.py",
        toolkit_dir / "checkpoint.py",
        toolkit_dir / "provenance.py",
    ]
    return compute_source_digest([f for f in files if f.exists()])


def _get_cpu_model() -> str:
    """Get CPU model name from system info."""
    try:
        import subprocess

        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # Fallback for macOS and Windows
    return platform.processor() or platform.uname().processor or "unknown"
