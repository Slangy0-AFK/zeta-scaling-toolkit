"""Tests for experiment provenance tracking and validation.

Verifies that:
1. Manifests capture complete environment and configuration
2. Manifests are created from current environment correctly
3. Manifest compatibility checking is strict and correct
4. Source digest computation is deterministic
5. Manifests reject incompatible changes
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from zeta_scaling_toolkit.provenance import (
    ExperimentManifest,
    compute_source_digest,
    compute_toolkit_source_digest,
)


class TestExperimentManifestCreation:
    """Test manifest creation from environment."""

    def test_manifest_from_environment_captures_versions(self) -> None:
        """Verify manifest captures Python and PyTorch versions."""
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
            seed=42,
        )

        assert manifest.python_version
        assert manifest.pytorch_version
        assert manifest.seed == 42
        assert manifest.torch_seed == 42
        assert manifest.random_seed == 42

    def test_manifest_captures_os_info(self) -> None:
        """Verify manifest captures OS information."""
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
        )

        assert manifest.os_name
        assert manifest.platform_string
        assert manifest.cpu_count >= 1

    def test_manifest_captures_cuda_state(self) -> None:
        """Verify manifest captures CUDA availability."""
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
        )

        assert isinstance(manifest.cuda_available, bool)
        # cuda_enabled should match cuda_available (unless explicitly disabled)
        assert isinstance(manifest.cuda_enabled, bool)

    def test_manifest_captures_torch_settings(self) -> None:
        """Verify manifest captures PyTorch backend settings."""
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
        )

        assert isinstance(manifest.cudnn_enabled, bool)
        assert isinstance(manifest.cudnn_deterministic, bool)
        assert isinstance(manifest.cudnn_benchmark, bool)
        assert manifest.pytorch_dtype_default

    def test_manifest_captures_experiment_config(self) -> None:
        """Verify manifest stores experiment configuration."""
        config = {"steps": 1000, "batch_size": 32, "learning_rate": 0.001}
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
            experiment_config=config,
        )

        assert manifest.experiment_config == config

    def test_manifest_protocol_version_set(self) -> None:
        """Verify manifest protocol version is recorded."""
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
        )

        assert manifest.protocol_version == ExperimentManifest._MANIFEST_VERSION

    def test_manifest_as_dict(self) -> None:
        """Verify manifest can be converted to dict."""
        manifest = ExperimentManifest.from_environment(
            toolkit_source_digest="abc123",
            benchmark_source_digest="def456",
            experiment_config={"steps": 500},
        )

        d = manifest.as_dict()

        assert isinstance(d, dict)
        assert d["protocol_version"] == ExperimentManifest._MANIFEST_VERSION
        assert d["toolkit_source_digest"] == "abc123"
        assert d["benchmark_source_digest"] == "def456"
        assert d["experiment_config"] == {"steps": 500}


class TestManifestComparison:
    """Test manifest compatibility checking."""

    def test_manifests_with_different_protocol_versions_incompatible(self) -> None:
        """Verify different protocol versions are incompatible."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )
        m2 = ExperimentManifest(
            protocol_version=2,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )

        assert not m1.matches(m2)
        assert not m2.matches(m1)

    def test_manifests_with_different_toolkit_digest_incompatible(self) -> None:
        """Verify different toolkit source digests are incompatible."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="xyz",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )

        assert not m1.matches(m2)

    def test_manifests_with_different_benchmark_digest_incompatible(self) -> None:
        """Verify different benchmark source digests are incompatible."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="ghi",
            python_version="3.10",
            pytorch_version="2.0.0",
        )

        assert not m1.matches(m2)

    def test_manifests_with_different_python_version_incompatible(self) -> None:
        """Verify different Python versions are incompatible."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.9",
            pytorch_version="2.0.0",
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )

        assert not m1.matches(m2)

    def test_manifests_with_different_pytorch_version_incompatible(self) -> None:
        """Verify different PyTorch versions are incompatible."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.1.0",
        )

        assert not m1.matches(m2)

    def test_manifests_identical_core_match_non_strict(self) -> None:
        """Verify compatible core fields allow non-strict match."""
        m1 = ExperimentManifest(
            protocol_version=1,
            git_commit_sha="sha1",
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            os_name="Linux",
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            git_commit_sha="sha2",
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            os_name="Darwin",
        )

        # Non-strict: only core requirements match
        assert m1.matches(m2, strict=False)

    def test_manifests_identical_strict_match(self) -> None:
        """Verify identical manifests match in strict mode."""
        m1 = ExperimentManifest(
            protocol_version=1,
            git_commit_sha="sha1",
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            os_name="Linux",
            cuda_available=False,
            seed=42,
            experiment_config={"steps": 100},
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            git_commit_sha="sha1",
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            os_name="Linux",
            cuda_available=False,
            seed=42,
            experiment_config={"steps": 100},
        )

        assert m1.matches(m2, strict=True)

    def test_manifests_different_seed_incompatible_strict(self) -> None:
        """Verify different seeds are incompatible in strict mode."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            seed=42,
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            seed=43,
        )

        assert not m1.matches(m2, strict=True)

    def test_manifests_different_config_incompatible_strict(self) -> None:
        """Verify different configs are incompatible in strict mode."""
        m1 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            experiment_config={"steps": 100},
        )
        m2 = ExperimentManifest(
            protocol_version=1,
            toolkit_source_digest="abc",
            benchmark_source_digest="def",
            python_version="3.10",
            pytorch_version="2.0.0",
            experiment_config={"steps": 200},
        )

        assert not m1.matches(m2, strict=True)


class TestSourceDigestComputation:
    """Test source digest computation."""

    def test_compute_source_digest_deterministic(self) -> None:
        """Verify source digest is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            file1 = tmpdir_path / "file1.py"
            file1.write_text("x = 1")

            digest1 = compute_source_digest([file1])
            digest2 = compute_source_digest([file1])

            assert digest1 == digest2
            assert len(digest1) == 64  # SHA256 hex string

    def test_compute_source_digest_different_content(self) -> None:
        """Verify different content produces different digests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            file1 = tmpdir_path / "file1.py"
            file1.write_text("x = 1")
            file2 = tmpdir_path / "file2.py"
            file2.write_text("x = 2")

            digest1 = compute_source_digest([file1])
            digest2 = compute_source_digest([file2])

            assert digest1 != digest2

    def test_compute_source_digest_multiple_files(self) -> None:
        """Verify digest includes all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            file1 = tmpdir_path / "file1.py"
            file1.write_text("x = 1")
            file2 = tmpdir_path / "file2.py"
            file2.write_text("y = 2")

            digest1 = compute_source_digest([file1])
            digest2 = compute_source_digest([file1, file2])

            assert digest1 != digest2

    def test_compute_source_digest_nonexistent_file(self) -> None:
        """Verify nonexistent files are skipped (not included in digest)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            file1 = tmpdir_path / "exists.py"
            file1.write_text("x = 1")
            file2 = tmpdir_path / "nonexistent.py"

            # Should not raise, just skip nonexistent files
            digest = compute_source_digest([file1, file2])
            assert len(digest) == 64

    def test_compute_toolkit_source_digest(self) -> None:
        """Verify toolkit source digest can be computed."""
        digest = compute_toolkit_source_digest()
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
