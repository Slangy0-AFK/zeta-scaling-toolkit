"""Static, deterministic adapter used only for run-level audit events."""

from __future__ import annotations


class ExperimentAuditAdapter:
    version = "zeta-fresh-run-audit-v1"

    def retrieve(self, query: str) -> list[str]:
        return [
            "This deterministic source records Zeta experiment run metadata and "
            f"lifecycle audit details. Request: {query}"
        ]

    def generate(self, query: str, chunks: list[str]) -> str:
        return chunks[0]


def create_adapter() -> ExperimentAuditAdapter:
    return ExperimentAuditAdapter()
