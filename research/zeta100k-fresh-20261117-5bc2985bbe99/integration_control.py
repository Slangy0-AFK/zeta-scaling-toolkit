"""Run-level integration calls; intentionally separate from Zeta inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_ID = "zeta100k-fresh-20261117-5bc2985bbe99"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
ACTION = "zeta_100k_pipeline"
TIMEOUT_SECONDS = 3.0


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"request to {path} failed: {type(error).__name__}") from None
    return data, time.perf_counter() - started


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable {name} is missing")
    return value


def call_mathmesh(run_id: str, phase: str, run_dir: Path) -> dict[str, Any]:
    base = require_env("MATHMESH_URL")
    health, health_seconds = request_json(base, "/api/health")
    if health.get("ok") is not True:
        raise RuntimeError("MathMesh health check failed")
    login, login_seconds = request_json(
        base,
        "/api/auth/login",
        method="POST",
        payload={
            "email": require_env("MATHMESH_EMAIL"),
            "password": require_env("MATHMESH_PASSWORD"),
        },
    )
    token = login.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("MathMesh login returned no access token")
    auth = {"Authorization": f"Bearer {token}"}

    if phase == "admit":
        agent, create_agent_seconds = request_json(
            base,
            "/api/entities/AgentIdentity/create",
            method="POST",
            headers=auth,
            payload={
                "agent_id": run_id,
                "role": "reader",
                "status": "active",
                "allowed_actions": [ACTION],
                "session_nonce": run_id,
            },
        )
        issued, issue_key_seconds = request_json(
            base,
            "/api/functions/issueAgentKey",
            method="POST",
            headers=auth,
            payload={"agentId": run_id},
        )
        key = issued.get("key")
        if not isinstance(key, str) or not key:
            raise RuntimeError("MathMesh did not issue the run admission key")
        verdict, admission_seconds = request_json(
            base,
            "/api/functions/admissionControl",
            method="POST",
            headers=auth,
            payload={"agentId": run_id, "action": ACTION, "agentKey": key, "sessionNonce": run_id},
        )
        if verdict.get("admitted") is not True:
            raise RuntimeError(
                f"MathMesh denied run admission at {verdict.get('haltedAt', 'unknown gate')}"
            )
        pipeline, metadata_seconds = request_json(
            base,
            "/api/entities/PipelineRun/create",
            method="POST",
            headers=auth,
            payload={
                "run_id": run_id,
                "status": "admitted",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "phase": phase,
                "zeta_configuration_sha256": configuration_hash(),
            },
        )
        audit, audit_seconds = request_json(
            base,
            "/api/functions/recordAudit",
            method="POST",
            headers=auth,
            payload={
                "event_type": "GATE_PASS",
                "agent_id": run_id,
                "session_nonce": run_id,
                "gate": "zeta_run_admission",
                "action": ACTION,
                "details": "Run-level admission passed before Zeta execution; no forward-pass calls are made.",
            },
        )
        return {
            "health_seconds": health_seconds,
            "login_seconds": login_seconds,
            "agent_id": agent.get("id"),
            "create_agent_seconds": create_agent_seconds,
            "issue_key_seconds": issue_key_seconds,
            "admission_seconds": admission_seconds,
            "metadata_id": pipeline.get("id"),
            "metadata_seconds": metadata_seconds,
            "audit_sequence": audit.get("seq"),
            "audit_seconds": audit_seconds,
        }

    state = json.loads((run_dir / "integration-state.json").read_text(encoding="utf-8"))
    pipeline_id = state["services"]["mathmesh"]["metadata_id"]
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise RuntimeError("MathMesh pipeline metadata ID is unavailable")
    updated, update_seconds = request_json(
        base,
        "/api/entities/PipelineRun/update",
        method="POST",
        headers=auth,
        payload={
            "id": pipeline_id,
            "patch": {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()},
        },
    )
    audit, audit_seconds = request_json(
        base,
        "/api/functions/recordAudit",
        method="POST",
        headers=auth,
        payload={
            "event_type": "PIPELINE_COMPLETE",
            "agent_id": run_id,
            "session_nonce": run_id,
            "gate": "zeta_run_lifecycle",
            "action": ACTION,
            "details": "Zeta result artifacts remain authoritative; this is a run-level completion reference.",
        },
    )
    return {
        "health_seconds": health_seconds,
        "login_seconds": login_seconds,
        "metadata_status": updated.get("status"),
        "update_seconds": update_seconds,
        "audit_sequence": audit.get("seq"),
        "audit_seconds": audit_seconds,
    }


def call_rag(run_id: str, phase: str) -> dict[str, Any]:
    base = require_env("RAG_GOVERNOR_URL")
    key = require_env("RAG_GOVERNOR_OPERATOR_KEY")
    health, health_seconds = request_json(base, "/health")
    if health.get("ok") is not True:
        raise RuntimeError("RAG Governor audit-chain health check failed")
    request_id = f"{run_id}:{phase}"
    agent_id = run_id[:128]
    _, register_seconds = request_json(
        base,
        "/admin/agent",
        method="POST",
        headers={"X-Operator-Key": key},
        payload={"agent_id": agent_id},
    )
    _, credit_seconds = request_json(
        base,
        "/admin/agent/credit",
        method="POST",
        headers={"X-Operator-Key": key},
        payload={"agent_id": agent_id, "amount": 10},
    )
    result, govern_seconds = request_json(
        base,
        "/govern",
        method="POST",
        payload={
            "agent_id": agent_id,
            "request_id": request_id,
            "query": f"Record Zeta experiment run {run_id} lifecycle phase {phase} as audit metadata.",
        },
    )
    if result.get("released") is not True:
        raise RuntimeError(
            f"RAG Governor did not release audit event: {result.get('reason', 'unknown')}"
        )
    return {
        "health_seconds": health_seconds,
        "register_seconds": register_seconds,
        "credit_seconds": credit_seconds,
        "govern_seconds": govern_seconds,
        "request_id": request_id,
        "decision": result.get("reason"),
    }


def call_stability(run_dir: Path) -> dict[str, Any]:
    source = Path(require_env("STABILITY_TOOLKIT_PATH")) / "crash_fix.py"
    spec = importlib.util.spec_from_file_location("zeta_stability_advisor", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned System-Stability-Toolkit source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu = module.detect_cpu_cores()
    memory = module.detect_available_memory_mb()
    started = time.perf_counter()
    recommendation = module.get_recommendation(100_000, cpu_cores=cpu, available_memory_mb=memory)
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": "System-Stability-Toolkit",
        "workload_tokens": 100_000,
        "cpu_cores": cpu,
        "available_memory_mb": memory,
        "recommendation": recommendation,
        "applied_to_zeta": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    path = run_dir / "resource-recommendation.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def configuration_hash() -> str:
    config = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "seed": 20261117,
        "target_tokens": 100_000,
        "chunk_size": 64,
        "sink_cache_positions": 64,
        "recent_cache_positions": 64,
        "precision": "float32",
        "device": "cpu",
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_integration_failure(
    run_dir: Path, run_id: str, phase: str, service: str, error: Exception
) -> None:
    path = run_dir / "integration-failure.json"
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        failure_count = int(previous.get("failure_count", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        failure_count = 0
    record = {
        "run_id": run_id,
        "phase": phase,
        "service": service,
        "failure_count": failure_count + 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=run_dir, prefix=".integration-failure-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("admit", "complete"), required=True)
    parser.add_argument(
        "--services", nargs="+", choices=("mathmesh", "stability", "rag"), required=True
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.perf_counter()
    records: dict[str, Any] = {}
    current_service = "unknown"
    try:
        for service in args.services:
            current_service = service
            if service == "mathmesh":
                records[service] = call_mathmesh(args.run_id, args.phase, run_dir)
            elif service == "stability":
                records[service] = call_stability(run_dir)
            else:
                if args.phase != "admit":
                    continue
                records[service] = call_rag(args.run_id, args.phase)
    except Exception as error:
        record_integration_failure(run_dir, args.run_id, args.phase, current_service, error)
        raise
    elapsed = time.perf_counter() - started
    state_path = run_dir / "integration-state.json"
    if args.phase == "admit":
        state = {
            "run_id": args.run_id,
            "phase": args.phase,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": configuration_hash(),
            "services": records,
            "elapsed_seconds": elapsed,
        }
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("completion", {}).update(records)
        state["completion"]["elapsed_seconds"] = elapsed
        state["completion"]["recorded_at"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "phase": args.phase,
                "services": records,
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
