"""Bounded, asynchronous observer for Zeta's append-only progress JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import resource
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

MAX_PROGRESS_LINE_BYTES = 4096
MAX_EVENT_PATH_CHARS = 256
DEFAULT_QUEUE_CAPACITY = 128
HTTP_TIMEOUT_SECONDS = 1.0


class TelemetryHTTPError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"http_{status}")
        self.status = status


def _read_json_response(response: Any) -> dict[str, Any]:
    body = response.read(4097)
    if len(body) > 4096:
        raise ValueError("telemetry response exceeded 4096 bytes")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("telemetry response must be a JSON object")
    return value


def _request_json(
    url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return _read_json_response(response)
    except urllib.error.HTTPError as error:
        error.close()
        raise TelemetryHTTPError(error.code) from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(type(error).__name__) from None


def normalize_progress(entry: dict[str, Any]) -> dict[str, Any]:
    processed = entry.get("processed_tokens")
    last_position = entry.get("last_position")
    if (
        isinstance(processed, bool)
        or not isinstance(processed, int)
        or not 1 <= processed <= 100_000
        or entry.get("first_position") != 0
        or last_position != processed - 1
    ):
        raise ValueError("invalid token progression fields")
    stream_hash = entry.get("stream_sha256")
    if not isinstance(stream_hash, str) or len(stream_hash) != 64:
        raise ValueError("invalid stream hash")
    try:
        int(stream_hash, 16)
    except ValueError:
        raise ValueError("invalid stream hash") from None
    checkpoint = entry.get("checkpoint")
    if not isinstance(checkpoint, str) or len(checkpoint) > MAX_EVENT_PATH_CHARS:
        raise ValueError("invalid checkpoint reference")
    retained = entry.get("retained_positions")
    if (
        not isinstance(retained, list)
        or len(retained) > 128
        or any(isinstance(value, bool) or not isinstance(value, int) for value in retained)
    ):
        raise ValueError("invalid retained positions")

    event = {
        "processed_tokens": processed,
        "first_position": 0,
        "last_position": last_position,
        "stream_sha256": stream_hash,
        "checkpoint": checkpoint,
        "retained_positions": retained,
        "cache_bytes": _nonnegative_int(entry.get("cache_bytes")),
        "tokens_per_second": _finite_nonnegative(entry.get("tokens_per_second")),
        "available_memory_mib": _finite_nonnegative(entry.get("available_memory_mib")),
        "peak_rss_mib": _finite_nonnegative(entry.get("peak_rss_mib")),
    }
    return event


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid integer telemetry field")
    return value


def _counter(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _finite_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid numeric telemetry field")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid numeric telemetry field")
    return result


class MathMeshProgressSender:
    """Publish the latest checkpoint summary; never reads or writes Zeta state."""

    def __init__(self, base_url: str, email: str, password: str, pipeline_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.pipeline_id = pipeline_id
        self._token: str | None = None

    def _login(self) -> str:
        result = _request_json(
            f"{self.base_url}/api/auth/login",
            {"email": self.email, "password": self.password},
        )
        token = result.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("MathMesh login returned no token")
        self._token = token
        return token

    def __call__(self, event: dict[str, Any]) -> None:
        token = self._token or self._login()
        payload = {
            "id": self.pipeline_id,
            "patch": {
                "telemetry_processed_tokens": event["processed_tokens"],
                "telemetry_last_position": event["last_position"],
                "telemetry_stream_sha256": event["stream_sha256"],
                "telemetry_checkpoint": event["checkpoint"],
                "telemetry_cache_bytes": event["cache_bytes"],
                "telemetry_available_memory_mib": event["available_memory_mib"],
                "telemetry_peak_rss_mib": event["peak_rss_mib"],
                "telemetry_tokens_per_second": event["tokens_per_second"],
            },
        }
        try:
            _request_json(
                f"{self.base_url}/api/entities/PipelineRun/update",
                payload,
                {"Authorization": f"Bearer {token}"},
            )
        except TelemetryHTTPError as error:
            if error.status != 401:
                raise
            token = self._login()
            _request_json(
                f"{self.base_url}/api/entities/PipelineRun/update",
                payload,
                {"Authorization": f"Bearer {token}"},
            )


class ProgressTelemetry:
    """Tail local progress in one thread and deliver summaries in another."""

    def __init__(
        self,
        progress_path: Path,
        status_path: Path,
        sender: Callable[[dict[str, Any]], None] | None,
        *,
        queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
        poll_seconds: float = 0.25,
    ) -> None:
        if not 1 <= queue_capacity <= DEFAULT_QUEUE_CAPACITY or poll_seconds <= 0:
            raise ValueError(
                "queue capacity must be 1 through 128 and poll interval must be positive"
            )
        self.progress_path = progress_path
        self.status_path = status_path
        self.sender = sender
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_capacity)
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.offset = 0
        previous = self._read_previous_status()
        self.state: dict[str, Any] = {
            "queue_capacity": queue_capacity,
            "queue_high_water": _counter(previous.get("queue_high_water")),
            "queued": _counter(previous.get("queued")),
            "delivered": _counter(previous.get("delivered")),
            "failed_requests": _counter(previous.get("failed_requests")),
            "retries": _counter(previous.get("retries")),
            "dropped": _counter(previous.get("dropped")),
            "malformed_records": _counter(previous.get("malformed_records")),
            "backpressure_events": _counter(previous.get("backpressure_events")),
            "last_seen_tokens": 0,
            "last_delivered_tokens": _counter(previous.get("last_delivered_tokens")),
            "last_failed_tokens": previous.get("last_failed_tokens"),
            "last_failure": previous.get("last_failure"),
            "last_deferred_tokens": previous.get("last_deferred_tokens"),
            "last_error": None,
            "mode": "local_only" if sender is None else "mathmesh_async",
            "process_peak_rss_mib": 0.0,
            "process_cpu_seconds": 0.0,
        }
        self.last_seen_tokens = self.state["last_delivered_tokens"]
        self.reader_thread = threading.Thread(
            target=self._read_loop, name="zeta-progress-reader", daemon=True
        )
        self.worker_thread = threading.Thread(
            target=self._send_loop, name="zeta-progress-sender", daemon=True
        )
        self.started_at = 0.0

    def _read_previous_status(self) -> dict[str, Any]:
        try:
            value = json.loads(self.status_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError, AttributeError):
            return {}

    def _write_status(self) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        with self.lock:
            self.state["queue_depth"] = self.queue.qsize()
            self.state["last_seen_tokens"] = self.last_seen_tokens
            self.state["elapsed_seconds"] = (
                max(0.0, time.monotonic() - self.started_at) if self.started_at else 0.0
            )
            self.state["process_peak_rss_mib"] = usage.ru_maxrss / 1024
            self.state["process_cpu_seconds"] = usage.ru_utime + usage.ru_stime
            payload = json.dumps(self.state, sort_keys=True, indent=2) + "\n"
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.status_path.parent,
                    prefix=".telemetry-",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.status_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def _increment(self, key: str, value: int = 1) -> None:
        with self.lock:
            self.state[key] += value

    def poll_once(self) -> int:
        if not self.progress_path.exists():
            return 0
        accepted = 0
        with self.progress_path.open("rb") as stream:
            stream.seek(self.offset)
            while True:
                line_start = stream.tell()
                raw = stream.readline(MAX_PROGRESS_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_PROGRESS_LINE_BYTES:
                    while not raw.endswith(b"\n"):
                        raw = stream.readline(MAX_PROGRESS_LINE_BYTES + 1)
                        if not raw:
                            break
                    if not raw:
                        break
                    self.offset = stream.tell()
                    self._increment("malformed_records")
                    continue
                if not raw.endswith(b"\n"):
                    stream.seek(line_start)
                    break
                self.offset = stream.tell()
                try:
                    decoded = json.loads(raw)
                    if not isinstance(decoded, dict):
                        raise ValueError("progress record must be an object")
                    event = normalize_progress(decoded)
                except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                    self._increment("malformed_records")
                    continue
                if event["processed_tokens"] <= self.last_seen_tokens:
                    continue
                try:
                    self.queue.put_nowait(event)
                except queue.Full:
                    self.offset = line_start
                    with self.lock:
                        self.state["backpressure_events"] += 1
                        self.state["last_deferred_tokens"] = event["processed_tokens"]
                    break
                self.last_seen_tokens = event["processed_tokens"]
                accepted += 1
                self._increment("queued")
                with self.lock:
                    self.state["queue_high_water"] = max(
                        self.state["queue_high_water"], self.queue.qsize()
                    )
        self._write_status()
        return accepted

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            self.poll_once()
            self.stop_event.wait(self.poll_seconds)

    def _send_loop(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            attempt = 0
            delivered = False
            while True:
                try:
                    if self.sender is not None:
                        self.sender(event)
                    with self.lock:
                        self.state["delivered"] += 1
                        self.state["last_delivered_tokens"] = event["processed_tokens"]
                        self.state["last_error"] = None
                    delivered = True
                    break
                except Exception as error:
                    attempt += 1
                    with self.lock:
                        self.state["failed_requests"] += 1
                        self.state["retries"] += 1
                        self.state["last_failed_tokens"] = event["processed_tokens"]
                        self.state["last_error"] = type(error).__name__
                        self.state["last_failure"] = type(error).__name__
                    self._write_status()
                    if self.stop_event.wait(min(0.1 * (2 ** min(attempt - 1, 4)), 1.0)):
                        break
            self.queue.task_done()
            self._write_status()
            if not delivered:
                return

    def start(self) -> None:
        if self.started_at:
            raise RuntimeError("telemetry already started")
        self.started_at = time.monotonic()
        self.worker_thread.start()
        self.reader_thread.start()
        self._write_status()

    def wait_until_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.queue.unfinished_tasks == 0:
                return True
            self.stop_event.wait(0.01)
        return self.queue.unfinished_tasks == 0

    def stop(self, join_seconds: float = 2.0) -> None:
        self.stop_event.set()
        if self.reader_thread.is_alive():
            self.reader_thread.join(join_seconds)
        if self.worker_thread.is_alive():
            self.worker_thread.join(join_seconds)
        self._write_status()

    def run_once(self, timeout: float = 5.0) -> dict[str, Any]:
        self.started_at = time.monotonic()
        self.worker_thread.start()
        deadline = time.monotonic() + timeout
        idle = False
        while time.monotonic() < deadline:
            with self.lock:
                previous_backpressure = self.state["backpressure_events"]
            accepted = self.poll_once()
            if not self.wait_until_idle(max(0.0, deadline - time.monotonic())):
                break
            with self.lock:
                deferred = self.state["backpressure_events"] > previous_backpressure
            if accepted == 0 and not deferred:
                idle = True
                break
        self.stop_event.set()
        self.worker_thread.join(HTTP_TIMEOUT_SECONDS + 1.0)
        if not idle:
            with self.lock:
                self.state["last_error"] = "drain_timeout"
        self._write_status()
        return dict(self.state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queue-capacity", type=int, default=DEFAULT_QUEUE_CAPACITY)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--once", action="store_true", help="read current progress and exit")
    parser.add_argument(
        "--local-only", action="store_true", help="do not contact a support service"
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    progress = run_dir / "progress.jsonl"
    status = run_dir / "telemetry-status.json"
    sender = None
    if not args.local_only:
        state = json.loads((run_dir / "integration-state.json").read_text(encoding="utf-8"))
        mathmesh = state.get("services", {}).get("mathmesh", {})
        pipeline_id = mathmesh.get("metadata_id")
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise RuntimeError("MathMesh pipeline metadata ID is missing")
        sender = MathMeshProgressSender(
            os.environ.get("MATHMESH_URL", "http://127.0.0.1:13001"),
            os.environ["MATHMESH_EMAIL"],
            os.environ["MATHMESH_PASSWORD"],
            pipeline_id,
        )
    telemetry = ProgressTelemetry(
        progress,
        status,
        sender,
        queue_capacity=args.queue_capacity,
        poll_seconds=args.poll_seconds,
    )
    if args.once:
        print(json.dumps(telemetry.run_once(), sort_keys=True))
        return
    telemetry.start()
    try:
        while True:
            telemetry.stop_event.wait(1.0)
    except KeyboardInterrupt:
        telemetry.stop()


if __name__ == "__main__":
    main()
