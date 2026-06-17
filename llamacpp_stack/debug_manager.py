from __future__ import annotations

import base64
import json
import hashlib
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import struct

import requests

if TYPE_CHECKING:
    from llamacpp_stack.cli import ManagedModel

# Debug mode configuration
MAX_LINES_PER_REQUEST = 500  # Prevent OOM from huge trace files
MAX_TRACE_FILE_SIZE_MB = 1024  # Alert if trace file exceeds this


def build_debug_model_name(base_model_id: str, session_id: str | None = None) -> str:
    date_tag = datetime.now(timezone.utc).strftime("%m-%d")
    short_id = (session_id or uuid.uuid4().hex)[:8]
    return f"{base_model_id} [DEBUG {date_tag} {short_id}]"


def build_optimized_model_name(base_model_id: str) -> str:
    date_tag = datetime.now(timezone.utc).strftime("%m-%d")
    return f"{base_model_id} [Optimised {date_tag}]"


def parse_debug_flags(flags: object) -> tuple[list[str], dict[str, object]]:
    if flags is None:
        return [], {}
    if isinstance(flags, str):
        tokens = shlex.split(flags)
        return tokens, {"raw_flags": flags}
    if not isinstance(flags, dict):
        return [], {"raw_flags": flags}

    tokens: list[str] = []
    normalized: dict[str, object] = {}
    for raw_key, raw_value in flags.items():
        key = str(raw_key).strip()
        if not key:
            continue
        normalized[key] = raw_value
        flag = key if key.startswith("-") else f"--{key.replace('_', '-')}"
        if isinstance(raw_value, (list, tuple)):
            for item in raw_value:
                if item is None or item is True:
                    tokens.append(flag)
                else:
                    tokens.extend([flag, str(item)])
            continue
        if raw_value is None or raw_value is True:
            tokens.append(flag)
        elif raw_value is False:
            tokens.extend([flag, "false"])
        else:
            tokens.extend([flag, str(raw_value)])
    return tokens, normalized


def _tail_text(text: str, lines: int = 120) -> str:
    chunks = text.splitlines()
    if not chunks:
        return ""
    return "\n".join(chunks[-lines:])


def _tail_file(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    try:
        return _tail_text(path.read_text(encoding="utf-8", errors="replace"), lines=lines)
    except Exception:
        return ""


def _count_file_lines(path: Path) -> int:
    """Count total lines in file without loading entire file into memory."""
    if not path.exists():
        return 0
    try:
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
        return count
    except Exception:
        return 0


def _get_file_size_bytes(path: Path) -> int:
    """Get file size in bytes."""
    if not path.exists():
        return 0
    try:
        return path.stat().st_size
    except Exception:
        return 0


def _query_gpu_status() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": float(parts[2]),
                    "memory_free_mib": float(parts[3]),
                    "temperature_c": float(parts[4]),
                }
            )
        except Exception:
            continue
    return gpus


def _ws_accept_key(key: str) -> str:
    magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    digest = hashlib.sha1((key.strip() + magic).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def _ws_frame_text(payload: str) -> bytes:
    data = payload.encode("utf-8")
    header = bytearray([0x81])
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < (1 << 16):
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + data


def _read_exact(connection: Any, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        piece = connection.recv(size - len(chunks))
        if not piece:
            break
        chunks.extend(piece)
    return bytes(chunks)


def _ws_read_frame(connection: Any) -> dict[str, object] | None:
    header = _read_exact(connection, 2)
    if len(header) < 2:
        return None
    first, second = header[0], header[1]
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length_raw = _read_exact(connection, 2)
        if len(length_raw) < 2:
            return None
        length = struct.unpack("!H", length_raw)[0]
    elif length == 127:
        length_raw = _read_exact(connection, 8)
        if len(length_raw) < 8:
            return None
        length = struct.unpack("!Q", length_raw)[0]
    mask_key = _read_exact(connection, 4) if masked else b""
    payload = _read_exact(connection, int(length)) if length else b""
    if masked and mask_key and payload:
        payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
    return {"opcode": opcode, "payload": payload}


@dataclass
class DebugSessionRecord:
    session_id: str
    base_model_id: str
    debug_model_id: str
    port: int
    ctx_size: int | None = None
    n_gpu_layers: int | None = None
    tensor_split: str | None = None
    description: str | None = None
    process: subprocess.Popen[str] | None = None
    trace_path: Path | None = None
    trace_handle: object | None = None
    command: list[str] = field(default_factory=list)
    flags: dict[str, object] = field(default_factory=dict)
    extra_tokens: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    catalog_path: Path | None = None
    debug_entry_written: bool = False
    previous_server_output: str = ""  # Output from server before last restart
    # Operation tracking
    current_operation: str = ""  # "loading", "restarting", "", etc.
    operation_start_time: float = 0.0
    operation_progress: dict[str, object] = field(default_factory=dict)
    operation_error: str | None = None


class DebugSessionManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session: DebugSessionRecord | None = None

    def _session_is_alive(self, record: DebugSessionRecord | None) -> bool:
        if record is None:
            return False
        if record.process is None:
            return False
        return record.process.poll() is None

    def _cleanup_catalog_entry(self, record: DebugSessionRecord) -> None:
        if record.catalog_path is None:
            return
        from llamacpp_stack import cli as cli_mod

        try:
            catalog = cli_mod.load_catalog(record.catalog_path)
            catalog = [
                model
                for model in catalog
                if not (
                    getattr(model, "debug_session_id", None) == record.session_id
                    or getattr(model, "debug_mode", False)
                    and model.model_id == record.debug_model_id
                )
            ]
            cli_mod.save_catalog(record.catalog_path, catalog)
        except Exception:
            pass

    def _finalize_stale_session_locked(self) -> None:
        record = self._session
        if record is None:
            return
        if self._session_is_alive(record):
            return
        self._cleanup_record(record)
        self._cleanup_catalog_entry(record)
        self._session = None

    def _active(self) -> DebugSessionRecord | None:
        self._finalize_stale_session_locked()
        return self._session

    def get_session(self) -> DebugSessionRecord | None:
        with self._lock:
            return self._active()

    def _session_tail(self, record: DebugSessionRecord) -> str:
        if record.trace_path is None:
            return ""
        return _tail_file(record.trace_path)

    def _session_trace_metrics(self, record: DebugSessionRecord) -> dict[str, object]:
        from llamacpp_stack import cli as cli_mod

        parsed = cli_mod._parse_probe_trace_metrics(record.trace_path) if record.trace_path else None
        if parsed is None:
            return {}
        return {
            "model_buffers_mib": parsed.model_buffers_mib,
            "kv_buffers_mib": parsed.kv_buffers_mib,
            "compute_buffers_mib": parsed.compute_buffers_mib,
            "projector_gpu": parsed.projector_gpu,
            "oom_gpu": parsed.oom_gpu,
            "oom_requested_mib": parsed.oom_requested_mib,
        }

    def _metrics_payload(self, record: DebugSessionRecord | None) -> dict[str, object]:
        if record is None:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "idle",
                "active": False,
                "session": None,
                "current_config": None,
                "runtime": None,
                "model": None,
                "gpu": {"devices": [], "summary": {}},
                "trace": None,
                "llama_output": "",
            }

        from llamacpp_stack import cli as cli_mod

        gpu_devices = _query_gpu_status()
        gpu_free = cli_mod._query_gpu_free_memory_mib()
        gpu_process_map = cli_mod.get_gpu_process_map()
        processes = cli_mod.get_llama_server_processes()
        process = next((item for item in processes if item.get("pid") == (record.process.pid if record.process else None)), None)
        trace_metrics = self._session_trace_metrics(record)
        output_tail = self._session_tail(record)
        current_config = {
            "model_id": record.base_model_id,
            "debug_model_id": record.debug_model_id,
            "ctx_size": getattr(record, "ctx_size", None),
            "n_gpu_layers": getattr(record, "n_gpu_layers", None),
            "tensor_split": getattr(record, "tensor_split", None),
            "description": getattr(record, "description", None),
            "command": record.command,
            "flags": dict(record.flags),
            "extra_tokens": list(record.extra_tokens),
        }
        summary = {
            "free_memory_mib": sum(device.get("memory_free_mib", 0.0) for device in gpu_devices),
            "used_memory_mib": sum(device.get("memory_used_mib", 0.0) for device in gpu_devices),
            "device_count": len(gpu_devices),
        }
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running" if record.process is not None and record.process.poll() is None else "stopped",
            "active": record.process is not None and record.process.poll() is None,
            "session": {
                "session_id": record.session_id,
                "model_id": record.base_model_id,
                "debug_model_id": record.debug_model_id,
                "port": record.port,
                "pid": record.process.pid if record.process is not None else None,
                "uptime_s": round(time.monotonic() - record.started_at, 3),
                "health_url": f"http://127.0.0.1:{record.port}/health",
                "command": record.command,
                "flags": dict(record.flags),
                "extra_tokens": list(record.extra_tokens),
            },
            "current_config": current_config,
            "runtime": {
                "process": process,
                "gpu_free_memory_mib": gpu_free,
                "gpu_process_map": gpu_process_map,
            },
            "model": {
                "model_id": record.base_model_id,
                "debug_model_id": record.debug_model_id,
                "ctx_size": getattr(record, "ctx_size", None),
                "n_gpu_layers": getattr(record, "n_gpu_layers", None),
                "tensor_split": getattr(record, "tensor_split", None),
                "description": getattr(record, "description", None),
            },
            "gpu": {
                "devices": gpu_devices,
                "summary": summary,
            },
            "trace": {
                "metrics": trace_metrics,
                "output_tail": output_tail,
            },
            "llama_output": output_tail,
        }

    def _cleanup_record(self, record: DebugSessionRecord) -> None:
        try:
            if record.process is not None:
                if record.process.poll() is None:
                    record.process.terminate()
                    try:
                        record.process.wait(timeout=8)
                    except Exception:
                        record.process.kill()
                        try:
                            record.process.wait(timeout=4)
                        except Exception:
                            pass
        finally:
            if record.trace_handle is not None:
                try:
                    record.trace_handle.close()
                except Exception:
                    pass

    def stop_session(self, *, remove_from_catalog: bool = True) -> dict[str, object]:
        with self._lock:
            record = self._session
            if record is None:
                return {"status": "idle"}

            self._cleanup_record(record)
            if remove_from_catalog and record.catalog_path is not None:
                self._cleanup_catalog_entry(record)

            self._session = None
            return {
                "status": "stopped",
                "session_id": record.session_id,
                "model_id": record.base_model_id,
                "debug_model_id": record.debug_model_id,
            }

    def start_session(
        self,
        *,
        args,
        catalog_path: Path,
        model_id: str,
        flags: object = None,
        timeout_s: int = 120,
    ) -> dict[str, object]:
        from llamacpp_stack import cli as cli_mod

        with self._lock:
            if self._active() is not None:
                raise RuntimeError("A debug session is already running. Stop it before starting another one.")

            catalog = cli_mod.load_catalog(catalog_path, cli_mod._args_server_config_path(args))
            resolved_name = cli_mod.resolve_catalog_model_name(model_id, catalog)
            model = next((item for item in catalog if item.model_id == resolved_name), None)
            if model is None:
                raise RuntimeError(f"model '{model_id}' not found")

            extra_tokens, normalized_flags = parse_debug_flags(flags)
            session_id = uuid.uuid4().hex
            debug_model_id = build_debug_model_name(model.model_id, session_id=session_id)

            debug_model = replace(model)
            debug_model.model_id = debug_model_id
            debug_model.description = (model.description or model.model_id) + " [DEBUG]"
            debug_model.debug_mode = True
            debug_model.debug_session_id = session_id
            debug_model.debug_flags = dict(normalized_flags)

            cleaned_catalog = [
                item
                for item in catalog
                if not (
                    getattr(item, "debug_mode", False)
                    and getattr(item, "debug_session_id", None) == session_id
                )
            ]
            cleaned_catalog.append(debug_model)
            cli_mod.save_catalog(catalog_path, cleaned_catalog)

            try:
                cli_mod.temporarily_unload_published_models(args, progress_callback=None, timeout=45)
            except Exception:
                pass

            port = cli_mod._find_free_port()
            trace_path, trace_handle = cli_mod.create_llamacpp_trace_file(debug_model_id, int(debug_model.ctx_size))
            command = cli_mod.build_llama_server_command(
                debug_model,
                args.llama_server,
                port=str(port),
                host="127.0.0.1",
                server_defaults=cli_mod.resolve_llama_server_defaults(args),
                extra_flags=extra_tokens,
            )

            process = subprocess.Popen(
                command,
                stdout=trace_handle,
                stderr=subprocess.STDOUT,
                env=cli_mod._probe_runtime_env(),
                text=False,
            )

            record = DebugSessionRecord(
                session_id=session_id,
                base_model_id=model.model_id,
                debug_model_id=debug_model_id,
                port=port,
                ctx_size=int(debug_model.ctx_size),
                n_gpu_layers=int(debug_model.n_gpu_layers),
                tensor_split=str(debug_model.tensor_split),
                description=str(debug_model.description),
                process=process,
                trace_path=trace_path,
                trace_handle=trace_handle,
                command=command,
                flags=dict(normalized_flags),
                extra_tokens=list(extra_tokens),
                catalog_path=catalog_path,
                debug_entry_written=True,
            )

            health_url = f"http://127.0.0.1:{port}/health"
            deadline = time.time() + max(10, int(timeout_s))
            last_error = "startup-timeout"
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                try:
                    response = requests.get(health_url, timeout=2)
                    if response.status_code == 200:
                        self._session = record
                        return {
                            "status": "loaded",
                            "session_id": session_id,
                            "model_id": model.model_id,
                            "debug_model_id": debug_model_id,
                            "port": port,
                            "health_url": health_url,
                            "command": command,
                            "flags": dict(normalized_flags),
                            "extra_tokens": list(extra_tokens),
                            "llama_output": self._session_tail(record),
                        }
                    last_error = f"health-http-{response.status_code}"
                except Exception as exc:
                    last_error = str(exc)
                time.sleep(1.0)

            tail = self._session_tail(record)
            self._cleanup_record(record)
            try:
                catalog = cli_mod.load_catalog(catalog_path)
                catalog = [item for item in catalog if getattr(item, "debug_session_id", None) != session_id]
                cli_mod.save_catalog(catalog_path, catalog)
            except Exception:
                pass
            raise RuntimeError(f"Debug session failed to start: {last_error}. {tail[-1000:]}")

    def get_status(self) -> dict[str, object]:
        return self.get_metrics_snapshot()

    def get_metrics_snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._metrics_payload(self._active())

    def stream_metrics(
        self,
        connection: Any,
        *,
        interval_s: float = 2.0,
        send_json_line: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        interval = max(0.5, float(interval_s))
        while True:
            with self._lock:
                payload = self._metrics_payload(self._active())
            if send_json_line is not None:
                send_json_line(payload)
            else:
                connection.sendall(_ws_frame_text(json.dumps(payload, ensure_ascii=False)))
            if not payload.get("active"):
                return
            time.sleep(interval)

    def stream_metrics_websocket(self, connection: Any, *, interval_s: float = 2.0) -> None:
        self.stream_metrics(connection, interval_s=interval_s)

    def get_logs(self, session_id: str | None = None) -> dict[str, object]:
        with self._lock:
            record = self._active()
            if record is None:
                return {"status": "idle", "logs": ""}
            if session_id and session_id != record.session_id:
                return {"status": "not-found", "logs": ""}
            from llamacpp_stack import cli as cli_mod

            trace_text = _tail_file(record.trace_path) if record.trace_path else ""
            parsed = cli_mod._parse_probe_trace_metrics(record.trace_path) if record.trace_path else None
            parsed_payload = {}
            if parsed is not None:
                parsed_payload = {
                    "model_buffers_mib": parsed.model_buffers_mib,
                    "kv_buffers_mib": parsed.kv_buffers_mib,
                    "compute_buffers_mib": parsed.compute_buffers_mib,
                    "projector_gpu": parsed.projector_gpu,
                    "oom_gpu": parsed.oom_gpu,
                    "oom_requested_mib": parsed.oom_requested_mib,
                }
            return {
                "status": "running",
                "session_id": record.session_id,
                "model_id": record.base_model_id,
                "debug_model_id": record.debug_model_id,
                "logs": trace_text,
                "parsed_metrics": parsed_payload,
            }

    def get_debug_health(self) -> dict[str, object]:
        """Get lightweight health status of debug session."""
        with self._lock:
            record = self._active()
            if record is None:
                return {
                    "status": "idle",
                    "session_id": None,
                }
            
            uptime = time.monotonic() - record.started_at
            trace_size_mb = (_get_file_size_bytes(record.trace_path) / (1024 * 1024)) if record.trace_path else 0
            
            warnings = []
            if trace_size_mb > MAX_TRACE_FILE_SIZE_MB:
                warnings.append(f"Trace file > {MAX_TRACE_FILE_SIZE_MB}MB")
            if record.process and record.process.poll() is not None:
                warnings.append("Server process has exited")
            
            return {
                "status": "active",
                "session_id": record.session_id,
                "debug_model_id": record.debug_model_id,
                "uptime_seconds": round(uptime, 2),
                "process_pid": record.process.pid if record.process else None,
                "trace_file_size_mb": round(trace_size_mb, 2),
                "warnings": warnings,
            }

    def get_operation_status(self) -> dict[str, object]:
        """Get status of currently running operation."""
        with self._lock:
            record = self._active()
            if record is None:
                return {"status": "idle", "operation": None}
            
            if not record.current_operation:
                return {"status": "active", "operation": None}
            
            elapsed = time.monotonic() - record.operation_start_time if record.operation_start_time else 0
            return {
                "status": "active",
                "operation": record.current_operation,
                "progress": record.operation_progress,
                "elapsed_seconds": round(elapsed, 2),
                "error": record.operation_error,
            }

    def set_operation(self, operation_name: str, progress: dict[str, object] | None = None) -> None:
        """Mark that an operation has started."""
        with self._lock:
            record = self._active()
            if record is not None:
                record.current_operation = operation_name
                record.operation_start_time = time.monotonic()
                record.operation_progress = progress or {}
                record.operation_error = None

    def update_operation_progress(self, progress: dict[str, object]) -> None:
        """Update progress of current operation."""
        with self._lock:
            record = self._active()
            if record is not None:
                record.operation_progress.update(progress)

    def end_operation(self, success: bool = True, error_msg: str | None = None) -> None:
        """Mark that operation has completed."""
        with self._lock:
            record = self._active()
            if record is not None:
                record.current_operation = ""
                record.operation_start_time = 0.0
                if not success and error_msg:
                    record.operation_error = error_msg
                else:
                    record.operation_error = None

    def websocket_accept_key(self, key: str) -> str:
        return _ws_accept_key(key)

    def websocket_frame_text(self, payload: str) -> bytes:
        return _ws_frame_text(payload)

    def websocket_read_frame(self, connection: Any) -> dict[str, object] | None:
        return _ws_read_frame(connection)

    def save_as_optimal(
        self,
        *,
        args,
        catalog_path: Path,
        model_id: str | None,
        flags: object = None,
        metrics: dict[str, object] | None = None,
    ) -> dict[str, object]:
        from llamacpp_stack import cli as cli_mod

        with self._lock:
            catalog = cli_mod.load_catalog(catalog_path, cli_mod._args_server_config_path(args))
            record = self._active()
            if model_id:
                resolved_name = cli_mod.resolve_catalog_model_name(model_id, catalog)
            elif record is not None:
                resolved_name = record.base_model_id
            else:
                raise RuntimeError("model_id is required when no debug session is active")

            base_model = next((item for item in catalog if item.model_id == resolved_name), None)
            if base_model is None:
                raise RuntimeError(f"model '{resolved_name}' not found")

            extra_tokens, normalized_flags = parse_debug_flags(flags)
            optimized = replace(base_model)
            optimized_model_id = build_optimized_model_name(base_model.model_id)
            existing_ids = {item.model_id for item in catalog}
            suffix = 2
            candidate = optimized_model_id
            while candidate in existing_ids:
                candidate = f"{optimized_model_id} #{suffix}"
                suffix += 1
            optimized.model_id = candidate
            optimized.debug_mode = False
            optimized.optimization_session_id = record.session_id if record is not None else uuid.uuid4().hex
            optimized.optimized_from = base_model.model_id
            optimized.optimized_at = datetime.now(timezone.utc).isoformat()
            optimized.debug_flags = dict(normalized_flags)
            if metrics:
                optimized.debug_flags = {**optimized.debug_flags, "metrics": metrics}
            if extra_tokens:
                optimized.debug_flags = {**optimized.debug_flags, "extra_tokens": extra_tokens}
            catalog.append(optimized)
            cli_mod.save_catalog(catalog_path, catalog)
            return {
                "status": "saved",
                "saved_model_id": optimized.model_id,
                "catalog_entry": asdict(optimized),
            }

    def get_server_output(self, lines: int = 50) -> dict[str, object]:
        """Get the last N lines of server output (stdout/stderr combined)."""
        with self._lock:
            record = self._active()
            if record is None:
                return {"status": "idle", "output": "", "total_lines": 0, "total_bytes": 0}
            if record.trace_path is None:
                return {"status": "running", "output": "", "total_lines": 0, "total_bytes": 0}
            
            # Enforce limits to prevent OOM
            lines = min(max(1, int(lines)), MAX_LINES_PER_REQUEST)
            
            # Get metadata about trace file
            total_lines = _count_file_lines(record.trace_path)
            total_bytes = _get_file_size_bytes(record.trace_path)
            
            output = _tail_file(record.trace_path, lines=lines)
            
            warns = []
            if total_bytes > (MAX_TRACE_FILE_SIZE_MB * 1024 * 1024):
                warns.append(f"Trace file > {MAX_TRACE_FILE_SIZE_MB}MB")
            
            return {
                "status": "running",
                "session_id": record.session_id,
                "debug_model_id": record.debug_model_id,
                "lines_requested": lines,
                "total_lines": total_lines,
                "total_bytes": total_bytes,
                "output": output,
                "warnings": warns,
            }

    def get_server_output_previous(self, lines: int = 50) -> dict[str, object]:
        """Get the last N lines of server output from before the last restart."""
        with self._lock:
            record = self._active()
            if record is None:
                return {"status": "idle", "previous_output": "", "total_lines": 0}
            
            # Enforce limits
            lines = min(max(1, int(lines)), MAX_LINES_PER_REQUEST)
            
            if not record.previous_server_output:
                return {
                    "status": "running",
                    "session_id": record.session_id,
                    "debug_model_id": record.debug_model_id,
                    "note": "No previous output (server has not been restarted)",
                    "previous_output": "",
                    "total_lines": 0,
                }
            
            # Get last N lines from saved previous output
            prev_lines = record.previous_server_output.splitlines()
            tail_lines = prev_lines[-lines:] if prev_lines else []
            total_prev_lines = len(prev_lines)
            
            return {
                "status": "running",
                "session_id": record.session_id,
                "debug_model_id": record.debug_model_id,
                "lines_requested": lines,
                "total_lines": total_prev_lines,
                "from_restart": "Yes, this is output before last restart",
                "previous_output": "\n".join(tail_lines),
            }

    def restart_server(self, *, timeout_s: int = 30) -> dict[str, object]:
        """Restart the llama-cpp server within the current debug session."""
        with self._lock:
            record = self._active()
            if record is None:
                return {"status": "error", "message": "No active debug session"}
            if record.process is None:
                return {"status": "error", "message": "No server process found"}
            if not record.command:
                return {"status": "error", "message": "No command stored to restart"}

            old_pid = record.process.pid if record.process else None
            
            # Save current output before terminating (for /api/debug/server-output-previous)
            try:
                if record.trace_path is not None:
                    record.previous_server_output = _tail_file(record.trace_path, lines=100)
            except Exception:
                pass
            
            # Terminate old process gracefully
            try:
                if record.process.poll() is None:  # Process is still running
                    record.process.terminate()
                    try:
                        record.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        record.process.kill()
                        record.process.wait(timeout=2)
            except Exception as exc:
                return {"status": "error", "message": f"Failed to terminate old process: {exc}"}

            # Start new process with same command
            try:
                if record.trace_handle is not None:
                    try:
                        record.trace_handle.close()
                    except Exception:
                        pass

                # Reopen trace file for new process
                from llamacpp_stack import cli as cli_mod
                trace_path, trace_handle = cli_mod.create_llamacpp_trace_file(
                    record.debug_model_id, 
                    int(record.ctx_size) if record.ctx_size else 2048
                )
                
                new_process = subprocess.Popen(
                    record.command,
                    stdout=trace_handle,
                    stderr=subprocess.STDOUT,
                    env=cli_mod._probe_runtime_env(),
                    text=False,
                )
                
                # Update record with new process and trace file
                record.process = new_process
                record.trace_path = trace_path
                record.trace_handle = trace_handle

                # Wait for server to become healthy
                health_url = f"http://127.0.0.1:{record.port}/health"
                deadline = time.time() + max(5, int(timeout_s))
                last_error = "startup-timeout"
                
                while time.time() < deadline:
                    if new_process.poll() is not None:
                        break
                    try:
                        response = requests.get(health_url, timeout=2)
                        if response.status_code == 200:
                            return {
                                "status": "restarted",
                                "old_pid": old_pid,
                                "new_pid": new_process.pid,
                                "session_id": record.session_id,
                                "debug_model_id": record.debug_model_id,
                                "health_url": health_url,
                                "message": "Server restarted successfully",
                            }
                        last_error = f"health-http-{response.status_code}"
                    except Exception as exc:
                        last_error = str(exc)
                    time.sleep(0.5)

                # If we get here, server failed to start
                new_output = _tail_file(record.trace_path, lines=20) if record.trace_path else ""
                return {
                    "status": "error",
                    "message": f"Restarted server failed to become healthy: {last_error}",
                    "server_output": new_output,
                }

            except Exception as exc:
                return {
                    "status": "error",
                    "message": f"Failed to restart server: {exc}",
                }


DEBUG_SESSION_MANAGER = DebugSessionManager()
