"""Models/types extracted from llamacpp_stack/cli.py:1384-1543.

Spinner, LoadingBar, ManagedModel, ReplicaConfig, ReplicaRecord, ProbeTraceMetrics.
No top-level import of command_router/managed_commands to avoid cycles.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from dataclasses import dataclass, field

from .constants import (
    DEFAULT_CTX_SIZE,
    DEFAULT_IDLE_TTL,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_TENSOR_SPLIT,
)


class Spinner:
    """Universal ASCII Spinner for maximum terminal compatibility."""

    def __init__(self, label="assistant: "):
        self.label = label
        self._frames = itertools.cycle(["|", "/", "-", "\\"])
        self._running = False
        self._thread = None
        self.cyan = "\033[36;1m"
        self.reset = "\033[0m"

    def _spin(self):
        sys.stdout.write("\033[?25l")
        while self._running:
            sys.stdout.write(f"\r{self.label}{self.cyan}{next(self._frames)}{self.reset}")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{self.label}\033[K")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        self._thread = None


class LoadingBar:
    """Indeterminate loading bar for model warmup."""

    def __init__(self, label="Loading model: ", width=24):
        self.label = label
        self.width = width
        self._running = False
        self._thread = None
        self.cyan = "\033[36;1m"
        self.reset = "\033[0m"

    def _render_frame(self, pos):
        cells = ["-"] * self.width
        for idx in range(4):
            cell = pos + idx
            if 0 <= cell < self.width:
                cells[cell] = "="
        return "[" + "".join(cells) + "]"

    def _spin(self):
        sys.stdout.write("\033[?25l")
        travel = max(1, self.width - 3)
        pos = 0
        direction = 1
        while self._running:
            bar = self._render_frame(pos)
            sys.stdout.write(f"\r{self.label}{self.cyan}{bar}{self.reset}")
            sys.stdout.flush()
            time.sleep(0.08)
            pos += direction
            if pos >= travel:
                pos = travel
                direction = -1
            elif pos <= 0:
                pos = 0
                direction = 1

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{self.label}\033[K")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        self._thread = None


@dataclass
class ManagedModel:
    model_id: str
    repo_id: str
    quant: str | None
    filename: str
    local_path: str
    backend: str = "llama.cpp"
    mmproj_filename: str | None = None
    mmproj_path: str | None = None
    load_capabilities: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    ctx_size: int = DEFAULT_CTX_SIZE
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    tensor_split: str = DEFAULT_TENSOR_SPLIT
    host: str = "127.0.0.1"
    jinja: bool = True
    ttl: int = DEFAULT_IDLE_TTL
    description: str = ""
    downloaded_at: str = ""
    speculative: bool = False
    spec_variant_of: str | None = None
    spec_meta: dict[str, object] = field(default_factory=dict)
    auto_ctx_failed: bool = False
    auto_ctx_error: str = ""
    ctx_probe_read_s: float | None = None
    ctx_probe_tokens_s: float | None = None
    ctx_probe_totals_s: float | None = None
    ctx_probe_latency_ms: float | None = None
    ctx_probe_speed_tps: float | None = None
    ctx_probe_kv_gb: float | None = None
    ctx_probe_prompt_tokens: int | None = None
    server_overrides: dict[str, object] = field(default_factory=dict)


@dataclass
class ReplicaConfig:
    enabled: bool = False
    max: int = 1
    gpus_per_replica: int = 1
    placement: str = "exclusive_gpus"
    safety_vram_mib: int = 2048
    max_models_per_gpu: int = 2
    max_pack_fraction: float = 0.35
    sticky_ttl_s: int = 3600


@dataclass
class ReplicaRecord:
    base_model_id: str
    replica_model_id: str
    gpu_set: list[int] = field(default_factory=list)
    status: str = "cold"
    estimated_mib: float | None = None
    actual_mib: float | None = None
    gpu_actual_mib: dict[int, float] = field(default_factory=dict)
    pid: int | None = None
    port: int | None = None
    in_flight: int = 0
    last_used: float = 0.0
    blacklist_until: float = 0.0


@dataclass
class ProbeTraceMetrics:
    model_buffers_mib: dict[int, float] = field(default_factory=dict)
    kv_buffers_mib: dict[int, float] = field(default_factory=dict)
    compute_buffers_mib: dict[int, float] = field(default_factory=dict)
    projector_gpu: int | None = None
    oom_gpu: int | None = None
    oom_requested_mib: float | None = None


__all__ = [
    "Spinner",
    "LoadingBar",
    "ManagedModel",
    "ReplicaConfig",
    "ReplicaRecord",
    "ProbeTraceMetrics",
]
