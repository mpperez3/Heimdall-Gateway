from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import textwrap
import urllib.request
import yaml
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLAMA_CPP_REPO = "ggml-org/llama.cpp"
DEFAULT_LLAMASWAP_REPO = "mostlygeek/llama-swap"
DEFAULT_IDLE_TTL = 300
DEFAULT_SERVER_KEEP = 512
DEFAULT_SERVICE_USER = "llamaswap"
MANAGER_SERVICE_NAME = "llamacpp-superserver-manager.service"
SWAP_SERVICE_NAME = "llamacpp-superserver-swap.service"
LEGACY_MANAGER_SERVICE_NAME = "llamacpp-manager.service"
LEGACY_SWAP_SERVICE_NAME = "llamaswap.service"
CLI_COMMAND = "llamacpp-superserver"
LEGACY_CLI_COMMAND = "llamacpp-server"
MANAGER_WRAPPER_NAME = "llamacpp-superserver-manager-start"
SWAP_WRAPPER_NAME = "llamacpp-superserver-swap-start"
OLLAMA_DEFAULT_PORT = 11434
SERVER_CONFIG_BASENAME = "conf.json"
LLAMA_SERVER_DEFAULTS_BASENAME = "llama_server_defaults.yaml"
LEGACY_SERVER_CONFIG_BASENAME = "llamacpp-server.json"
ENV_BASENAME = "llamacpp-superserver.env"
LEGACY_ENV_BASENAME = "llamacpp-stack.env"
LLAMA_CPP_MODES = ("native", "prebuilt", "source")
ELEVATED_INSTALL_ENV = "LLAMACPP_INSTALL_ELEVATED"

CONFIG_YAML_HEADER = textwrap.dedent(
    """\
    # llamacpp-superserver config.yaml
    # Purpose: llama-swap runtime routing + per-model launch command map.
    # This file is generated/updated by installer and `llamacpp-superserver update`.
    # Example:
    #   models:
    #     my-model-id:
    #       cmd: /opt/llama-server --model /models/my.gguf --ctx-size 65536
    #       checkEndpoint: /health
    #       ttl: 300

    """
)

ENV_FILE_HEADER = textwrap.dedent(
    """\
    # llamacpp-superserver.env
    # Purpose: global process environment consumed by manager/swap wrappers.
    # Example:
    #   LLAMACPP_CATALOG=/var/lib/llamacpp-superserver/catalog.json
    #   LLAMACPP_SERVER_CONFIG=/etc/llamacpp-superserver/conf.json
    #   LLAMACPP_IDLE_TTL=300

    """
)


def _ensure_server_config_metadata(payload: dict[str, object]) -> dict[str, object]:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("purpose", "Global superserver settings consumed by CLI/services.")
    meta.setdefault(
        "note",
        "Per-model context source of truth is catalog/config.yaml; this file stores global defaults and service-level settings.",
    )
    meta.setdefault(
        "example",
        {
            "idle_ttl": 300,
            "api_port": 11436,
            "llama_server_defaults": {
                "ctx_size": 65536,
                "n_gpu_layers": 999,
                "keep": 512,
                "mirostat": 2,
                "mirostat_ent": 4.5,
                "mirostat_lr": 0.1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        },
    )
    meta.setdefault(
        "service_restart_help",
        {
            "system_mode": f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
            "user_mode": f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        },
    )
    payload["_meta"] = meta
    return payload


def _bundle_llama_server_defaults_path() -> Path:
    return Path(__file__).resolve().parent / "bundle" / LLAMA_SERVER_DEFAULTS_BASENAME


def _ensure_llama_server_defaults_file(config_dir: Path) -> Path:
    target = config_dir / LLAMA_SERVER_DEFAULTS_BASENAME
    # Do not copy the bundled defaults into the user's config dir by default.
    # Fall back to reading the bundled preset in `bundle/` without creating a
    # copy. This avoids cluttering user config directories with installer
    # artifacts while still providing sensible defaults.
    source = _bundle_llama_server_defaults_path()
    if source.exists():
        return source
    return target


def _load_llama_server_defaults_preset(config_dir: Path) -> dict[str, object]:
    defaults_path = _ensure_llama_server_defaults_file(config_dir)
    try:
        payload = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    base_defaults = payload.get("default")
    if not isinstance(base_defaults, dict):
        base_defaults = {}
    presets = payload.get("presets")
    selected = {}
    if isinstance(presets, dict):
        gpu_count = detect_cuda_device_count()
        for key in (str(gpu_count), gpu_count):
            preset = presets.get(key)
            if isinstance(preset, dict):
                selected = dict(preset)
                break

    merged = dict(base_defaults)
    merged.update(selected)
    return merged


def _merge_missing_llama_server_defaults(target: dict[str, object], config_dir: Path) -> bool:
    if not isinstance(target, dict):
        return False
    preset_defaults = _load_llama_server_defaults_preset(config_dir)
    if not preset_defaults:
        return False
    changed = False
    for key, value in preset_defaults.items():
        if key not in target:
            target[key] = value
            changed = True
    return changed


@dataclass
class InstallLayout:
    mode: str
    state_dir: Path
    bin_dir: Path
    install_root: Path
    models_dir: Path
    config_dir: Path
    run_dir: Path
    service_user: str
    service_group: str
    public_host: str
    public_port: int
    manager_socket: Path
    python_root: Path
    runtime_venv: Path
    cuda_root: Path
    nccl_root: Path = Path()


def prompt_bool(message: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{message} {suffix} ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_path(message: str, default: Path) -> Path:
    if not sys.stdin.isatty():
        return default
    raw = input(f"{message} [{default}] ").strip()
    return Path(raw).expanduser() if raw else default


def prompt_choice(message: str, options: list[tuple[str, str]], default: str) -> str:
    if not sys.stdin.isatty():
        return default
    labels = {key: label for key, label in options}
    lines = [message]
    numeric_to_key: dict[str, str] = {}
    for index, (key, label) in enumerate(options, start=1):
        numeric_to_key[str(index)] = key
        suffix = " [default]" if key == default else ""
        title = key.capitalize()
        lines.append(f"  {index}. {title}{suffix}")
        lines.append(f"     {label}")
    lines.append(f"Choice [{default}]")
    prompt = "\n".join(lines) + ": "
    while True:
        raw = input(prompt).strip().lower()
        choice = numeric_to_key.get(raw, raw) or default
        if choice in labels:
            return choice
        print(f"Choose one of: {', '.join(numeric_to_key)} or {', '.join(labels)}")


def resolve_install_mode(requested_mode: str | None) -> str:
    if requested_mode:
        return requested_mode
    existing = detect_existing_mode()
    default_mode = existing or ("system" if os.geteuid() == 0 else "user")
    if not sys.stdin.isatty():
        return default_mode
    if existing:
        print(f"Existing installation detected in {existing} mode.")
    if prompt_bool("Install for all users?", default=(default_mode == "system")):
        return "system"
    return "user"


def resolve_llama_cpp_mode(requested_mode: str | None) -> str:
    if requested_mode:
        return requested_mode
    return prompt_choice(
        "How should llama.cpp be installed?",
        [
            ("source", "build locally from source (best default, best GPU tuning)"),
            ("prebuilt", "download a precompiled binary (fastest install)"),
            ("native", "use a system-wide llama.cpp already installed on the machine"),
        ],
        default="source",
    )


def resolve_public_host(requested_host: str | None) -> str:
    if requested_host and requested_host != "127.0.0.1":
        return requested_host
    if not sys.stdin.isatty():
        return requested_host or "127.0.0.1"
    if prompt_bool("Expose superserver API and llama-swap on all network interfaces (0.0.0.0)?", default=False):
        return "0.0.0.0"
    return "127.0.0.1"


def find_next_free_port(host: str = "127.0.0.1", start: int = 11437, limit: int = 11550) -> int:
    for port in range(start, limit + 1):
        with socket.socket() as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"No free port found between {start} and {limit}.")


def _port_is_free(host: str, port: int) -> bool:
    try:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) != 0
    except OSError:
        return True


def detect_ollama_port(default: int = OLLAMA_DEFAULT_PORT) -> int:
    raw_env = os.environ.get("OLLAMA_HOST", "").strip()
    if raw_env:
        match = re.search(r":(\d+)(?:/)?$", raw_env)
        if match:
            return int(match.group(1))
    try:
        result = subprocess.run(
            ["systemctl", "cat", "ollama"],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(r"OLLAMA_HOST=[^\"\n:]+:(\d+)", result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return default


def choose_default_swap_port(host: str, mode: str, explicit_public_port: int | None, args: argparse.Namespace | None = None) -> int:
    if explicit_public_port:
        return explicit_public_port

    # If we already have a chosen port in args from a previous pass, use it.
    if args and getattr(args, "public_port", None):
        return args.public_port

    ollama_port = detect_ollama_port()
    ideal_swap_port = ollama_port + 2
    ideal_api_port = ollama_port + 1

    existing = existing_public_port(mode)

    # If it matches our ideal logic, use it.
    if existing == ideal_swap_port:
        return existing

    # If we have an existing port that differs from the ideal
    if existing:
        if not sys.stdin.isatty():
            return existing

        print(f"\nExisting installation uses port {existing} for llama-swap UI.")
        print(f"Recommended ports based on Ollama ({ollama_port}) are:")
        print(f"  UI:  {ideal_swap_port}")
        print(f"  API: {ideal_api_port}")

        # Don't check if ports are free here because they might be occupied by 
        # the currently running services we are about to upgrade/replace.
        if prompt_bool(f"Migrate to recommended ports ({ideal_swap_port}/{ideal_api_port})?", default=True):
            if args:
                args.public_port = ideal_swap_port
            return ideal_swap_port
        if args:
            args.public_port = existing
        return existing

    # No existing config, follow ideal logic
    if _port_is_free(host, ideal_api_port) and _port_is_free(host, ideal_swap_port):
        return ideal_swap_port

    return find_next_free_port(host, start=ideal_swap_port)


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "OpenCodeAutoModelDiscover/llamacpp_stack",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "OpenCodeAutoModelDiscover/llamacpp_stack"})
    with urllib.request.urlopen(req, timeout=60) as response, dest.open("wb") as output:
        shutil.copyfileobj(response, output)


def latest_release(repo: str) -> dict:
    return _fetch_json(f"https://api.github.com/repos/{repo}/releases/latest")


def dry_run_release_placeholder(repo: str) -> dict:
    if repo == DEFAULT_LLAMA_CPP_REPO:
        return {"tag_name": "dry-run", "assets": [{"name": "llama-dry-run-bin-ubuntu-x64.tar.gz", "browser_download_url": "https://example.invalid/llama.cpp.tar.gz"}]}
    return {"tag_name": "dry-run", "assets": [{"name": "llama-swap_dry-run_linux_amd64.tar.gz", "browser_download_url": "https://example.invalid/llama-swap.tar.gz"}]}


def detect_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return any(line.strip() for line in result.stdout.splitlines())


def detect_cuda_device_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def locate_nvcc() -> str | None:
    for candidate in (
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def locate_nvcc_for_python(python_exec: str) -> str | None:
    probe_script = """
import glob
import os
import site
import sysconfig

candidates = []
scripts = sysconfig.get_path("scripts")
if scripts:
    candidates.append(os.path.join(scripts, "nvcc"))

roots = site.getsitepackages() + [site.getusersitepackages()]
for root in roots:
    candidates.extend(
        [
            os.path.join(root, "nvidia", "cuda_nvcc", "bin", "nvcc"),
            os.path.join(root, "nvidia", "cuda_nvcc", "bin", "nvcc.real"),
        ]
    )
    candidates.extend(glob.glob(os.path.join(root, "nvidia", "*", "bin", "nvcc")))
    candidates.extend(glob.glob(os.path.join(root, "nvidia", "*", "bin", "nvcc.real")))

print(next((p for p in candidates if p and os.path.exists(p)), ""))
""".strip()
    try:
        probe = subprocess.run(
            [
                python_exec,
                "-c",
                probe_script,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    path = probe.stdout.strip()
    return path or None


def locate_cuda_root_for_python(python_exec: str) -> Path | None:
    nvcc_path = locate_nvcc_for_python(python_exec)
    if not nvcc_path:
        return None
    nvcc = Path(nvcc_path).resolve()
    candidate_root = nvcc.parent.parent
    if (candidate_root / "include").exists() and any((candidate_root / "lib").glob("libcudart.so*")):
        return candidate_root
    if (candidate_root / "include").exists() and any((candidate_root / "lib64").glob("libcudart.so*")):
        return candidate_root
    return None


def locate_nccl_root_for_python(python_exec: str) -> Path | None:
    probe_script = """
import glob
import os
import site

roots = site.getsitepackages() + [site.getusersitepackages()]
for root in roots:
    for candidate in glob.glob(os.path.join(root, "nvidia", "nccl*")):
        include_dir = os.path.join(candidate, "include")
        for lib_name in ("lib", "lib64"):
            lib_dir = os.path.join(candidate, lib_name)
            if os.path.isdir(include_dir) and os.path.isdir(lib_dir):
                if glob.glob(os.path.join(lib_dir, "libnccl.so*")):
                    print(candidate)
                    raise SystemExit(0)
print("")
""".strip()
    try:
        probe = subprocess.run([python_exec, "-c", probe_script], check=True, capture_output=True, text=True)
    except Exception:
        return None
    path = probe.stdout.strip()
    return Path(path).resolve() if path else None


def detect_cuda_toolkit() -> bool:
    return locate_nvcc() is not None


def _sudo_prefix() -> list[str]:
    return [] if os.geteuid() == 0 else ["sudo"]


def _args_to_cli(argv: argparse.Namespace, chosen_mode: str, chosen_llama_cpp_mode: str, chosen_models_dir: Path) -> list[str]:
    cmd = [
        "--mode",
        chosen_mode,
        "--llama-cpp-mode",
        chosen_llama_cpp_mode,
        "--models-dir",
        str(chosen_models_dir),
        "--idle-ttl",
        str(argv.idle_ttl),
    ]
    if argv.public_host:
        cmd.extend(["--public-host", str(argv.public_host)])
    if argv.public_port is not None:
        cmd.extend(["--public-port", str(argv.public_port)])
    if argv.enable_tls:
        cmd.append("--enable-tls")
    cmd.append("--prefer-source-cuda" if argv.prefer_source_cuda else "--no-prefer-source-cuda")
    cmd.append("--prefer-binary" if argv.prefer_binary else "--no-prefer-binary")
    cmd.append("--install-services" if argv.install_services else "--no-install-services")
    update_binaries = getattr(argv, "update_binaries", None)
    if update_binaries is not None:
        cmd.append("--update-binaries" if update_binaries else "--no-update-binaries")
    migrate_model_ids = getattr(argv, "migrate_model_ids", None)
    if migrate_model_ids is not None:
        cmd.append("--migrate-model-ids" if migrate_model_ids else "--no-migrate-model-ids")
    if argv.dry_run:
        cmd.append("--dry-run")
    return cmd


def maybe_reexec_system_install(argv: argparse.Namespace, chosen_mode: str, chosen_llama_cpp_mode: str, chosen_models_dir: Path) -> int | None:
    if chosen_mode != "system" or os.geteuid() == 0 or os.environ.get(ELEVATED_INSTALL_ENV) == "1":
        return None
    cmd = [
        "sudo",
        "-E",
        sys.executable,
        str(Path(__file__).resolve()),
        *_args_to_cli(argv, chosen_mode, chosen_llama_cpp_mode, chosen_models_dir),
    ]
    env = os.environ.copy()
    env[ELEVATED_INSTALL_ENV] = "1"
    print("System-wide install selected. Re-running installer with sudo.")
    subprocess.run(cmd, check=True, env=env)
    return 0


def detect_cuda_toolkit_package() -> str | None:
    if shutil.which("apt-cache") is None:
        return None
    candidates = [
        "cuda-toolkit-12-6",
        "cuda-toolkit-12-5",
        "cuda-toolkit-12-4",
        "cuda-toolkit-12-3",
        "cuda-toolkit-12-2",
        "cuda-toolkit-12-1",
        "cuda-toolkit-12-0",
        "nvidia-cuda-toolkit",
    ]
    for package in candidates:
        try:
            result = subprocess.run(
                ["apt-cache", "policy", package],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        if result.returncode != 0:
            continue
        if re.search(r"Candidate:\s+(?!\(none\))", result.stdout):
            return package
    return None


def resolve_uv_executable() -> str | None:
    bootstrap_uv = os.environ.get("LLAMACPP_BOOTSTRAP_UV")
    if bootstrap_uv and Path(bootstrap_uv).exists():
        return bootstrap_uv
    if uv_bin := shutil.which("uv"):
        return uv_bin
    sibling = Path(sys.executable).resolve().parent / "uv"
    if sibling.exists():
        return str(sibling)
    return None


def _export_nvcc_path(nvcc_path: str | None) -> bool:
    if not nvcc_path:
        return False
    nvcc_resolved = str(Path(nvcc_path).resolve())
    nvcc_dir = str(Path(nvcc_resolved).parent)
    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    if nvcc_dir not in parts:
        os.environ["PATH"] = nvcc_dir + (os.pathsep + current_path if current_path else "")
    os.environ["CUDACXX"] = nvcc_resolved
    return True


def _export_cuda_root(cuda_root: Path | None) -> bool:
    if not cuda_root:
        return False
    cuda_root = cuda_root.resolve()
    os.environ["CUDAToolkit_ROOT"] = str(cuda_root)
    os.environ["CUDA_PATH"] = str(cuda_root)
    bin_dir = cuda_root / "bin"
    if bin_dir.exists():
        current_path = os.environ.get("PATH", "")
        parts = current_path.split(os.pathsep) if current_path else []
        if str(bin_dir) not in parts:
            os.environ["PATH"] = str(bin_dir) + (os.pathsep + current_path if current_path else "")
    lib_dirs = [cuda_root / "lib64", cuda_root / "lib"]
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in existing_ld.split(os.pathsep) if part] if existing_ld else []
    updated = False
    for lib_dir in lib_dirs:
        if lib_dir.exists() and str(lib_dir) not in ld_parts:
            ld_parts.insert(0, str(lib_dir))
            updated = True
    if updated:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    return True


def _export_nccl_root(nccl_root: Path | None) -> bool:
    if not nccl_root:
        return False
    nccl_root = nccl_root.resolve()
    os.environ["NCCL_ROOT"] = str(nccl_root)
    ld_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    updated = False
    for lib_dir in (nccl_root / "lib64", nccl_root / "lib"):
        if lib_dir.exists() and str(lib_dir) not in ld_parts:
            ld_parts.insert(0, str(lib_dir))
            updated = True
    if updated:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    return True


def normalize_python_cuda_layout(cuda_root: Path | None) -> bool:
    if not cuda_root or not cuda_root.exists():
        return False
    changed = False
    lib_dir = cuda_root / "lib"
    lib64_dir = cuda_root / "lib64"
    if lib_dir.exists() and not lib64_dir.exists():
        lib64_dir.symlink_to(lib_dir)
        changed = True
    if lib_dir.exists():
        for soname in lib_dir.glob("lib*.so.*"):
            base_name = soname.name.split(".so.", 1)[0] + ".so"
            link = lib_dir / base_name
            if not link.exists():
                link.symlink_to(soname.name)
                changed = True
    return changed


def maybe_install_cuda_toolkit_via_uv(python_exec: str, dry_run: bool) -> bool:
    uv_bin = resolve_uv_executable()
    if dry_run:
        print(f"[dry-run] would offer CUDA toolkit install via: {(uv_bin or 'uv')} pip install --python {python_exec} cuda-toolkit[all]")
        return False
    if uv_bin is None:
        print("Could not find uv in PATH or next to the bootstrap Python; cannot install CUDA toolkit into the Python environment automatically.")
        return False

    if not prompt_bool(
        "NVIDIA GPU detected but CUDA toolkit is incomplete. Install 'cuda-toolkit[all]' into the bundle Python environment with uv now?",
        default=True,
    ):
        return False

    try:
        subprocess.run([uv_bin, "pip", "install", "--python", python_exec, "cuda-toolkit[all]"], check=True)
    except Exception as exc:
        print(f"Could not install cuda-toolkit[all] with uv: {exc}")
        print("Falling back to a smaller nvcc-only Python package.")
        try:
            subprocess.run([uv_bin, "pip", "install", "--python", python_exec, "nvidia-cuda-nvcc"], check=True)
        except Exception as fallback_exc:
            print(f"Could not install nvidia-cuda-nvcc with uv: {fallback_exc}")
            return False
    try:
        subprocess.run([uv_bin, "pip", "install", "--python", python_exec, "nvidia-nccl-cu12"], check=True)
    except Exception as exc:
        print(f"Could not install optional multi-GPU NCCL package with uv: {exc}")
    nvcc_path = locate_nvcc() or locate_nvcc_for_python(python_exec)
    cuda_root = locate_cuda_root_for_python(python_exec)
    normalize_python_cuda_layout(cuda_root)
    if _export_nvcc_path(nvcc_path):
        print(f"Using nvcc from Python environment: {nvcc_path}")
    if _export_cuda_root(cuda_root):
        print(f"Using CUDA toolkit root from Python environment: {cuda_root}")
    nccl_root = locate_nccl_root_for_python(python_exec)
    if _export_nccl_root(nccl_root):
        print(f"Using NCCL root from Python environment: {nccl_root}")
    return nvcc_path is not None


def maybe_install_nccl_via_uv(python_exec: str, dry_run: bool) -> bool:
    uv_bin = resolve_uv_executable()
    if dry_run:
        print(f"[dry-run] would offer NCCL install via: {(uv_bin or 'uv')} pip install --python {python_exec} nvidia-nccl-cu12")
        return False
    if uv_bin is None:
        return False
    try:
        subprocess.run([uv_bin, "pip", "install", "--python", python_exec, "nvidia-nccl-cu12"], check=True)
    except Exception as exc:
        print(f"Could not install optional multi-GPU NCCL package with uv: {exc}")
        return False
    nccl_root = locate_nccl_root_for_python(python_exec)
    if _export_nccl_root(nccl_root):
        print(f"Using NCCL root from Python environment: {nccl_root}")
        return True
    return False


def maybe_install_cuda_toolkit(gpu_present: bool, dry_run: bool, prefer_source_cuda: bool, python_exec: str) -> bool:
    if not (sys.platform.startswith("linux") and gpu_present and prefer_source_cuda):
        return False
    if locate_nvcc():
        return True
    if maybe_install_cuda_toolkit_via_uv(python_exec, dry_run):
        return True

    package = detect_cuda_toolkit_package()
    if package is None:
        print("NVIDIA GPU detected but neither uv-installed nvcc nor an apt-installable CUDA toolkit package was found. Continuing without nvcc.")
        return False

    if dry_run:
        print(f"[dry-run] would offer CUDA toolkit install via: {' '.join(_sudo_prefix() + ['apt-get', 'install', '-y', package])}")
        return False

    if not prompt_bool(
        f"NVIDIA GPU detected but nvcc is missing. Install CUDA toolkit package '{package}' with sudo now?",
        default=True,
    ):
        return False

    subprocess.run(_sudo_prefix() + ["apt-get", "update"], check=True)
    subprocess.run(_sudo_prefix() + ["apt-get", "install", "-y", package], check=True)
    return locate_nvcc() is not None


def missing_source_build_packages() -> list[str]:
    missing: list[str] = []
    if shutil.which("git") is None:
        missing.append("git")
    if shutil.which("cc") is None or shutil.which("c++") is None:
        missing.append("build-essential")
    return missing


def maybe_install_source_build_prereqs(dry_run: bool) -> None:
    missing = missing_source_build_packages()
    if not missing:
        return
    if dry_run:
        print(f"[dry-run] would offer source-build prerequisites via: {' '.join(_sudo_prefix() + ['apt-get', 'install', '-y'] + missing)}")
        return
    approved: list[str] = []
    for package in missing:
        if prompt_bool(f"Install source-build prerequisite '{package}' with sudo apt now?", default=True):
            approved.append(package)
    if approved:
        subprocess.run(_sudo_prefix() + ["apt-get", "update"], check=True)
        subprocess.run(_sudo_prefix() + ["apt-get", "install", "-y", *approved], check=True)
    remaining = missing_source_build_packages()
    if remaining:
        raise RuntimeError(
            "Missing native build prerequisites remain: "
            + ", ".join(remaining)
            + ". The bootstrap environment already provides cmake/ninja/compiletools via uv, but source builds still need a real host compiler toolchain."
        )


def source_tree_supports_flag(src_dir: Path, flag_name: str) -> bool:
    for pattern in ("CMakeLists.txt", "*.cmake"):
        for candidate in src_dir.rglob(pattern):
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if flag_name in text:
                return True
    return False


def parse_ollama_models_from_systemctl(text: str) -> str | None:
    for line in text.splitlines():
        if "OLLAMA_MODELS=" not in line:
            continue
        match = re.search(r'OLLAMA_MODELS=([^"\n]+)', line)
        if match:
            return match.group(1).strip()
    return None


def detect_ollama_models_dir() -> Path | None:
    if env_value := os.environ.get("OLLAMA_MODELS"):
        return Path(env_value).expanduser()

    try:
        result = subprocess.run(
            ["systemctl", "cat", "ollama"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        result = None

    if result:
        parsed = parse_ollama_models_from_systemctl(result.stdout)
        if parsed:
            return Path(parsed).expanduser()

    for candidate in (
        Path("/var/llamacpp_models"),
        Path("/var/lib/ollama/models"),
        Path("/usr/share/ollama/.ollama/models"),
        Path.home() / ".ollama" / "models",
    ):
        try:
            if candidate.exists():
                # Just check if it has ANY .gguf files in top or one level deep
                # This is much faster than rglob() for huge directories
                if any(candidate.glob("*.gguf")) or any(candidate.glob("*/*.gguf")):
                    return candidate
        except (PermissionError, Exception):
            continue

    # Try paths without GGUF check if none found with GGUF
    for candidate in (
        Path("/var/llamacpp_models"),
        Path("/var/lib/ollama/models"),
        Path.home() / ".ollama" / "models",
    ):
        try:
            if candidate.exists():
                return candidate
        except (PermissionError, Exception):
            continue

    return None


def existing_public_port(mode: str) -> int | None:
    env_path = env_path_for_mode(mode)
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean.startswith("LLAMACPP_PUBLIC_PORT="):
            raw = clean.split("=", 1)[1].strip()
            if raw.isdigit():
                return int(raw)
    return None


def env_path_for_mode(mode: str) -> Path:
    base = Path("/etc/llamacpp-superserver") if mode == "system" else Path.home() / ".config/llamacpp-superserver"
    return base / ENV_BASENAME


def legacy_env_path_for_mode(mode: str) -> Path:
    return Path("/etc/llamacpp/llamacpp-stack.env") if mode == "system" else Path.home() / ".config/llamacpp/llamacpp-stack.env"


def detect_existing_mode() -> str | None:
    if env_path_for_mode("system").exists() or legacy_env_path_for_mode("system").exists():
        return "system"
    if env_path_for_mode("user").exists() or legacy_env_path_for_mode("user").exists():
        return "user"
    return None


def existing_models_dir(mode: str) -> Path | None:
    for env_path in (env_path_for_mode(mode), legacy_env_path_for_mode(mode)):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean.startswith("LLAMACPP_MODELS="):
                return Path(clean.split("=", 1)[1].strip()).expanduser()
    return None


def derive_models_dir(base: Path | None, mode: str) -> Path:
    if base:
        sibling = base.parent / "llamacpp_models"
        if mode == "system" or os.access(base.parent, os.W_OK):
            return sibling
    if mode == "system":
        return Path("/var/lib/llamacpp-superserver/models")
    return Path.home() / ".local/share/llamacpp-superserver/models"


def choose_layout(mode: str | None, public_host: str, public_port: int | None, models_dir: Path | None = None, args: argparse.Namespace | None = None) -> InstallLayout:
    resolved_mode = mode or detect_existing_mode() or ("system" if os.geteuid() == 0 else "user")
    resolved_port = choose_default_swap_port(public_host, resolved_mode, public_port, args=args)
    ollama_models = detect_ollama_models_dir()
    resolved_models_dir = models_dir or existing_models_dir(resolved_mode) or derive_models_dir(ollama_models, resolved_mode)
    if resolved_mode == "system":
        install_root = Path("/opt/llamacpp-superserver")
        state_dir = Path("/var/lib/llamacpp-superserver")
        config_dir = Path("/etc/llamacpp-superserver")
        run_dir = Path("/run/llamacpp-superserver")
        user = DEFAULT_SERVICE_USER
        group = user
        bin_dir = Path("/usr/local/bin")
    else:
        install_root = Path.home() / ".local/opt/llamacpp-superserver"
        state_dir = Path.home() / ".local/state/llamacpp-superserver"
        config_dir = Path.home() / ".config/llamacpp-superserver"
        run_dir = Path.home() / ".local/run/llamacpp-superserver"
        user = os.environ.get("USER", "unknown")
        group = user
        bin_dir = Path.home() / ".local/bin"
    return InstallLayout(
        mode=resolved_mode,
        state_dir=state_dir,
        bin_dir=bin_dir,
        install_root=install_root,
        models_dir=resolved_models_dir,
        config_dir=config_dir,
        run_dir=run_dir,
        service_user=user,
        service_group=group,
        public_host=public_host,
        public_port=resolved_port,
        manager_socket=run_dir / "manager.sock",
        python_root=install_root / "python",
        runtime_venv=install_root / "venv",
        cuda_root=install_root / "cuda",
        nccl_root=install_root / "nccl",
    )


def _user_exists(name: str) -> bool:
    try:
        import pwd

        pwd.getpwnam(name)
        return True
    except Exception:
        return False


def ensure_system_identity(layout: InstallLayout, dry_run: bool) -> None:
    if layout.mode != "system" or _user_exists(layout.service_user):
        return
    cmd = [
        "useradd",
        "--system",
        "--user-group",
        "--no-create-home",
        "--home-dir",
        "/nonexistent",
        "--shell",
        "/usr/sbin/nologin",
        layout.service_user,
    ]
    if dry_run:
        print(f"[dry-run] would create service user: {' '.join(_sudo_prefix() + cmd)}")
        return
    _run(_sudo_prefix() + cmd)


def choose_llamaswap_asset(release: dict) -> dict:
    for asset in release.get("assets", []):
        if asset["name"].endswith("linux_amd64.tar.gz"):
            return asset
    raise RuntimeError("No Linux AMD64 llama-swap asset found in latest release.")


def choose_llamacpp_linux_asset(release: dict) -> dict | None:
    candidates = []
    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if "ubuntu-x64" in name and name.endswith(".tar.gz"):
            candidates.append(asset)
    return candidates[0] if candidates else None


def detect_native_llama_server() -> Path | None:
    candidates = [
        shutil.which("llama-server"),
        "/usr/bin/llama-server",
        "/usr/local/bin/llama-server",
        "/opt/homebrew/bin/llama-server",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def detect_native_llama_cpp_package() -> str | None:
    if shutil.which("apt-cache") is None:
        return None
    patterns = ("^llama", "llama.cpp", "llama-cpp", "llamacpp")
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["apt-cache", "search", pattern],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        for line in result.stdout.splitlines():
            package = line.split(" ", 1)[0].strip()
            if package and ("llama" in package or "ggml" in package):
                return package
    return None


def maybe_install_native_llama_cpp(dry_run: bool) -> Path | None:
    existing = detect_native_llama_server()
    if existing:
        return existing
    package = detect_native_llama_cpp_package()
    if package is None:
        return None
    if dry_run:
        print(f"[dry-run] would offer native llama.cpp package install via: {' '.join(_sudo_prefix() + ['apt-get', 'install', '-y', package])}")
        return Path("/usr/bin/llama-server")
    if not prompt_bool(
        f"Install native llama.cpp package '{package}' with sudo apt now?",
        default=True,
    ):
        return None
    subprocess.run(_sudo_prefix() + ["apt-get", "update"], check=True)
    subprocess.run(_sudo_prefix() + ["apt-get", "install", "-y", package], check=True)
    return detect_native_llama_server()


def _extract_tarball(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)


def _find_executable(root: Path, name: str) -> Path:
    for candidate in root.rglob(name):
        if candidate.is_file():
            candidate.chmod(candidate.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return candidate

    # If we didn't find it, show what WAS there for debugging
    contents = []
    try:
        if root.exists():
            for p in root.rglob("*"):
                contents.append(str(p.relative_to(root)))
    except Exception:
        pass

    hint = " Try re-running the installer with binary updates enabled (remove --no-update-binaries or pass --update-binaries)."
    detail = f" (contents: {', '.join(contents[:20])})" if contents else f" (directory is empty or missing){hint}"
    raise RuntimeError(f"Executable {name} not found under {root}{detail}")


def _is_executable_working(path: Path, timeout: float = 3.0) -> bool:
    """Return True if the executable at `path` appears runnable.

    This makes a conservative check by ensuring the file exists, is executable,
    and responds successfully to one of a few common flags. Returns False on
    any failure or timeout.
    """
    if path is None:
        return False
    try:
        if not path.exists() or not path.is_file():
            return False
        try:
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
        if not os.access(str(path), os.X_OK):
            return False
        for arg in ("--version", "-v", "-h", "--help"):
            try:
                res = subprocess.run([str(path), arg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
                if res.returncode == 0:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _resolve_existing_stable_target(install_root: Path, stable_link: Path, name: str) -> Path | None:
    candidates: list[Path] = []
    for extracted_root in install_root.glob("*.d"):
        if not extracted_root.is_dir():
            continue
        for candidate in extracted_root.rglob(name):
            if not candidate.is_file():
                continue
            candidates.append(candidate)

    if not candidates:
        return None

    def _sort_key(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    stable_abs = os.path.abspath(str(stable_link))
    for candidate in sorted(candidates, key=_sort_key, reverse=True):
        candidate_abs = os.path.abspath(str(candidate))
        if candidate_abs == stable_abs:
            continue
        candidate.chmod(candidate.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return candidate
    return None


def _is_self_referential_symlink(path: Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        raw_target = os.readlink(path)
    except OSError:
        return False
    resolved_target = os.path.abspath(os.path.join(str(path.parent), raw_target))
    return resolved_target == os.path.abspath(str(path))


def _link_stable_binary(target: Path, link_path: Path, dry_run: bool) -> Path:
    if dry_run:
        return link_path
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_abs = os.path.abspath(str(target))
    link_abs = os.path.abspath(str(link_path))
    if target_abs == link_abs:
        if link_path.is_symlink():
            raw_target = os.readlink(link_path)
            resolved_target = os.path.abspath(os.path.join(str(link_path.parent), raw_target))
            if resolved_target == link_abs:
                raise RuntimeError(
                    f"Detected broken self-referential symlink at {link_path}. "
                    "Re-run installer with binary update enabled to recreate llama-server."
                )
        return link_path
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path


def _render_env(
    layout: InstallLayout,
    llama_server: Path,
    llamaswap: Path,
    python_exec: str,
    python_path: Path,
    idle_ttl: int,
    cuda_root: Path | None,
    nccl_root: Path | None,
) -> str:
    extra = ""
    if cuda_root is not None:
        extra = textwrap.dedent(
            f"""\
            LLAMACPP_CUDA_ROOT={cuda_root}
            """
        )
    if nccl_root is not None:
        extra += textwrap.dedent(
            f"""\
            LLAMACPP_NCCL_ROOT={nccl_root}
            """
        )
    return ENV_FILE_HEADER + textwrap.dedent(
        f"""\
        LLAMACPP_STACK_ROOT={layout.install_root}
        LLAMACPP_MODELS={layout.models_dir}
        LLAMACPP_CONFIG={layout.state_dir / 'config.yaml'}
        LLAMACPP_CATALOG={layout.state_dir / 'catalog.json'}
        LLAMACPP_SERVER_CONFIG={layout.config_dir / SERVER_CONFIG_BASENAME}
        LLAMACPP_MANAGER_SOCKET={layout.manager_socket}
        LLAMA_SERVER_BIN={llama_server}
        LLAMASWAP_BIN={llamaswap}
        LLAMACPP_PUBLIC_HOST={layout.public_host}
        LLAMACPP_PUBLIC_PORT={layout.public_port}
        LLAMACPP_API_PORT={layout.public_port - 1}
        LLAMACPP_IDLE_TTL={idle_ttl}
        LLAMACPP_INSTALL_MODE={layout.mode}
        LLAMACPP_SERVICE_NAME={SWAP_SERVICE_NAME}
        PYTHON_BIN={python_exec}
        LLAMACPP_PYTHONPATH={python_path}
        {extra}"""
    )


def render_manager_service(layout: InstallLayout) -> str:
    wanted_by = "multi-user.target" if layout.mode == "system" else "default.target"
    identity_lines: list[str] = []
    runtime_lines: list[str] = []
    if layout.mode == "system":
        identity_lines = [f"User={layout.service_user}", f"Group={layout.service_group}"]
        runtime_lines = [f"RuntimeDirectory={layout.run_dir.name}", "RuntimeDirectoryMode=0755"]
    service_lines = [
        "[Unit]",
        "Description=llamacpp superserver manager",
        "After=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        *identity_lines,
        *runtime_lines,
        f"ExecStart={layout.bin_dir / MANAGER_WRAPPER_NAME}",
        "Restart=always",
        "RestartSec=2",
        "",
        "[Install]",
        f"WantedBy={wanted_by}",
    ]
    return "\n".join(service_lines) + "\n"


def render_llamaswap_service(layout: InstallLayout) -> str:
    wanted_by = "multi-user.target" if layout.mode == "system" else "default.target"
    identity_lines: list[str] = []
    runtime_lines: list[str] = []
    if layout.mode == "system":
        identity_lines = [f"User={layout.service_user}", f"Group={layout.service_group}"]
        runtime_lines = [f"RuntimeDirectory={layout.run_dir.name}", "RuntimeDirectoryMode=0755"]
    service_lines = [
        "[Unit]",
        "Description=llamacpp superserver llama-swap backend",
        f"After=network-online.target {MANAGER_SERVICE_NAME}",
        "",
        "[Service]",
        "Type=simple",
        *identity_lines,
        *runtime_lines,
        f"ExecStart={layout.bin_dir / SWAP_WRAPPER_NAME}",
        "Restart=always",
        "RestartSec=2",
        "",
        "[Install]",
        f"WantedBy={wanted_by}",
    ]
    return "\n".join(service_lines) + "\n"


def render_manager_wrapper(layout: InstallLayout) -> str:
    env_file = layout.config_dir / ENV_BASENAME
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
                if [[ ! -f {env_file} ]]; then
                    echo "[llamacpp-manager] Missing env file: {env_file}" >&2
                    exit 1
                fi
        set -a
        source {env_file}
        set +a
                if [[ -z "${{PYTHON_BIN:-}}" ]] || [[ ! -x "$PYTHON_BIN" ]]; then
                    echo "[llamacpp-manager] PYTHON_BIN is missing or not executable: '${{PYTHON_BIN:-}}'" >&2
                    exit 1
                fi
        export PYTHONPATH="$LLAMACPP_PYTHONPATH${{PYTHONPATH:+:$PYTHONPATH}}"
                if [[ -n "${{LLAMA_SERVER_BIN:-}}" ]] && [[ -e "$LLAMA_SERVER_BIN" || -L "$LLAMA_SERVER_BIN" ]]; then
                    LLAMA_SERVER_REAL="$(readlink -f "$LLAMA_SERVER_BIN" || printf '%s' "$LLAMA_SERVER_BIN")"
                    if [[ -n "$LLAMA_SERVER_REAL" ]]; then
                        LLAMA_SERVER_LIB_DIR="$(dirname "$LLAMA_SERVER_REAL")"
                        export LD_LIBRARY_PATH="$LLAMA_SERVER_LIB_DIR${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
                    fi
                else
                    echo "[llamacpp-manager] Warning: LLAMA_SERVER_BIN not found yet: '${{LLAMA_SERVER_BIN:-}}'" >&2
                fi
        if [[ -n "${{LLAMACPP_CUDA_ROOT:-}}" ]]; then
          export CUDA_PATH="$LLAMACPP_CUDA_ROOT"
          export LD_LIBRARY_PATH="$LLAMACPP_CUDA_ROOT/lib64:$LLAMACPP_CUDA_ROOT/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
        fi
        if [[ -n "${{LLAMACPP_NCCL_ROOT:-}}" ]]; then
          export LD_LIBRARY_PATH="$LLAMACPP_NCCL_ROOT/lib64:$LLAMACPP_NCCL_ROOT/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
        fi
        mkdir -p "$(dirname "$LLAMACPP_MANAGER_SOCKET")" || true
        exec "$PYTHON_BIN" -m llamacpp_stack.cli \\
          --models-dir "$LLAMACPP_MODELS" \\
          --config "$LLAMACPP_CONFIG" \\
          --catalog "$LLAMACPP_CATALOG" \\
          --llama-server "$LLAMA_SERVER_BIN" \\
          --public-host "$LLAMACPP_PUBLIC_HOST" \\
          --public-port "$LLAMACPP_PUBLIC_PORT" \\
          daemon
        """
    )


def render_llamaswap_wrapper(layout: InstallLayout) -> str:
    env_file = layout.config_dir / ENV_BASENAME
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        set -a
        source {env_file}
        set +a
        export PYTHONPATH="$LLAMACPP_PYTHONPATH${{PYTHONPATH:+:$PYTHONPATH}}"
        if [[ -n "${{LLAMA_SERVER_BIN:-}}" ]] && [[ -e "$LLAMA_SERVER_BIN" || -L "$LLAMA_SERVER_BIN" ]]; then
          LLAMA_SERVER_REAL="$(readlink -f "$LLAMA_SERVER_BIN" || printf '%s' "$LLAMA_SERVER_BIN")"
          if [[ -n "$LLAMA_SERVER_REAL" ]]; then
            LLAMA_SERVER_LIB_DIR="$(dirname "$LLAMA_SERVER_REAL")"
            export LD_LIBRARY_PATH="$LLAMA_SERVER_LIB_DIR${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
          fi
        fi
        if [[ -n "${{LLAMACPP_CUDA_ROOT:-}}" ]]; then
          export CUDA_PATH="$LLAMACPP_CUDA_ROOT"
          export LD_LIBRARY_PATH="$LLAMACPP_CUDA_ROOT/lib64:$LLAMACPP_CUDA_ROOT/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
        fi
        if [[ -n "${{LLAMACPP_NCCL_ROOT:-}}" ]]; then
          export LD_LIBRARY_PATH="$LLAMACPP_NCCL_ROOT/lib64:$LLAMACPP_NCCL_ROOT/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
        fi
        exec "$LLAMASWAP_BIN" \\
          --config "$LLAMACPP_CONFIG" \\
          --listen "$LLAMACPP_PUBLIC_HOST:$LLAMACPP_PUBLIC_PORT" \\
          --watch-config
        """
    )


def render_initial_config(config_path: Path, start_port: int = 18080) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = CONFIG_YAML_HEADER + textwrap.dedent(
        f"""\
        healthCheckTimeout: 600
        logLevel: info
        logToStdout: proxy
        startPort: {start_port}
        sendLoadingState: true
        includeAliasesInList: true
        models: {{}}
        """
    )
    config_path.write_text(payload, encoding="utf-8")


def ensure_dirs(layout: InstallLayout) -> None:
    for path in (
        layout.state_dir,
        layout.models_dir,
        layout.config_dir,
        layout.run_dir,
        layout.install_root,
        layout.bin_dir,
        layout.python_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


def chown_tree(path: Path, user: str, group: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    shutil.chown(path, user=user, group=group)
    if path.is_dir() and not path.is_symlink():
        for item in path.rglob("*"):
            try:
                shutil.chown(item, user=user, group=group)
            except FileNotFoundError:
                pass


def ensure_service_writable_dirs(layout: InstallLayout, dry_run: bool) -> None:
    if layout.mode != "system":
        return
    targets = [layout.state_dir, layout.run_dir]
    if dry_run:
        print(
            "[dry-run] would chown service-writable dirs to "
            f"{layout.service_user}:{layout.service_group}: {', '.join(str(path) for path in targets)}"
        )
        return
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)
        chown_tree(path, layout.service_user, layout.service_group)


def desired_models_dir_owner(layout: InstallLayout) -> tuple[str, str]:
    if layout.mode == "user":
        user_name = os.environ.get("SUDO_USER") or os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name
        try:
            gid = pwd.getpwnam(user_name).pw_gid
            group_name = grp.getgrgid(gid).gr_name
        except Exception:
            group_name = os.environ.get("SUDO_GID") or os.environ.get("USER") or user_name
        return user_name, group_name
    return layout.service_user, layout.service_group


def _sudo_prefix() -> list[str]:
    return [] if os.geteuid() == 0 else ["sudo"]


def ensure_models_dir_ready(layout: InstallLayout, dry_run: bool) -> None:
    owner_user, owner_group = desired_models_dir_owner(layout)
    models_dir = layout.models_dir
    local_ready = False
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(models_dir, 0o775)
        if layout.mode == "system":
            shutil.chown(models_dir, user=owner_user, group=owner_group)
        local_ready = os.access(models_dir, os.W_OK | os.X_OK)
    except PermissionError:
        local_ready = False
    except Exception:
        local_ready = models_dir.exists() and os.access(models_dir, os.W_OK | os.X_OK)

    if local_ready:
        return

    sudo_cmd = _sudo_prefix()
    if dry_run:
        print(
            f"[dry-run] would ensure models dir {models_dir} exists and is owned by "
            f"{owner_user}:{owner_group} with mode 0775 via {' '.join(sudo_cmd + ['mkdir', '-p', str(models_dir)])}"
        )
        return

    if not prompt_bool(
        f"Models directory {models_dir} is not writable. Create/fix it with sudo for {owner_user}:{owner_group}?",
        default=True,
    ):
        raise RuntimeError(
            f"Models directory {models_dir} is not writable and sudo fix was declined."
        )

    _run(sudo_cmd + ["mkdir", "-p", str(models_dir)])
    _run(sudo_cmd + ["chown", f"{owner_user}:{owner_group}", str(models_dir)])
    _run(sudo_cmd + ["chmod", "0775", str(models_dir)])

    if not (models_dir.exists() and os.access(models_dir, os.W_OK | os.X_OK)):
        raise RuntimeError(
            f"Models directory {models_dir} still is not writable after attempting to fix permissions."
        )


def determine_build_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def _systemctl_cmd(mode: str, *args: str) -> list[str]:
    if mode == "system":
        return _sudo_prefix() + ["systemctl", *args]
    return ["systemctl", "--user", *args]


def ensure_runtime_python(layout: InstallLayout, dry_run: bool) -> tuple[Path, Path]:
    runtime_python = layout.runtime_venv / "bin" / "python"
    python_path = layout.python_root
    uv_bin = resolve_uv_executable()
    if dry_run:
        print(f"[dry-run] would create runtime venv at {layout.runtime_venv}")
        print(f"[dry-run] would copy Python package to {python_path / 'llamacpp_stack'}")
        return runtime_python, python_path

    if uv_bin is None:
        raise RuntimeError("uv is required to create the runtime Python environment.")

    source_pkg = Path(__file__).resolve().parent
    target_pkg = python_path / "llamacpp_stack"
    python_path.mkdir(parents=True, exist_ok=True)
    if target_pkg.exists():
        shutil.rmtree(target_pkg)
    shutil.copytree(
        source_pkg,
        target_pkg,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".bootstrap-venv",
            ".pytest_cache",
            ".mypy_cache",
        ),
    )

    if layout.runtime_venv.exists():
        shutil.rmtree(layout.runtime_venv)
    _run([uv_bin, "venv", "--python", sys.executable, str(layout.runtime_venv)])
    _run(
        [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(runtime_python),
            "requests",
            "pyyaml",
            "huggingface_hub",
            "hf_transfer",
        ]
    )
    return runtime_python, python_path


def _sync_dir(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would sync {src} to {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    # Using cp -au is much faster than shutil.copytree for large directories like CUDA
    # -a: archive (preserve links, permissions, etc)
    # -u: update (only copy when src is newer than dst or dst is missing)
    # Using ./. to copy contents of src into dst
    subprocess.run(["cp", "-au", f"{src}/.", str(dst)], check=True)


def sync_cuda_runtime(layout: InstallLayout, python_exec: str, dry_run: bool) -> Path | None:
    cuda_root = locate_cuda_root_for_python(python_exec)
    if cuda_root is None or not cuda_root.exists():
        return layout.cuda_root if layout.cuda_root.exists() else None
    normalize_python_cuda_layout(cuda_root)
    _sync_dir(cuda_root, layout.cuda_root, dry_run)
    if not dry_run:
        normalize_python_cuda_layout(layout.cuda_root)
    return layout.cuda_root


def sync_nccl_runtime(layout: InstallLayout, python_exec: str, dry_run: bool) -> Path | None:
    nccl_root = locate_nccl_root_for_python(python_exec)
    if nccl_root is None or not nccl_root.exists():
        return layout.nccl_root if layout.nccl_root.exists() else None
    _sync_dir(nccl_root, layout.nccl_root, dry_run)
    return layout.nccl_root


def build_llama_cpp_from_source(
    release: dict,
    install_root: Path,
    enable_tls: bool,
    dry_run: bool,
    python_exec: str,
    enable_cuda: bool,
) -> Path:
    tag = release["tag_name"]
    source_url = f"https://github.com/{DEFAULT_LLAMA_CPP_REPO}/archive/refs/tags/{tag}.tar.gz"
    src_archive = install_root / f"{tag}.tar.gz"
    src_dir = install_root / f"llama.cpp-{tag}"
    build_dir = src_dir / "build"
    if dry_run:
        print(f"[dry-run] would download source {source_url}")
        return build_dir / "bin/llama-server"

    _download(source_url, src_archive)
    if src_dir.exists():
        shutil.rmtree(src_dir)
    _extract_tarball(src_archive, install_root)

    cmake_args = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        "-DLLAMA_BUILD_SERVER=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_NATIVE=ON",
    ]
    if source_tree_supports_flag(src_dir, "GGML_LTO"):
        cmake_args.append("-DGGML_LTO=ON")
    if enable_cuda:
        cmake_args.append("-DGGML_CUDA=ON")
    else:
        cmake_args.append("-DGGML_CUDA=OFF")
    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])
    if enable_cuda and (arch := detect_cuda_arch()):
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    build_env = os.environ.copy()
    nvcc_path = build_env.get("CUDACXX") or locate_nvcc() or locate_nvcc_for_python(python_exec)
    if enable_cuda and nvcc_path:
        _export_nvcc_path(nvcc_path)
        build_env = os.environ.copy()
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={Path(nvcc_path).resolve()}")
    rpath_dirs: list[str] = []
    cuda_root = Path(build_env["CUDAToolkit_ROOT"]) if build_env.get("CUDAToolkit_ROOT") else locate_cuda_root_for_python(python_exec)
    if enable_cuda and cuda_root:
        normalize_python_cuda_layout(cuda_root)
        _export_cuda_root(cuda_root)
        build_env = os.environ.copy()
        cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
        for lib_dir in (cuda_root / "lib64", cuda_root / "lib", cuda_root / "targets" / "x86_64-linux" / "lib"):
            if lib_dir.exists():
                rpath_dirs.append(str(lib_dir))
    nccl_root = locate_nccl_root_for_python(python_exec)
    if enable_cuda and nccl_root:
        _export_nccl_root(nccl_root)
        build_env = os.environ.copy()
        include_dir = nccl_root / "include"
        library = None
        for lib_dir in (nccl_root / "lib64", nccl_root / "lib"):
            if lib_dir.exists():
                rpath_dirs.append(str(lib_dir))
                library = next(iter(lib_dir.glob("libnccl.so*")), None)
                if library:
                    break
        if include_dir.exists() and library:
            cmake_args.append(f"-DNCCL_INCLUDE_DIR={include_dir}")
            cmake_args.append(f"-DNCCL_LIBRARY={library}")
    if rpath_dirs:
        rpath_value = ";".join(dict.fromkeys(rpath_dirs))
        cmake_args.append(f"-DCMAKE_BUILD_RPATH={rpath_value}")
        cmake_args.append(f"-DCMAKE_INSTALL_RPATH={rpath_value}")
        cmake_args.append("-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON")
        cmake_args.append("-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON")
    if enable_cuda:
        for flag in ("GGML_CUDA_GRAPHS", "GGML_CUDA_FA_ALL_QUANTS"):
            if source_tree_supports_flag(src_dir, flag):
                cmake_args.append(f"-D{flag}=ON")
    if enable_tls:
        for flag in ("LLAMA_CURL", "LLAMA_HTTP_SERVER"):
            if source_tree_supports_flag(src_dir, flag):
                cmake_args.append(f"-D{flag}=ON")
    build_jobs = determine_build_jobs()
    _run(cmake_args, env=build_env)
    _run(["cmake", "--build", str(build_dir), "--target", "llama-server", "-j", str(build_jobs)], env=build_env)
    return build_dir / "bin/llama-server"


def detect_cuda_arch() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    values = []
    for line in out:
        item = line.strip()
        if not item:
            continue
        item = item.replace(".", "")
        if item.isdigit():
            values.append(item)
    return ";".join(sorted(set(values))) if values else None


def install_release_asset(asset: dict, install_root: Path, dry_run: bool) -> Path:
    archive = install_root / asset["name"]
    extract_root = install_root / f"{asset['name']}.d"
    if dry_run:
        print(f"[dry-run] would download {asset['browser_download_url']}")
        return extract_root
    _download(asset["browser_download_url"], archive)
    if extract_root.exists():
        if extract_root.is_dir() and not extract_root.is_symlink():
            shutil.rmtree(extract_root)
        else:
            extract_root.unlink()
    _extract_tarball(archive, extract_root)
    return extract_root


def write_manifest(layout: InstallLayout, llama_cpp_release: dict, llamaswap_release: dict, strategy: str, dry_run: bool) -> None:
    payload = {
        "mode": layout.mode,
        "models_dir": str(layout.models_dir),
        "public_host": layout.public_host,
        "public_port": layout.public_port,
        "llama_cpp_tag": llama_cpp_release["tag_name"],
        "llamaswap_tag": llamaswap_release["tag_name"],
        "llama_cpp_strategy": strategy,
    }
    if dry_run:
        print(json.dumps(payload, indent=2))
        return
    (layout.state_dir / "install-manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_install_manifest(layout: InstallLayout) -> dict[str, object]:
    manifest_path = layout.state_dir / "install-manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def detect_existing_llama_cpp_mode(layout: InstallLayout) -> str:
    payload = read_install_manifest(layout)
    strategy = str(payload.get("llama_cpp_strategy") or "").strip()
    if strategy == "native":
        return "native"
    if strategy == "binary":
        return "prebuilt"
    if strategy.startswith("source-build"):
        return "source"
    return "source"


def is_existing_install(layout: InstallLayout) -> bool:
    return (layout.state_dir / "install-manifest.json").exists() or layout.install_root.exists()


def install_systemd_units(layout: InstallLayout, dry_run: bool) -> None:
    if layout.mode == "system":
        systemd_dir = Path("/etc/systemd/system")
        env_dir = layout.config_dir
        reload_cmd = _systemctl_cmd(layout.mode, "daemon-reload")
        enable_cmd = _systemctl_cmd(layout.mode, "enable", "--now", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME)
    else:
        systemd_dir = Path.home() / ".config/systemd/user"
        env_dir = layout.config_dir
        reload_cmd = _systemctl_cmd(layout.mode, "daemon-reload")
        enable_cmd = _systemctl_cmd(layout.mode, "enable", "--now", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME)

    if dry_run:
        print(f"[dry-run] would write units to {systemd_dir} and run: {' '.join(reload_cmd)} && {' '.join(enable_cmd)}")
        return

    systemd_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(layout.config_dir / "systemd" / MANAGER_SERVICE_NAME, systemd_dir / MANAGER_SERVICE_NAME)
    shutil.copy2(layout.config_dir / "systemd" / SWAP_SERVICE_NAME, systemd_dir / SWAP_SERVICE_NAME)
    _run(reload_cmd)
    _run(enable_cmd)


def restart_systemd_units(layout: InstallLayout, dry_run: bool) -> bool:
    restart_cmd = _systemctl_cmd(layout.mode, "restart", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME)

    if dry_run:
        print(f"[dry-run] would run: {' '.join(restart_cmd)}")
        return True

    try:
        _run(restart_cmd)
        return True
    except Exception as exc:
        print(
            "Warning: could not restart services automatically at the end of install "
            f"({exc}). You can restart them manually with: {' '.join(restart_cmd)}"
        )
        return False


def wait_for_manager_socket(layout: InstallLayout, dry_run: bool, timeout_seconds: int = 20) -> bool:
    if dry_run:
        print(f"[dry-run] would wait for manager socket: {layout.manager_socket}")
        return True
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if layout.manager_socket.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    sock.connect(str(layout.manager_socket))
                return True
            except OSError:
                pass
        time.sleep(0.25)
    print(
        "Warning: manager socket did not become ready after service restart "
        f"({layout.manager_socket}). Automatic auto-ctx may be skipped for now."
    )
    return False


def stop_systemd_units(layout: InstallLayout, dry_run: bool) -> bool:
    stop_cmd = _systemctl_cmd(layout.mode, "stop", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME)

    if dry_run:
        print(f"[dry-run] would run: {' '.join(stop_cmd)}")
        return True

    try:
        _run(stop_cmd)
        return True
    except Exception as exc:
        print(
            "Warning: could not stop existing services before reinstall "
            f"({exc}). Install will continue and services will be restarted at the end."
        )
        return False


def maybe_offer_ufw_ports(layout: InstallLayout, dry_run: bool) -> None:
    if layout.mode != "system" or layout.public_host != "0.0.0.0":
        return
    if shutil.which("ufw") is None:
        return
    try:
        result = subprocess.run(["ufw", "status"], check=False, capture_output=True, text=True)
    except Exception:
        return
    if result.returncode != 0 or "Status: active" not in result.stdout:
        return
    if not prompt_bool(
        f"UFW is active. Allow TCP ports {layout.public_port - 1} and {layout.public_port} through the firewall?",
        default=True,
    ):
        return
    if dry_run:
        print(
            f"[dry-run] would run: {' '.join(_sudo_prefix() + ['ufw', 'allow', f'{layout.public_port - 1}/tcp'])}"
            f" && {' '.join(_sudo_prefix() + ['ufw', 'allow', f'{layout.public_port}/tcp'])}"
        )
        return
    _run(_sudo_prefix() + ["ufw", "allow", f"{layout.public_port - 1}/tcp"])
    _run(_sudo_prefix() + ["ufw", "allow", f"{layout.public_port}/tcp"])


def print_install_summary(layout: InstallLayout, install_services: bool) -> None:
    ui_base_url = f"http://{layout.public_host}:{layout.public_port}"
    api_url = f"http://{layout.public_host}:{layout.public_port - 1}"
    ui_url = f"{ui_base_url}/ui/#/activity"
    help_cmd = layout.bin_dir / CLI_COMMAND
    manifest = read_install_manifest(layout)
    current_llama_cpp = str(manifest.get("llama_cpp_tag") or "unknown")
    current_llamaswap = str(manifest.get("llamaswap_tag") or "unknown")
    if layout.mode == "system":
        start_cmd = f"sudo systemctl start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
        restart_cmd = f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
        status_cmd = f"sudo systemctl status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
    else:
        start_cmd = f"systemctl --user start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
        restart_cmd = f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
        status_cmd = f"systemctl --user status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"

    print("\nInstallation complete.")
    print("Use:")
    print(f"  {CLI_COMMAND} --help")
    print(f"  {CLI_COMMAND} ps")
    print(f"  {CLI_COMMAND} list")
    print(f"  {CLI_COMMAND} run <repo-or-hf-ref>")
    print(f"Installed llama.cpp: {current_llama_cpp}")
    print(f"Installed llama-swap: {current_llamaswap}")
    print(f"Superserver API:     {api_url}")
    print(f"UI activity:         {ui_url}")
    print(f"llama-swap UI/backend: {ui_base_url}")
    if install_services:
        print(f"Services enabled and running. Check status with: {status_cmd}")
        print(f"Restart them with: {restart_cmd}")
    else:
        print("Services were not enabled automatically.")
        print(f"Start them with: {start_cmd}")
        print(f"Then check with: {status_cmd}")
    if help_cmd.exists():
        print("\nShowing command help:\n")
        try:
            subprocess.run([str(help_cmd), "--help"], check=False)
        except Exception as exc:
            print(f"Could not show {CLI_COMMAND} --help automatically: {exc}")


def _catalog_model_count(catalog_path: Path) -> int:
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(payload, list):
        return 0
    return sum(1 for item in payload if isinstance(item, dict) and item.get("model_id"))


def _models_dir_has_gguf(models_dir: Path) -> bool:
    try:
        return any(models_dir.rglob("*.gguf"))
    except Exception:
        return False


def _slugify_model_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", (value or "").lower()).strip("-._")
    return slug or "model"


def _normalize_model_id_token(value: str) -> str:
    token = (value or "").strip().lower()
    if not token:
        return ""
    # Normalize legacy artifacts so API-facing IDs are concise and stable.
    token = re.sub(r"\.gguf\b", "", token)
    token = re.sub(r"(^|[-._])gguf(?=($|[-._]))", r"\1", token)
    token = re.sub(r"[-._]?\d{5}-of-\d{5}$", "", token)
    token = re.sub(r"[-._]{2,}", "-", token)
    token = re.sub(r"[^a-z0-9._-]+", "-", token).strip("-._")
    return token


def _append_quant_suffix_if_missing(base_id: str, quant: str | None) -> str:
    quant_token = _normalize_model_id_token(quant or "")
    if not quant_token:
        return base_id
    if re.search(rf"(^|[-._]){re.escape(quant_token)}($|[-._])", base_id):
        return base_id
    return _normalize_model_id_token(f"{base_id}-{quant_token}")


def _normalized_model_id_components(repo_id: str, quant: str | None, filename: str, fallback: str) -> str:
    filename_seed = Path(filename).stem if filename else ""
    candidate = _normalize_model_id_token(filename_seed)
    candidate = _append_quant_suffix_if_missing(candidate, quant)
    if candidate:
        return candidate
    repo_tail = (repo_id or "").split("/")[-1].strip()
    candidate = _normalize_model_id_token(repo_tail)
    candidate = _append_quant_suffix_if_missing(candidate, quant)
    if candidate:
        return candidate
    if filename:
        candidate = _normalize_model_id_token(Path(filename).stem)
        candidate = _append_quant_suffix_if_missing(candidate, quant)
        if candidate:
            return candidate
    return _normalize_model_id_token(fallback) or "model"


def _infer_quant_from_filename(filename: str) -> str | None:
    upper = (filename or "").upper()
    match = re.search(
        r"(?<![A-Z0-9])(IQ\d(?:_[A-Z0-9]+)?|Q\d_K_[SML]|Q\d_[01]|Q\d_K|Q\d|BF16|F16|F32)(?![A-Z0-9])",
        upper,
    )
    return match.group(1) if match else None


def _derived_aliases_for_import(repo_id: str, quant: str | None, filename: str) -> list[str]:
    filename = (filename or "").strip()
    aliases: list[str] = []
    if not filename and not repo_id:
        return aliases

    def _append(v: str | None) -> None:
        if not v:
            return
        s = str(v).strip()
        if s and s not in aliases:
            aliases.append(s)

    # Filename variants
    if filename:
        _append(filename)
        basename = Path(filename).name
        _append(basename)
        no_ext = re.sub(r"(?i)\.gguf$", "", filename)
        _append(no_ext)
        no_shard = re.sub(r"[-._]?\d+-of-\d+$", "", no_ext)
        _append(no_shard)

    # Repo variants (with/without hf.co and optional :quant)
    repo = (repo_id or "").strip()
    if repo:
        _append(f"hf.co/{repo}")
        _append(repo)
        if quant:
            q = str(quant).strip()
            if q:
                _append(f"hf.co/{repo}:{q}")
                _append(f"{repo}:{q}")

    return aliases


def _normalize_catalog_aliases(value: object) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for raw in raw_values:
        alias = str(raw).strip()
        if alias and alias not in normalized:
            normalized.append(alias)
    return normalized


def _effective_catalog_idle_ttl(layout: InstallLayout, server_config: dict[str, object]) -> int:
    fallback = DEFAULT_IDLE_TTL
    try:
        server_idle = server_config.get("idle_ttl") if isinstance(server_config, dict) else None
        if server_idle is not None:
            fallback = int(server_idle)
    except Exception:
        fallback = DEFAULT_IDLE_TTL

    config_path = layout.state_dir / "config.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return fallback
    models = payload.get("models") or {}
    if not isinstance(models, dict):
        return fallback
    for model_config in models.values():
        if not isinstance(model_config, dict):
            continue
        ttl = model_config.get("ttl")
        if ttl is None:
            continue
        try:
            return int(ttl)
        except Exception:
            return fallback
    return fallback


def _catalog_payload_for_import(catalog_path: Path) -> tuple[list[dict[str, object]] | None, str | None]:
    if not catalog_path.exists():
        return [], None
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Could not parse existing catalog {catalog_path}: {exc}"
    if not isinstance(payload, list):
        return None, f"Catalog {catalog_path} has invalid format (expected a JSON array)."
    filtered: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            filtered.append(dict(item))
    return filtered, None


def _plan_model_id_migration(payload: list[dict[str, object]]) -> dict[str, str]:
    current_ids = {
        str(item.get("model_id") or "").strip()
        for item in payload
        if isinstance(item, dict) and str(item.get("model_id") or "").strip()
    }
    planned_ids = set(current_ids)
    mapping: dict[str, str] = {}

    for item in payload:
        if not isinstance(item, dict):
            continue
        current = str(item.get("model_id") or "").strip()
        if not current:
            continue
        target_base = _normalized_model_id_components(
            str(item.get("repo_id") or ""),
            str(item.get("quant") or "") or None,
            str(item.get("filename") or ""),
            current,
        )
        if not target_base or target_base == current:
            continue

        planned_ids.discard(current)
        target = target_base
        suffix = 2
        while target in planned_ids:
            target = f"{target_base}-{suffix}"
            suffix += 1
        planned_ids.add(target)
        mapping[current] = target

    return mapping


def _remap_model_maps(layout: InstallLayout, id_map: dict[str, str]) -> None:
    if not id_map:
        return

    server_config_path = layout.config_dir / SERVER_CONFIG_BASENAME
    if server_config_path.exists():
        try:
            payload = json.loads(server_config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
            original = payload["models"]
            remapped: dict[str, object] = {}
            changed = False
            for key, value in original.items():
                new_key = id_map.get(str(key), str(key))
                if new_key != key:
                    changed = True
                remapped[new_key] = value
            if changed:
                payload["models"] = remapped
                _ensure_server_config_metadata(payload)
                server_config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ui_config_path = layout.state_dir / "config.yaml"
    if ui_config_path.exists():
        try:
            payload = yaml.safe_load(ui_config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
            original = payload["models"]
            remapped: dict[str, object] = {}
            changed = False
            for key, value in original.items():
                new_key = id_map.get(str(key), str(key))
                if new_key != key:
                    changed = True
                remapped[new_key] = value
            if changed:
                payload["models"] = remapped
                ui_config_path.write_text(
                    yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )


def _maybe_migrate_catalog_model_ids(layout: InstallLayout, dry_run: bool, args: argparse.Namespace) -> int:
    catalog_path = layout.state_dir / "catalog.json"
    payload, error = _catalog_payload_for_import(catalog_path)
    if error or payload is None:
        return 0

    mapping = _plan_model_id_migration(payload)
    if not mapping:
        return 0

    migrate_ids = getattr(args, "migrate_model_ids", None)
    if migrate_ids is None:
        if not sys.stdin.isatty():
            migrate_ids = False
        else:
            preview_items = sorted(mapping.items())
            preview_limit = 12
            print(
                "Model IDs are the API names exposed to clients."
            )
            print(
                "New naming removes '.gguf' and shard suffixes like '-00001-of-00009' "
                "to align better with API client configs."
            )
            print("If you accept, existing references in catalog/config files are updated automatically.")
            print("Preview of model ID renames:")
            for old_id, new_id in preview_items[:preview_limit]:
                print(f"  {old_id} -> {new_id}")
            if len(preview_items) > preview_limit:
                print(f"  ... and {len(preview_items) - preview_limit} more")
            migrate_ids = prompt_bool(
                f"Apply this renaming for {len(mapping)} model ID(s) now?",
                default=False,
            )
        args.migrate_model_ids = migrate_ids

    if not migrate_ids:
        print("Keeping existing model IDs.")
        return 0

    if dry_run:
        print(f"[dry-run] would rename {len(mapping)} model ID(s) to the new naming format.")
        return len(mapping)

    updated = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        current = str(item.get("model_id") or "").strip()
        renamed = mapping.get(current)
        if not renamed:
            continue
        item["model_id"] = renamed
        updated += 1

    if updated <= 0:
        return 0

    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _remap_model_maps(layout, mapping)
    print(f"Renamed {updated} model ID(s) to the new naming format in {catalog_path}.")
    return updated


def _auto_register_local_gguf_models(layout: InstallLayout, dry_run: bool) -> int:
    catalog_path = layout.state_dir / "catalog.json"
    payload, error = _catalog_payload_for_import(catalog_path)
    if error:
        print(
            "Warning: found GGUF files but automatic catalog import was skipped because the existing "
            f"catalog is invalid ({error})."
        )
        return 0
    assert payload is not None

    existing_local_paths: set[str] = set()
    used_model_ids: set[str] = set()
    for item in payload:
        model_id = str(item.get("model_id") or "").strip()
        if model_id:
            used_model_ids.add(model_id)
        local_path = str(item.get("local_path") or "").strip()
        if local_path:
            try:
                existing_local_paths.add(str(Path(local_path).resolve()))
            except Exception:
                existing_local_paths.add(local_path)

    candidates: list[Path] = []
    try:
        for gguf_path in sorted(layout.models_dir.rglob("*.gguf")):
            if not gguf_path.is_file():
                continue
            lowered = gguf_path.name.lower()
            if "mmproj" in lowered:
                continue
            shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", lowered)
            if shard and shard.group(1) != "00001":
                continue
            resolved_path = str(gguf_path.resolve())
            if resolved_path in existing_local_paths:
                continue
            candidates.append(gguf_path)
    except Exception:
        return 0

    if not candidates:
        return 0

    imported: list[dict[str, object]] = []
    for gguf_path in candidates:
        try:
            parent_rel = gguf_path.parent.relative_to(layout.models_dir)
            repo_id = parent_rel.as_posix() if str(parent_rel) != "." else "."
        except Exception:
            repo_id = "."
        quant = _infer_quant_from_filename(gguf_path.name)
        base_model_id = _normalized_model_id_components(repo_id, quant, gguf_path.name, gguf_path.stem)
        model_id = base_model_id
        suffix = 2
        while model_id in used_model_ids:
            model_id = f"{base_model_id}-{suffix}"
            suffix += 1
        used_model_ids.add(model_id)
        imported.append(
            {
                "model_id": model_id,
                "repo_id": repo_id,
                "quant": quant,
                "filename": gguf_path.name,
                "local_path": str(gguf_path),
                "description": f"local / {gguf_path.name}",
                "aliases": _derived_aliases_for_import(repo_id, quant, gguf_path.name),
            }
        )

    if dry_run:
        print(
            f"[dry-run] would auto-register {len(imported)} GGUF file(s) into {catalog_path} "
            "because catalog had no entries."
        )
        return len(imported)

    payload.extend(imported)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Auto-registered {len(imported)} GGUF file(s) into catalog during reinstall: {catalog_path}"
    )
    return len(imported)


def _ensure_basic_server_config(layout: InstallLayout) -> None:
    server_config_path = layout.config_dir / SERVER_CONFIG_BASENAME
    catalog_path = layout.state_dir / "catalog.json"
    if not catalog_path.exists():
        return

    try:
        catalog_raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(catalog_raw, list):
        return

    # Update server config (Manager config)
    try:
        if server_config_path.exists():
            server_config = json.loads(server_config_path.read_text(encoding="utf-8"))
        else:
            server_config = {}
    except Exception:
        server_config = {}

    if "models" not in server_config:
        server_config["models"] = {}
    had_meta = isinstance(server_config.get("_meta"), dict)
    if "llama_server_defaults" not in server_config:
        server_config["llama_server_defaults"] = {}
    if not isinstance(server_config.get("llama_server_defaults"), dict):
        server_config["llama_server_defaults"] = {}
    _ensure_server_config_metadata(server_config)
    _ensure_llama_server_defaults_file(layout.config_dir)

    default_ctx = 8192
    default_n_gpu_layers = 999

    server_changed = not had_meta
    catalog_changed = False

    if _merge_missing_llama_server_defaults(server_config["llama_server_defaults"], layout.config_dir):
        server_changed = True

    # If server-level defaults are missing, try to infer common defaults from catalog
    try:
        total_models = sum(1 for m in catalog_raw if isinstance(m, dict) and m.get("model_id"))
        if total_models > 0:
            n_gpu_counts: dict[int, int] = {}
            ts_counts: dict[str, int] = {}
            ttl_counts: dict[int, int] = {}
            for model in catalog_raw:
                if not isinstance(model, dict):
                    continue
                # n_gpu_layers
                if "n_gpu_layers" in model:
                    try:
                        n = int(model.get("n_gpu_layers"))
                        n_gpu_counts[n] = n_gpu_counts.get(n, 0) + 1
                    except Exception:
                        pass
                # tensor_split
                if "tensor_split" in model and model.get("tensor_split") is not None:
                    ts = str(model.get("tensor_split")).strip()
                    ts_counts[ts] = ts_counts.get(ts, 0) + 1
                # ttl (per-model may use "ttl" or "idle_ttl")
                ttl_raw = model.get("ttl") if "ttl" in model else model.get("idle_ttl")
                if ttl_raw is not None:
                    try:
                        t = int(ttl_raw)
                        ttl_counts[t] = ttl_counts.get(t, 0) + 1
                    except Exception:
                        pass

            def _majority(counts: dict) -> tuple[object, int] | None:
                if not counts:
                    return None
                best, cnt = max(counts.items(), key=lambda kv: kv[1])
                return (best, cnt)

            # Promote n_gpu_layers to global default when a clear majority exists
            if not server_config.get("llama_server_defaults"):
                server_config.setdefault("llama_server_defaults", {})
            majority = _majority(n_gpu_counts)
            if majority and majority[1] >= max(2, (total_models // 2) + 1) and majority[0] != default_n_gpu_layers:
                server_config["llama_server_defaults"]["n_gpu_layers"] = int(majority[0])
                server_changed = True

            majority_ts = _majority(ts_counts)
            if majority_ts and majority_ts[1] >= max(2, (total_models // 2) + 1):
                server_config["llama_server_defaults"]["tensor_split"] = str(majority_ts[0])
                server_changed = True

            # Promote TTL to top-level idle_ttl when consistent across models
            majority_ttl = _majority(ttl_counts)
            if majority_ttl and majority_ttl[1] >= max(2, (total_models // 2) + 1) and majority_ttl[0] != DEFAULT_IDLE_TTL:
                server_config["idle_ttl"] = int(majority_ttl[0])
                server_changed = True
    except Exception:
        # Non-fatal: inference best-effort only
        pass

    effective_idle_ttl = _effective_catalog_idle_ttl(layout, server_config)

    all_used_aliases: set[str] = set()
    for model in catalog_raw:
        if not isinstance(model, dict):
            continue

        existing_aliases = _normalize_catalog_aliases(model.get("aliases"))
        merged_aliases = []
        # First pass: collect existing aliases that are already assigned to this specific model
        for alias in existing_aliases:
            if alias not in all_used_aliases:
                merged_aliases.append(alias)
                all_used_aliases.add(alias)

        # Second pass: try to add new derived aliases only if they are not already taken globally
        quant = str(model.get("quant") or "").strip() or None
        repo_id = str(model.get("repo_id") or "")
        filename = str(model.get("filename") or "")
        
        for derived in _derived_aliases_for_import(repo_id, quant, filename):
            if derived not in merged_aliases and derived not in all_used_aliases:
                merged_aliases.append(derived)
                all_used_aliases.add(derived)

        if model.get("aliases") != merged_aliases:
            model["aliases"] = merged_aliases
            catalog_changed = True

        for ttl_key in ("ttl", "idle_ttl"):
            ttl_raw = model.get(ttl_key)
            if ttl_raw is None:
                continue
            try:
                if int(ttl_raw) == effective_idle_ttl:
                    model.pop(ttl_key, None)
                    catalog_changed = True
            except Exception:
                continue

        model_id = str(model.get("model_id") or "")
        if not model_id:
            continue

        try:
            catalog_ctx = int(model.get("ctx_size") or default_ctx)
        except Exception:
            catalog_ctx = default_ctx
        try:
            catalog_n_gpu_layers = int(model.get("n_gpu_layers") or default_n_gpu_layers)
        except Exception:
            catalog_n_gpu_layers = default_n_gpu_layers

        current_entry = server_config["models"].get(model_id)
        if not isinstance(current_entry, dict):
            server_config["models"][model_id] = {
                "ctx_size": catalog_ctx,
                "n_gpu_layers": catalog_n_gpu_layers,
            }
            server_changed = True
            continue

        entry = dict(current_entry)
        entry_changed = False
        try:
            current_ctx = int(entry.get("ctx_size"))
        except Exception:
            current_ctx = None
        try:
            current_n_gpu_layers = int(entry.get("n_gpu_layers"))
        except Exception:
            current_n_gpu_layers = None

        # Repair stale defaults left behind by previous installs/auto-ctx runs.
        if current_ctx is None or (current_ctx == default_ctx and catalog_ctx != default_ctx):
            entry["ctx_size"] = catalog_ctx
            entry_changed = True
        if current_n_gpu_layers is None or (
            current_n_gpu_layers == default_n_gpu_layers and catalog_n_gpu_layers != default_n_gpu_layers
        ):
            entry["n_gpu_layers"] = catalog_n_gpu_layers
            entry_changed = True

        if entry_changed:
            server_config["models"][model_id] = entry
            server_changed = True

    if server_changed:
        server_config_path.parent.mkdir(parents=True, exist_ok=True)
        server_config_path.write_text(json.dumps(server_config, indent=2) + "\n", encoding="utf-8")
        print(f"Verified Manager configuration in {server_config_path}")

    if catalog_changed:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(catalog_raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Updated catalog aliases/ttl defaults in {catalog_path}")

    # 2. Update config.yaml (llama-swap UI config) preserving your achieved features
    config_path = layout.state_dir / "config.yaml"
    if not config_path.exists():
        render_initial_config(config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            ui_config = yaml.safe_load(f) or {}
    except Exception:
        ui_config = {}

    if "models" not in ui_config:
        ui_config["models"] = {}

    ui_changed = False
    for model in catalog_raw:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("model_id") or "")
        if not model_id:
            continue

        # Get current entry or create a new one
        current = ui_config["models"].get(model_id, {})
        
        # Keep UI metadata in sync, but do not force keepAlive defaults here.
        # Forcing a short keepAlive can unload models between long agent cycles.
        needs_update = False
        entry = current.copy()

        # Heal legacy auto-injected keepAlive from previous installer versions.
        # If catalog does not define keepAlive, remove the forced 5m value.
        if entry.get("keepAlive") == "5m" and not model.get("keepAlive"):
            entry.pop("keepAlive", None)
            needs_update = True

        # Sync metadata from catalog if missing or different
        if model.get("aliases") and entry.get("aliases") != model["aliases"]:
            entry["aliases"] = model["aliases"]
            needs_update = True
            
        if model.get("description") and entry.get("description") != model["description"]:
            entry["description"] = model["description"]
            needs_update = True

        # CRITICAL: ensure checkEndpoint is present for llama-swap
        if "checkEndpoint" not in entry:
            entry["checkEndpoint"] = "/health"
            needs_update = True

        if needs_update:
            ui_config["models"][model_id] = entry
            ui_changed = True

    if ui_changed:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(ui_config, f, default_flow_style=False, sort_keys=False)
        print(f"Restored and verified metadata (aliases, descriptions) for {len(catalog_raw)} models in UI configuration.")


def maybe_rerun_auto_ctx(layout: InstallLayout, install_services: bool, dry_run: bool, args: argparse.Namespace) -> None:
    def _run_auto_ctx_update() -> None:
        try:
            _run([str(layout.bin_dir / CLI_COMMAND), "update", "--auto"])
            _ensure_basic_server_config(layout)
        except subprocess.CalledProcessError as exc:
            print(
                "Warning: auto-ctx update failed during install "
                f"(exit code {exc.returncode}). You can retry later with: "
                f"{layout.bin_dir / CLI_COMMAND} update --auto"
            )

    def _run_sync_config() -> None:
        try:
            _run([str(layout.bin_dir / CLI_COMMAND), "update"])
            _ensure_basic_server_config(layout)
            print(f"Synced registered models to llama-swap UI configuration.")
        except subprocess.CalledProcessError as exc:
            print(f"Warning: model sync failed (exit code {exc.returncode}).")

    catalog_path = layout.state_dir / "catalog.json"
    catalog_models = _catalog_model_count(catalog_path)
    if catalog_models <= 0:
        if _models_dir_has_gguf(layout.models_dir):
            imported_count = _auto_register_local_gguf_models(layout, dry_run)
            if imported_count > 0:
                catalog_models = _catalog_model_count(catalog_path)
    
    if catalog_models <= 0:
        return

    renamed_models = _maybe_migrate_catalog_model_ids(layout, dry_run, args)
    if renamed_models > 0:
        catalog_models = _catalog_model_count(catalog_path)

    # Ensure we have at least a basic config even if they skip auto-ctx
    if not dry_run:
        _ensure_basic_server_config(layout)

    # Check if we already have an answer from a previous execution (pre-sudo)
    rerun = getattr(args, "rerun_auto_ctx", None)
    if rerun is None:
        if not sys.stdin.isatty():
            rerun = False
        else:
            rerun = prompt_bool(f"Detected {catalog_models} registered models. Re-run auto-ctx now?", default=False)
            args.rerun_auto_ctx = rerun

    if not rerun:
        # If they skip auto-ctx, we still need to sync models to config.yaml 
        # so they show up in the UI.
        if not dry_run and install_services:
            if wait_for_manager_socket(layout, dry_run):
                _run_sync_config()
        return

    if dry_run:
        print(f"[dry-run] would run: {layout.bin_dir / CLI_COMMAND} update --auto")
        return
    if not install_services:
        print("Services were not enabled, so auto-ctx cannot be re-run automatically yet.")
        return
    if not wait_for_manager_socket(layout, dry_run):
        return
    _run_auto_ctx_update()


def maybe_migrate_existing_install(target_mode: str, public_host: str, public_port: int | None, dry_run: bool) -> None:
    existing_mode = detect_existing_mode()
    if not existing_mode or existing_mode == target_mode:
        return
    print(
        f"Existing {existing_mode}-mode installation detected. "
        f"It will be uninstalled before installing in {target_mode} mode."
    )
    print("Downloaded models will be kept.")
    if dry_run:
        print(f"[dry-run] would uninstall previous {existing_mode}-mode installation with --keep-models.")
        return

    try:
        # Preferred when running as a package
        from llamacpp_stack.uninstall import uninstall_stack
    except ImportError:
        # Fallback for direct script execution
        from uninstall import uninstall_stack

    uninstall_stack(
        argparse.Namespace(
            mode=existing_mode,
            public_host=public_host,
            public_port=public_port,
            keep_models=True,
            dry_run=False,
        )
    )


def install_stack(args: argparse.Namespace) -> int:
    pre_mode = resolve_install_mode(args.mode)
    chosen_public_host = resolve_public_host(args.public_host)
    suggested_models_dir = existing_models_dir(pre_mode) or derive_models_dir(detect_ollama_models_dir(), pre_mode)
    chosen_models_dir = Path(args.models_dir).expanduser() if args.models_dir else prompt_path("Models directory", suggested_models_dir)
    maybe_migrate_existing_install(pre_mode, chosen_public_host, args.public_port, args.dry_run)
    args.public_host = chosen_public_host

    # 1. First tentative layout to stop services if they exist
    layout = choose_layout(pre_mode, chosen_public_host, args.public_port, chosen_models_dir, args=args)
    if args.install_services:
        # Stop existing services early so ports they occupy can be re-assigned or migrated to.
        stop_systemd_units(layout, args.dry_run)

    # 2. Re-calculate layout (specifically ports) after services are stopped
    layout = choose_layout(pre_mode, chosen_public_host, args.public_port, chosen_models_dir, args=args)

    if args.update_binaries is not None:
        update_binaries = bool(args.update_binaries)
    else:
        update_binaries = True
        if is_existing_install(layout):
            update_binaries = prompt_bool("Existing installation detected. Update llama.cpp and llama-swap binaries?", default=True)
    # Preserve the interactive selection when re-executing in system mode via sudo.
    args.update_binaries = update_binaries
    stable_llama_server = layout.install_root / "llama-server"
    if not update_binaries and _is_self_referential_symlink(stable_llama_server):
        print(
            f"Detected broken self-referential symlink at {stable_llama_server}. "
            "Forcing binary update so llama-server can be recreated."
        )
        update_binaries = True
        args.update_binaries = True
    if update_binaries:
        llama_cpp_mode = resolve_llama_cpp_mode(args.llama_cpp_mode)
    else:
        llama_cpp_mode = detect_existing_llama_cpp_mode(layout)
        print(f"Keeping existing llama.cpp mode: {llama_cpp_mode}")
    reexec_status = maybe_reexec_system_install(args, pre_mode, llama_cpp_mode, chosen_models_dir)
    if reexec_status is not None:
        return reexec_status

    if args.dry_run:
        print(f"[dry-run] would ensure directories under {layout.install_root}, {layout.state_dir}, {layout.models_dir}")
    else:
        ensure_system_identity(layout, args.dry_run)
        ensure_dirs(layout)
    ensure_models_dir_ready(layout, args.dry_run)
    ensure_service_writable_dirs(layout, args.dry_run)

    if args.dry_run:
        llama_cpp_release = dry_run_release_placeholder(DEFAULT_LLAMA_CPP_REPO)
        llamaswap_release = dry_run_release_placeholder(DEFAULT_LLAMASWAP_REPO)
    else:
        llama_cpp_release = latest_release(DEFAULT_LLAMA_CPP_REPO)
        llamaswap_release = latest_release(DEFAULT_LLAMASWAP_REPO)
    llamaswap_asset = choose_llamaswap_asset(llamaswap_release)
    llama_cpp_asset = choose_llamacpp_linux_asset(llama_cpp_release)

    gpu_present = detect_nvidia_gpu()
    nvcc_path = locate_nvcc()
    cuda_toolkit_present = nvcc_path is not None
    if llama_cpp_mode == "source" and update_binaries and gpu_present and not cuda_toolkit_present:
        cuda_toolkit_present = maybe_install_cuda_toolkit(
            gpu_present=gpu_present,
            dry_run=args.dry_run,
            prefer_source_cuda=args.prefer_source_cuda,
            python_exec=sys.executable,
        )
        nvcc_path = locate_nvcc()
    if llama_cpp_mode == "source" and update_binaries and gpu_present:
        maybe_install_nccl_via_uv(sys.executable, args.dry_run)
    if llama_cpp_mode == "source" and update_binaries:
        maybe_install_source_build_prereqs(args.dry_run)

    current_manifest = read_install_manifest(layout)
    current_llama_cpp_tag = str(current_manifest.get("llama_cpp_tag") or "not installed")
    current_llamaswap_tag = str(current_manifest.get("llamaswap_tag") or "not installed")
    target_llama_cpp_tag = llama_cpp_release["tag_name"] if update_binaries else current_llama_cpp_tag
    target_llamaswap_tag = llamaswap_release["tag_name"] if update_binaries else current_llamaswap_tag
    print(f"llama.cpp target: {target_llama_cpp_tag}")
    print(f"llama-swap target: {target_llamaswap_tag}")
    print(f"llama.cpp current: {current_llama_cpp_tag}")
    print(f"llama-swap current: {current_llamaswap_tag}")
    print(f"installation mode: {layout.mode}")
    print(f"models directory: {layout.models_dir}")
    print(f"llama-swap UI/backend: http://{layout.public_host}:{layout.public_port}")
    print(f"Superserver API:     http://{layout.public_host}:{layout.public_port - 1}")
    print(f"llama.cpp mode: {llama_cpp_mode}")
    if llama_cpp_mode == "source" and update_binaries and gpu_present and not cuda_toolkit_present:
        print("NVIDIA GPU detected but no nvcc/CUDA toolkit was found; falling back to prebuilt llama.cpp binary.")

    if update_binaries:
        llamaswap_root = install_release_asset(llamaswap_asset, layout.install_root, args.dry_run)
    else:
        llamaswap_root = layout.install_root / f"{llamaswap_asset['name']}.d"
    if args.dry_run:
        llamaswap_bin = layout.install_root / "llama-swap"
    else:
        # Prefer to use an existing extracted asset if present; otherwise offer to
        # download/install it. Avoid raising an unhandled exception here so the
        # user can choose to recover interactively.
        llamaswap_real = None
        try:
            print(f"Checking for llama-swap under {llamaswap_root}...")
            llamaswap_real = _find_executable(llamaswap_root, "llama-swap")
            if not _is_executable_working(llamaswap_real):
                print(f"Warning: found llama-swap at {llamaswap_real} but it did not respond to a basic run check.")
                if update_binaries:
                    print("Re-installing llama-swap asset due to failing binary.")
                    llamaswap_root = install_release_asset(llamaswap_asset, layout.install_root, args.dry_run)
                    llamaswap_real = _find_executable(llamaswap_root, "llama-swap")
                else:
                    if prompt_bool("llama-swap appears broken. Download and install a fresh binary now?", default=True):
                        llamaswap_root = install_release_asset(llamaswap_asset, layout.install_root, args.dry_run)
                        llamaswap_real = _find_executable(llamaswap_root, "llama-swap")
                    else:
                        print("Continuing without updating llama-swap; some features may be unavailable.")
                        llamaswap_real = None
        except RuntimeError as exc:
            print(f"Warning: {exc}")
            if update_binaries:
                # Unexpected: we attempted to install but couldn't find the binary.
                print(f"llama-swap binary missing after extraction of {llamaswap_asset['name']}; aborting install.")
                raise
            if prompt_bool(f"llama-swap not found under {llamaswap_root}. Download and install it now?", default=True):
                llamaswap_root = install_release_asset(llamaswap_asset, layout.install_root, args.dry_run)
                llamaswap_real = _find_executable(llamaswap_root, "llama-swap")

        if llamaswap_real is not None:
            llamaswap_bin = _link_stable_binary(llamaswap_real, layout.install_root / "llama-swap", args.dry_run)
        else:
            llamaswap_bin = layout.install_root / "llama-swap"

    prefer_cuda_build = llama_cpp_mode == "source" and sys.platform.startswith("linux") and gpu_present and cuda_toolkit_present and args.prefer_source_cuda

    strategy = "binary"
    if llama_cpp_mode == "native":
        strategy = "native"
        native_llama_server = detect_native_llama_server()
        if update_binaries and native_llama_server is None:
            native_llama_server = maybe_install_native_llama_cpp(args.dry_run)
        if native_llama_server is None:
            if args.dry_run:
                print("[dry-run] no native llama-server binary or apt package detected on this machine; using /usr/bin/llama-server as placeholder.")
                native_llama_server = Path("/usr/bin/llama-server")
            else:
                raise RuntimeError(
                    "Native llama.cpp mode was selected, but no system llama-server binary was found and no native package could be installed. "
                    "Retry with llama.cpp mode 'prebuilt' or 'source', or install a native llama.cpp package first."
                )
        llama_server_bin = native_llama_server
    elif prefer_cuda_build:
        strategy = "source-build-cuda"
        if update_binaries:
            llama_server_real = build_llama_cpp_from_source(
                llama_cpp_release,
                layout.install_root,
                args.enable_tls,
                args.dry_run,
                sys.executable,
                enable_cuda=True,
            )
        else:
            llama_server_real = _resolve_existing_stable_target(
                layout.install_root,
                layout.install_root / "llama-server",
                "llama-server",
            ) or (layout.install_root / "llama-server")
        llama_server_bin = (
            layout.install_root / "llama-server"
            if args.dry_run
            else _link_stable_binary(llama_server_real, layout.install_root / "llama-server", args.dry_run)
        )
    elif llama_cpp_mode == "prebuilt" and llama_cpp_asset:
        cpp_root = install_release_asset(llama_cpp_asset, layout.install_root, args.dry_run) if update_binaries else layout.install_root / f"{llama_cpp_asset['name']}.d"
        if args.dry_run:
            llama_server_bin = layout.install_root / "llama-server"
        else:
            # Similar robust handling as for llama-swap: check existing asset, validate
            # binary, and offer to (re-)install if missing or not working.
            llama_server_real = None
            try:
                print(f"Checking for llama-server under {cpp_root}...")
                llama_server_real = _find_executable(cpp_root, "llama-server")
                if not _is_executable_working(llama_server_real):
                    print(f"Warning: found llama-server at {llama_server_real} but it did not respond to a basic run check.")
                    if update_binaries:
                        print("Re-installing llama.cpp prebuilt asset due to failing binary.")
                        cpp_root = install_release_asset(llama_cpp_asset, layout.install_root, args.dry_run)
                        llama_server_real = _find_executable(cpp_root, "llama-server")
                    else:
                        if prompt_bool("llama-server appears broken. Download and install a fresh binary now?", default=True):
                            cpp_root = install_release_asset(llama_cpp_asset, layout.install_root, args.dry_run)
                            llama_server_real = _find_executable(cpp_root, "llama-server")
                        else:
                            print("Continuing without updating llama-server; some features may be unavailable.")
                            llama_server_real = None
            except RuntimeError as exc:
                print(f"Warning: {exc}")
                if update_binaries:
                    print(f"llama-server binary missing after extraction of {llama_cpp_asset['name']}; aborting install.")
                    raise
                if prompt_bool(f"llama-server not found under {cpp_root}. Download and install it now?", default=True):
                    cpp_root = install_release_asset(llama_cpp_asset, layout.install_root, args.dry_run)
                    llama_server_real = _find_executable(cpp_root, "llama-server")

            if llama_server_real is not None:
                llama_server_bin = _link_stable_binary(llama_server_real, layout.install_root / "llama-server", args.dry_run)
            else:
                llama_server_bin = layout.install_root / "llama-server"
    else:
        strategy = "source-build"
        if update_binaries:
            llama_server_real = build_llama_cpp_from_source(
                llama_cpp_release,
                layout.install_root,
                args.enable_tls,
                args.dry_run,
                sys.executable,
                enable_cuda=cuda_toolkit_present and gpu_present,
            )
        else:
            llama_server_real = _resolve_existing_stable_target(
                layout.install_root,
                layout.install_root / "llama-server",
                "llama-server",
            ) or (layout.install_root / "llama-server")
        llama_server_bin = (
            layout.install_root / "llama-server"
            if args.dry_run
            else _link_stable_binary(llama_server_real, layout.install_root / "llama-server", args.dry_run)
        )

    runtime_python, runtime_python_path = ensure_runtime_python(layout, args.dry_run)
    installed_cuda_root = sync_cuda_runtime(layout, sys.executable, args.dry_run)
    env_text = _render_env(
        layout,
        llama_server_bin,
        llamaswap_bin,
        str(runtime_python),
        runtime_python_path,
        args.idle_ttl,
        installed_cuda_root,
        sync_nccl_runtime(layout, sys.executable, args.dry_run),
    )
    manager_unit = render_manager_service(layout)
    swap_unit = render_llamaswap_service(layout)
    if args.dry_run:
        print(env_text)
        print(manager_unit)
        print(swap_unit)
    else:
        (layout.config_dir / ENV_BASENAME).write_text(env_text, encoding="utf-8")
        legacy_env_path = layout.config_dir.parent / "llamacpp" / LEGACY_ENV_BASENAME
        if not legacy_env_path.exists():
            legacy_env_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_env_path.write_text(env_text, encoding="utf-8")
        server_config_path = layout.config_dir / SERVER_CONFIG_BASENAME
        server_config_data: dict[str, object] = {}
        if server_config_path.exists():
            try:
                payload = json.loads(server_config_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    server_config_data = payload
            except Exception:
                server_config_data = {}
        server_config_data["idle_ttl"] = args.idle_ttl
        server_config_data["api_port"] = layout.public_port - 1
        llama_defaults = server_config_data.setdefault("llama_server_defaults", {})
        if not isinstance(llama_defaults, dict):
            llama_defaults = {}
            server_config_data["llama_server_defaults"] = llama_defaults
        _ensure_llama_server_defaults_file(layout.config_dir)
        _merge_missing_llama_server_defaults(llama_defaults, layout.config_dir)
        _ensure_server_config_metadata(server_config_data)
        server_config_payload = json.dumps(server_config_data, indent=2) + "\n"
        server_config_path.write_text(
            server_config_payload,
            encoding="utf-8",
        )
        legacy_server_config_path = layout.config_dir / LEGACY_SERVER_CONFIG_BASENAME
        if not legacy_server_config_path.exists():
            legacy_server_config_path.write_text(server_config_payload, encoding="utf-8")
        config_path = layout.state_dir / "config.yaml"
        catalog_path = layout.state_dir / "catalog.json"
        if not config_path.exists():
            render_initial_config(config_path)
        if not catalog_path.exists():
            catalog_path.write_text("[]\n", encoding="utf-8")
        systemd_dir = layout.config_dir / "systemd"
        systemd_dir.mkdir(parents=True, exist_ok=True)
        (systemd_dir / MANAGER_SERVICE_NAME).write_text(manager_unit, encoding="utf-8")
        (systemd_dir / SWAP_SERVICE_NAME).write_text(swap_unit, encoding="utf-8")
        manager_wrapper = layout.bin_dir / MANAGER_WRAPPER_NAME
        manager_wrapper.write_text(render_manager_wrapper(layout), encoding="utf-8")
        manager_wrapper.chmod(0o755)
        swap_wrapper = layout.bin_dir / SWAP_WRAPPER_NAME
        swap_wrapper.write_text(render_llamaswap_wrapper(layout), encoding="utf-8")
        swap_wrapper.chmod(0o755)
        target_wrapper = layout.bin_dir / CLI_COMMAND
        target_wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "set -a\n"
            f"source {layout.config_dir / ENV_BASENAME}\n"
            "set +a\n"
            "export PYTHONPATH=\"$LLAMACPP_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}\"\n"
            "if [[ -n \"${LLAMACPP_CUDA_ROOT:-}\" ]]; then\n"
            "  export CUDA_PATH=\"$LLAMACPP_CUDA_ROOT\"\n"
            "  export LD_LIBRARY_PATH=\"$LLAMACPP_CUDA_ROOT/lib64:$LLAMACPP_CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n"
            "fi\n"
            "if [[ -n \"${LLAMACPP_NCCL_ROOT:-}\" ]]; then\n"
            "  export LD_LIBRARY_PATH=\"$LLAMACPP_NCCL_ROOT/lib64:$LLAMACPP_NCCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n"
            "fi\n"
            "exec \"$PYTHON_BIN\" -m llamacpp_stack.llamacpp_api_install \"$@\"\n",
            encoding="utf-8",
        )
        target_wrapper.chmod(0o755)
        legacy_wrapper = layout.bin_dir / LEGACY_CLI_COMMAND
        if not legacy_wrapper.exists() and not legacy_wrapper.is_symlink():
            legacy_wrapper.symlink_to(target_wrapper)
        ensure_service_writable_dirs(layout, args.dry_run)

    write_manifest(layout, llama_cpp_release, llamaswap_release, strategy, args.dry_run)
    if args.install_services:
        install_systemd_units(layout, args.dry_run)
        maybe_offer_ufw_ports(layout, args.dry_run)
    if args.install_services:
        restart_systemd_units(layout, args.dry_run)
    if not args.dry_run:
        maybe_rerun_auto_ctx(layout, args.install_services, args.dry_run, args)
    if not args.dry_run:
        print_install_summary(layout, args.install_services)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install llama.cpp + llama-swap stack.")
    parser.add_argument("--mode", choices=("system", "user"))
    parser.add_argument(
        "--llama-cpp-mode",
        choices=LLAMA_CPP_MODES,
        help="How to provide llama.cpp: native system package, prebuilt binary, or build from source.",
    )
    parser.add_argument("--public-host")
    parser.add_argument(
        "--public-port",
        type=int,
        help="llama-swap backend port. By default new installs use ollama_port+2 and reserve ollama_port+1 for the superserver API.",
    )
    parser.add_argument("--models-dir", help="Models directory. If omitted, the installer asks interactively.")
    parser.add_argument("--idle-ttl", type=int, default=DEFAULT_IDLE_TTL, help="Global idle timeout in seconds before llama-swap unloads a model.")
    parser.add_argument("--enable-tls", action="store_true", help="Try to enable extra HTTP/TLS related llama.cpp flags when supported.")
    parser.add_argument(
        "--prefer-source-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer building llama.cpp with CUDA on Linux/NVIDIA when nvcc is available.",
    )
    parser.add_argument(
        "--prefer-binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow official release binaries when a CUDA source build is not selected.",
    )
    parser.add_argument(
        "--install-services",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install and enable systemd services after rendering them.",
    )
    parser.add_argument(
        "--update-binaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force whether installer updates llama.cpp and llama-swap binaries on existing installs.",
    )
    parser.add_argument(
        "--migrate-model-ids",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="During install, optionally rename legacy model IDs to the cleaner API naming format.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Ensure parent dir is in path so we can import as a package even when re-executing as a script
    pkg_dir = Path(__file__).resolve().parent.parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))

    parser = build_parser()
    args = parser.parse_args(argv)
    return install_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
