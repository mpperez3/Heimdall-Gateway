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
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LLAMA_CPP_REPO = "ggml-org/llama.cpp"
DEFAULT_LLAMASWAP_REPO = "mostlygeek/llama-swap"
DEFAULT_IDLE_TTL = 300
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
SERVER_CONFIG_BASENAME = "llamacpp-superserver.json"
LEGACY_SERVER_CONFIG_BASENAME = "llamacpp-server.json"
ENV_BASENAME = "llamacpp-superserver.env"
LEGACY_ENV_BASENAME = "llamacpp-stack.env"
LLAMA_CPP_MODES = ("native", "prebuilt", "source")
ELEVATED_INSTALL_ENV = "LLAMACPP_INSTALL_ELEVATED"


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


def choose_default_swap_port(host: str, mode: str, explicit_public_port: int | None) -> int:
    if explicit_public_port:
        return explicit_public_port
    if existing := existing_public_port(mode):
        return existing
    ollama_port = detect_ollama_port()
    api_port = ollama_port + 1
    swap_port = ollama_port + 2
    if _port_is_free(host, api_port) and _port_is_free(host, swap_port):
        return swap_port
    return find_next_free_port(host, start=swap_port)


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
        Path("/var/lib/ollama/models"),
        Path("/usr/share/ollama/.ollama/models"),
        Path.home() / ".ollama" / "models",
    ):
        try:
            if candidate.exists():
                return candidate
        except PermissionError:
            continue
    return None


def existing_public_port(mode: str) -> int | None:
    env_path = env_path_for_mode(mode)
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("LLAMACPP_PUBLIC_PORT="):
            raw = line.split("=", 1)[1].strip()
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
            if line.startswith("LLAMACPP_MODELS="):
                return Path(line.split("=", 1)[1].strip()).expanduser()
    return None


def derive_models_dir(base: Path | None, mode: str) -> Path:
    if base:
        sibling = base.parent / "llamacpp_models"
        if mode == "system" or os.access(base.parent, os.W_OK):
            return sibling
    if mode == "system":
        return Path("/var/lib/llamacpp-superserver/models")
    return Path.home() / ".local/share/llamacpp-superserver/models"


def choose_layout(mode: str | None, public_host: str, public_port: int | None, models_dir: Path | None = None) -> InstallLayout:
    resolved_mode = mode or detect_existing_mode() or ("system" if os.geteuid() == 0 else "user")
    resolved_port = choose_default_swap_port(public_host, resolved_mode, public_port)
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
    raise RuntimeError(f"Executable {name} not found under {root}")


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
    return textwrap.dedent(
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
    payload = textwrap.dedent(
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


def sync_cuda_runtime(layout: InstallLayout, python_exec: str, dry_run: bool) -> Path | None:
    cuda_root = locate_cuda_root_for_python(python_exec)
    if cuda_root is None or not cuda_root.exists():
        return None
    if dry_run:
        print(f"[dry-run] would copy CUDA runtime from {cuda_root} to {layout.cuda_root}")
        return layout.cuda_root
    normalize_python_cuda_layout(cuda_root)
    if layout.cuda_root.exists():
        shutil.rmtree(layout.cuda_root)
    shutil.copytree(cuda_root, layout.cuda_root, symlinks=True)
    normalize_python_cuda_layout(layout.cuda_root)
    return layout.cuda_root


def sync_nccl_runtime(layout: InstallLayout, python_exec: str, dry_run: bool) -> Path | None:
    nccl_root = locate_nccl_root_for_python(python_exec)
    if nccl_root is None or not nccl_root.exists():
        return None
    if dry_run:
        print(f"[dry-run] would copy NCCL runtime from {nccl_root} to {layout.nccl_root}")
        return layout.nccl_root
    if layout.nccl_root.exists():
        shutil.rmtree(layout.nccl_root)
    shutil.copytree(nccl_root, layout.nccl_root, symlinks=True)
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
    cuda_root = Path(build_env["CUDAToolkit_ROOT"]) if build_env.get("CUDAToolkit_ROOT") else locate_cuda_root_for_python(python_exec)
    if enable_cuda and cuda_root:
        normalize_python_cuda_layout(cuda_root)
        _export_cuda_root(cuda_root)
        build_env = os.environ.copy()
        cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
    nccl_root = locate_nccl_root_for_python(python_exec)
    if enable_cuda and nccl_root:
        _export_nccl_root(nccl_root)
        build_env = os.environ.copy()
        include_dir = nccl_root / "include"
        library = None
        for lib_dir in (nccl_root / "lib64", nccl_root / "lib"):
            if lib_dir.exists():
                library = next(iter(lib_dir.glob("libnccl.so*")), None)
                if library:
                    break
        if include_dir.exists() and library:
            cmake_args.append(f"-DNCCL_INCLUDE_DIR={include_dir}")
            cmake_args.append(f"-DNCCL_LIBRARY={library}")
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
        reload_cmd = ["systemctl", "daemon-reload"]
        enable_cmd = ["systemctl", "enable", "--now", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME]
    else:
        systemd_dir = Path.home() / ".config/systemd/user"
        env_dir = layout.config_dir
        reload_cmd = ["systemctl", "--user", "daemon-reload"]
        enable_cmd = ["systemctl", "--user", "enable", "--now", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME]

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
    if layout.mode == "system":
        restart_cmd = ["systemctl", "restart", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME]
    else:
        restart_cmd = ["systemctl", "--user", "restart", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME]

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
    if layout.mode == "system":
        stop_cmd = ["systemctl", "stop", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME]
    else:
        stop_cmd = ["systemctl", "--user", "stop", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME]

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
        status_cmd = f"sudo systemctl status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
    else:
        start_cmd = f"systemctl --user start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
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
        print(f"Services enabled. Check with: {status_cmd}")
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
    slug = re.sub(r"[^a-z0-9._-]+", "-", (value or "").lower()).strip("-")
    return slug or "model"


def _infer_quant_from_filename(filename: str) -> str | None:
    upper = (filename or "").upper()
    match = re.search(
        r"(?<![A-Z0-9])(IQ\d(?:_[A-Z0-9]+)?|Q\d_K_[SML]|Q\d_[01]|Q\d_K|Q\d|BF16|F16|F32)(?![A-Z0-9])",
        upper,
    )
    return match.group(1) if match else None


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
        base_model_id = _slugify_model_id(gguf_path.stem)
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
                "quant": _infer_quant_from_filename(gguf_path.name),
                "filename": gguf_path.name,
                "local_path": str(gguf_path),
                "description": f"local / {gguf_path.name}",
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


def maybe_rerun_auto_ctx(layout: InstallLayout, install_services: bool, dry_run: bool) -> None:
    def _run_auto_ctx_update() -> None:
        try:
            _run([str(layout.bin_dir / CLI_COMMAND), "update", "--auto"])
        except subprocess.CalledProcessError as exc:
            print(
                "Warning: auto-ctx update failed during install "
                f"(exit code {exc.returncode}). You can retry later with: "
                f"{layout.bin_dir / CLI_COMMAND} update --auto"
            )

    catalog_path = layout.state_dir / "catalog.json"
    catalog_models = _catalog_model_count(catalog_path)
    if catalog_models <= 0:
        if _models_dir_has_gguf(layout.models_dir):
            imported_count = _auto_register_local_gguf_models(layout, dry_run)
            if imported_count > 0:
                catalog_models = _catalog_model_count(catalog_path)
    if catalog_models <= 0:
        if _models_dir_has_gguf(layout.models_dir):
            print(
                "Detected GGUF files in the models directory, but no registered catalog entries exist yet. "
                "Auto-ctx only applies to registered catalog models, so it was skipped."
            )
            print(
                "Register models first with: "
                f"{layout.bin_dir / CLI_COMMAND} run <hf-repo[:quant]> "
                "or "
                f"{layout.bin_dir / CLI_COMMAND} add <hf-repo[:quant]>"
            )
        return
    if not prompt_bool(f"Detected {catalog_models} registered models. Re-run auto-ctx now?", default=False):
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
    from .uninstall import uninstall_stack

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
    layout = choose_layout(pre_mode, chosen_public_host, args.public_port, chosen_models_dir)
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

    if args.install_services:
        stop_systemd_units(layout, args.dry_run)

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
        llamaswap_real = _find_executable(llamaswap_root, "llama-swap")
        llamaswap_bin = _link_stable_binary(llamaswap_real, layout.install_root / "llama-swap", args.dry_run)

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
            llama_server_real = _find_executable(cpp_root, "llama-server")
            llama_server_bin = _link_stable_binary(llama_server_real, layout.install_root / "llama-server", args.dry_run)
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
        server_config_data.setdefault("llama_server_defaults", {})
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
        maybe_rerun_auto_ctx(layout, args.install_services, args.dry_run)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return install_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
