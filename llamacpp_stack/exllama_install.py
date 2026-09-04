"""ExLlamaV3 installer - mirrors beellama_install.py / install.py model.

Installs ExLlamaV3 (EXL3) as third engine selectable per-model
via server_overrides.engine = 'exllama' (fallback to llama.cpp).

Uses prebuilt wheels from turboderp-org/exllamav3 releases for fast,
reliable installation without CUDA compilation.

Layout (no admin, portable):
  <install_root>/exllama/
    src/exllamav3/          # git clone turboderp-org/exllamav3 (optional, for dev)
    venv/                   # isolated venv (uv or python -m venv)
    bin/llama-server-exllama  # wrapper (contains "llama-server" for cli.py:7609 detection)
    bin/exllama-server        # convenience symlink

Python path:
  Respects HEIMDALL_GATEWAY_PYTHONPATH. The wrapper exports
  PYTHONPATH="$HEIMDALL_GATEWAY_PYTHONPATH:$EXLLAMA_VENV_SITE:$PYTHONPATH"
  and LD_LIBRARY_PATH for CUDA/NCCL similar to beellama_install.py.

CUDA:
  --cuda auto|on|off  (default auto = detect_nvidia_gpu())
  --device 0,1 or auto
  Torch is installed with CUDA support matching the prebuilt exllamav3 wheel.

Usage:
  python -m llamacpp_stack.exllama_install --help
  python -m llamacpp_stack.exllama_install --dry-run
  python -m llamacpp_stack.exllama_install --install-root ~/.local/opt/heimdall-gateway --cuda auto
  HEIMDALL_GATEWAY_PYTHONPATH=/custom/python/path python -m llamacpp_stack.exllama_install --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# Reuse helpers from install.py (same as beellama_install.py)
try:
    from llamacpp_stack.install import (  # type: ignore
        detect_cuda_device_count,
        detect_nvidia_gpu,
        locate_nvcc,
        locate_nvcc_for_python,
        locate_cuda_root_for_python,
        locate_nccl_root_for_python,
        _export_nvcc_path,
        _export_cuda_root,
        _export_nccl_root,
    )
except Exception:  # fallback stubs when install.py unavailable in tests
    def detect_nvidia_gpu() -> bool:  # type: ignore
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
            return bool(r.stdout.strip())
        except Exception:
            return False

    def detect_cuda_device_count() -> int:  # type: ignore
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
            return len([l for l in r.stdout.splitlines() if l.strip()])
        except Exception:
            return 0

    def locate_nvcc() -> str | None:  # type: ignore
        return shutil.which("nvcc")

    def locate_nvcc_for_python(_: str) -> str | None:  # type: ignore
        return None

    def locate_cuda_root_for_python(_: str) -> Path | None:  # type: ignore
        return None

    def locate_nccl_root_for_python(_: str) -> Path | None:  # type: ignore
        return None

    def _export_nvcc_path(_: str | None) -> bool:  # type: ignore
        return False

    def _export_cuda_root(_: Path | None) -> bool:  # type: ignore
        return False

    def _export_nccl_root(_: Path | None) -> bool:  # type: ignore
        return False


DEFAULT_EXLLAMA_REPO = "turboderp-org/exllamav3"
DEFAULT_EXLLAMA_REF = "main"
DEFAULT_ENGINE_NAME = "exllama"
EXLLAMAV3_VERSION = "1.4.6"

# Prebuilt wheel URL template from GitHub releases
EXLLAMAV3_WHEEL_URL = (
    "https://github.com/turboderp-org/exllamav3/releases/download/v{version}/"
    "exllamav3-{version}+cu{cuda_tag}.torch{torch_ver}-cp{pyver}-cp{pyver}-linux_x86_64.whl"
)

# Torch CUDA version mapping: torch cu-tag -> CUDA version string for exllamav3 wheel
TORCH_CUDA_TAG_MAP = {
    "cu128": "128",
    "cu132": "132",
}


def _default_exllama_install_root(install_root: Path) -> Path:
    return install_root / "exllama"


def _resolve_install_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.geteuid() == 0:
        return Path("/opt/heimdall-gateway")
    env_root = os.environ.get("HEIMDALL_GATEWAY_ROOT") or os.environ.get("HEIMDALL_GATEWAY_INSTALL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.home() / ".local" / "opt" / "heimdall-gateway"


def _venv_python(exllama_root: Path) -> Path:
    return exllama_root / "venv" / "bin" / "python"


def _resolve_pythonpath_env() -> str:
    return os.environ.get("HEIMDALL_GATEWAY_PYTHONPATH", "").strip()


def _ensure_venv(exllama_root: Path, python_exec: str, dry_run: bool) -> Path:
    venv_dir = exllama_root / "venv"
    venv_python = venv_dir / "bin" / "python"
    if dry_run:
        print(f"[dry-run] would create venv at {venv_dir} (python {python_exec})")
        return venv_python
    if venv_python.exists():
        print(f"[*] Reusing existing venv: {venv_python}")
        return venv_python
    uv_bin = shutil.which("uv") or os.environ.get("HEIMDALL_GATEWAY_BOOTSTRAP_UV")
    if uv_bin and Path(str(uv_bin)).exists():
        print(f"[*] Creating venv with uv at {venv_dir}")
        subprocess.run([str(uv_bin), "venv", "--python", "3.12", "--seed", str(venv_dir)], check=True)
    else:
        print(f"[*] Creating venv with {python_exec} -m venv at {venv_dir}")
        subprocess.run([python_exec, "-m", "venv", str(venv_dir)], check=True)
        try:
            uv2 = shutil.which("uv") or os.environ.get("HEIMDALL_GATEWAY_BOOTSTRAP_UV")
            if uv2 and Path(str(uv2)).exists():
                subprocess.run([str(uv2), "pip", "install", "--python", str(venv_python), "--upgrade", "pip", "wheel", "setuptools"], check=False)
            else:
                subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], check=False)
        except Exception:
            pass
    return venv_python


def _uv_pip_install(venv_python: Path, packages: list[str], dry_run: bool, extra_args: list[str] | None = None) -> None:
    """Install packages using uv pip install (preferred) or fallback to pip."""
    if not packages:
        return
    uv_bin = shutil.which("uv") or os.environ.get("HEIMDALL_GATEWAY_BOOTSTRAP_UV")
    if uv_bin and Path(str(uv_bin)).exists():
        cmd = [str(uv_bin), "pip", "install", "--python", str(venv_python), "--upgrade"] + packages
    else:
        cmd = [str(venv_python), "-m", "pip", "install", "--upgrade"] + packages
    if extra_args:
        cmd.extend(extra_args)
    if dry_run:
        print(f"[dry-run] would run: {' '.join(cmd)}")
        return
    print(f"[*] uv pip install: {' '.join(packages)}")
    subprocess.run(cmd, check=True)


def _detect_torch_cuda_tag(venv_python: Path) -> tuple[str, str] | None:
    """Detect installed torch CUDA tag. Returns (torch_version, cuda_tag) like ('2.11.0', 'cu128')."""
    try:
        r = subprocess.run(
            [str(venv_python), "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None
        lines = r.stdout.strip().split("\n")
        torch_ver = lines[0]  # e.g. "2.11.0+cu128"
        cuda_ver = lines[1] if len(lines) > 1 else ""  # e.g. "12.8"
        # Extract base version and cuda tag
        m = re.match(r"(\d+\.\d+\.\d+)(?:\+(.+))?", torch_ver)
        if not m:
            return None
        base_ver = m.group(1)
        cuda_tag_suffix = m.group(2) or ""  # e.g. "cu128"
        return base_ver, cuda_tag_suffix
    except Exception:
        return None


def _resolve_exllamav3_wheel_url(venv_python: Path) -> str | None:
    """Find the best prebuilt exllamav3 wheel URL for the installed torch+CUDA."""
    info = _detect_torch_cuda_tag(venv_python)
    if not info:
        return None
    torch_ver, cuda_tag = info
    if cuda_tag not in TORCH_CUDA_TAG_MAP:
        print(f"[!] No prebuilt exllamav3 wheel for {cuda_tag}; try torch cu128 or cu132", file=sys.stderr)
        return None
    pyver = f"{sys.version_info.major}{sys.version_info.minor}"
    cuda_num = TORCH_CUDA_TAG_MAP[cuda_tag]
    url = EXLLAMAV3_WHEEL_URL.format(
        version=EXLLAMAV3_VERSION,
        cuda_tag=cuda_num,
        torch_ver=torch_ver,
        pyver=pyver,
    )
    return url


def _clone_or_update(repo: str, ref: str, dest: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would clone/update {repo}@{ref} -> {dest}")
        return
    if not dest.exists():
        print(f"[*] Cloning {repo}@{ref} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest)], check=True)
        if ref != "main":
            subprocess.run(["git", "-C", str(dest), "checkout", ref], check=True)
    else:
        print(f"[*] Updating {dest} to {ref}")
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", ref], check=False)
        subprocess.run(["git", "-C", str(dest), "checkout", ref], check=False)


def _write_server_module(exllama_root: Path, dry_run: bool) -> None:
    """Deploy exllamav3.server module into the venv's site-packages.

    Reads exllama_server.py from the heimdall-gateway package (MiaAI-Lab MIT)
    and installs it as exllamav3/server/__main__.py so
    ``python -m exllamav3.server`` works.
    """
    pkg_dir = Path(__file__).parent
    server_src = pkg_dir / "exllama_server.py"
    if not server_src.exists():
        print(f"[!] Server source not found: {server_src}", file=sys.stderr)
        return

    site_pkg = exllama_root / "venv" / "lib" / "python3.12" / "site-packages"
    if not site_pkg.exists():
        for cand in (exllama_root / "venv" / "lib").glob("python*/site-packages"):
            site_pkg = cand
            break
    server_dir = site_pkg / "exllamav3" / "server"
    if dry_run:
        print(f"[dry-run] would deploy {server_src} -> {server_dir}/__main__.py")
        return
    server_dir.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    _shutil.copy2(str(server_src), str(server_dir / "__main__.py"))
    (server_dir / "__init__.py").write_text("", encoding="utf-8")
    print(f"[*] Deployed exllamav3.server module from {server_src.name} -> {server_dir}")


def _write_wrapper(exllama_root: Path, venv_python: Path, install_root: Path, dry_run: bool) -> Path:
    """Create bin/llama-server-exllama wrapper (name contains 'llama-server' for cli.py:7609)."""
    bin_dir = exllama_root / "bin"
    wrapper = bin_dir / "llama-server-exllama"
    alias = bin_dir / "exllama-server"
    site_pkg = exllama_root / "venv" / "lib" / "python3.12" / "site-packages"
    if not site_pkg.exists():
        for cand in (exllama_root / "venv" / "lib").glob("python*/site-packages"):
            site_pkg = cand
            break
    # Find torch lib dir for LD_LIBRARY_PATH
    torch_lib = site_pkg / "torch" / "lib"
    pythonpath_env = _resolve_pythonpath_env()
    content = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        # ExLlamaV3 wrapper - EXL3 engine for Heimdall Gateway
        # Engine: {DEFAULT_ENGINE_NAME} (server_overrides.engine = 'exllama')
        EXLLAMA_ROOT="{exllama_root}"
        VENV_PYTHON="{venv_python}"
        INSTALL_ROOT="{install_root}"
        # Preserve caller's HEIMDALL_GATEWAY_PYTHONPATH and augment with venv site-packages
        if [[ -n "${{HEIMDALL_GATEWAY_PYTHONPATH:-}}" ]]; then
          export PYTHONPATH="${{HEIMDALL_GATEWAY_PYTHONPATH}}:{site_pkg}:${{PYTHONPATH:-}}"
        else
          export PYTHONPATH="{site_pkg}:${{PYTHONPATH:-}}"
        fi
        """)
    if pythonpath_env:
        content += f'export PYTHONPATH="{pythonpath_env}:$PYTHONPATH"\n'
    content += textwrap.dedent(f"""\
        # torch/lib needed for libc10.so, libc10_cuda.so etc.
        if [[ -d "{torch_lib}" ]]; then
          export LD_LIBRARY_PATH="{torch_lib}:${{LD_LIBRARY_PATH:-}}"
        fi
        # CUDA libs from nvidia pip packages
        for cuda_dir in "$EXLLAMA_ROOT/venv/lib/python3.12/site-packages/nvidia/cu128/lib" \\
                        "$EXLLAMA_ROOT/venv/lib/python3.12/site-packages/nvidia/cu132/lib" \\
                        "$EXLLAMA_ROOT/venv/lib/python3.12/site-packages/nvidia/cu13/lib"; do
          if [[ -d "$cuda_dir" ]]; then
            export LD_LIBRARY_PATH="$cuda_dir:${{LD_LIBRARY_PATH:-}}"
          fi
        done
        if [[ -d "$EXLLAMA_ROOT/venv/lib/python3.12/site-packages/nvidia/nccl/lib" ]]; then
          export LD_LIBRARY_PATH="$EXLLAMA_ROOT/venv/lib/python3.12/site-packages/nvidia/nccl/lib:${{LD_LIBRARY_PATH:-}}"
        fi
        if [[ -n "${{HEIMDALL_GATEWAY_CUDA_ROOT:-}}" ]]; then
          export LD_LIBRARY_PATH="$HEIMDALL_GATEWAY_CUDA_ROOT/lib64:$HEIMDALL_GATEWAY_CUDA_ROOT/lib:${{LD_LIBRARY_PATH:-}}"
        fi
        if [[ -n "${{HEIMDALL_GATEWAY_NCCL_ROOT:-}}" ]]; then
          export LD_LIBRARY_PATH="$HEIMDALL_GATEWAY_NCCL_ROOT/lib64:$HEIMDALL_GATEWAY_NCCL_ROOT/lib:${{LD_LIBRARY_PATH:-}}"
        fi
        if [[ ! -x "$VENV_PYTHON" ]]; then
          echo "[exllama] venv python not found: $VENV_PYTHON" >&2
          echo "[exllama] Run: python -m llamacpp_stack.exllama_install --install-root $INSTALL_ROOT" >&2
          exit 1
        fi
        exec "$VENV_PYTHON" -m exllamav3.server "$@"
        """)
    if dry_run:
        print(f"[dry-run] would write wrapper {wrapper}")
        return wrapper
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(content, encoding="utf-8")
    wrapper.chmod(0o755)
    try:
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(wrapper.name)
    except Exception:
        pass
    print(f"[*] Wrote wrapper {wrapper}")
    return wrapper


def build_exllama(
    repo: str = DEFAULT_EXLLAMA_REPO,
    ref: str = DEFAULT_EXLLAMA_REF,
    install_root: Path | None = None,
    python_exec: str | None = None,
    enable_cuda: bool | None = None,
    cuda_device: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Install ExLlamaV3 via prebuilt wheel + isolated venv, return wrapper path.

    No admin / no sudo required. Uses venv under <install_root>/exllama.
    Installs prebuilt CUDA extension wheel (fast, no compilation).
    """
    if install_root is None:
        install_root = _resolve_install_root(None)
    exllama_root = _default_exllama_install_root(install_root)
    venv_python_target = exllama_root / "venv" / "bin" / "python"

    python_exec = python_exec or sys.executable

    # CUDA detection
    if enable_cuda is None:
        enable_cuda = detect_nvidia_gpu()
    nvcc = locate_nvcc_for_python(python_exec) or locate_nvcc()
    cuda_root = locate_cuda_root_for_python(python_exec)
    nccl_root = locate_nccl_root_for_python(python_exec)
    _export_nvcc_path(nvcc)
    _export_cuda_root(cuda_root)
    _export_nccl_root(nccl_root)
    if cuda_device and cuda_device != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        print(f"[*] CUDA_VISIBLE_DEVICES={cuda_device}")

    # Venv
    venv_python = _ensure_venv(exllama_root, python_exec, dry_run=dry_run)

    # Install torch with CUDA support
    if enable_cuda:
        # Detect GPU compute capability to select best torch CUDA tag
        torch_cuda_tag = "cu128"  # default: CUDA 12.8
        if cuda_root:
            # Try to detect actual CUDA version from nvcc
            nvcc_path = nvcc or str(cuda_root / "bin" / "nvcc") if cuda_root else None
            if nvcc_path and Path(nvcc_path).exists():
                try:
                    r = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True, timeout=5)
                    m = re.search(r"release (\d+)\.(\d+)", r.stdout)
                    if m:
                        major, minor = int(m.group(1)), int(m.group(2))
                        if major >= 13 and minor >= 2:
                            torch_cuda_tag = "cu132"
                        elif major >= 12:
                            torch_cuda_tag = "cu128"
                except Exception:
                    pass
        _uv_pip_install(venv_python, [f"torch==2.11.0+{torch_cuda_tag}"], dry_run=dry_run,
                        extra_args=["--index-url", f"https://download.pytorch.org/whl/{torch_cuda_tag}"])
    else:
        _uv_pip_install(venv_python, ["torch==2.11.0"], dry_run=dry_run)

    # Install exllamav3 prebuilt wheel (fast, no compilation)
    if enable_cuda:
        wheel_url = _resolve_exllamav3_wheel_url(venv_python)
        if wheel_url:
            _uv_pip_install(venv_python, [wheel_url], dry_run=dry_run, extra_args=["--no-deps"])
        else:
            print("[!] Could not resolve prebuilt wheel URL; falling back to NOCOMPILE install", file=sys.stderr)
            # Fallback: install from PyPI without compilation
            _uv_pip_install(venv_python, ["exllamav3"], dry_run=dry_run,
                            extra_args=["--no-deps", "--no-build-isolation"])
    else:
        _uv_pip_install(venv_python, ["exllamav3"], dry_run=dry_run, extra_args=["--no-deps"])

    # Install additional deps
    _uv_pip_install(venv_python, ["safetensors", "numpy", "aiohttp"], dry_run=dry_run)

    # Create exllamav3.server module (OpenAI-compatible HTTP server)
    _write_server_module(exllama_root, dry_run=dry_run)

    # Wrapper
    wrapper = _write_wrapper(exllama_root, venv_python_target, install_root, dry_run=dry_run)

    # EXL3 marker
    if not dry_run:
        try:
            (exllama_root / ".exl3-ready").write_text(
                f"repo={repo} ref={ref} cuda={enable_cuda} version={EXLLAMAV3_VERSION}\n", encoding="utf-8"
            )
        except Exception:
            pass

    return wrapper


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m llamacpp_stack.exllama_install",
        description="Install ExLlamaV3 (EXL3) as third engine (exllama) alongside llama.cpp and vLLM. No admin required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m llamacpp_stack.exllama_install --help
              python -m llamacpp_stack.exllama_install --dry-run
              python -m llamacpp_stack.exllama_install --dry-run --cuda auto --device auto
              python -m llamacpp_stack.exllama_install --install-root ~/.local/opt/heimdall-gateway --cuda on

            Engine selection:
              Set per-model server_overrides.engine = 'exllama' in catalog.json to route that model
              through bin/llama-server-exllama. Wrapper name contains 'llama-server' so cli.py:7609
              auto-detects it as a server process. Fallback is llama.cpp.

            EXL3:
              ExLlamaV3 is the inference library for EXL3 quantized models.
              Installed from prebuilt wheels (turboderp-org/exllamav3) for fast setup.
              No CUDA compilation required at install time.

            Portable / no-admin:
              Uses venv under <install_root>/exllama/venv and exports HEIMDALL_GATEWAY_PYTHONPATH
              in the wrapper. No sudo, no system packages.
        """),
    )
    p.add_argument("--install-root", default=None, help="Install root (default: $HEIMDALL_GATEWAY_ROOT or ~/.local/opt/heimdall-gateway)")
    p.add_argument("--python", dest="python_exec", default=None, help="Python executable for venv creation (default: sys.executable)")
    p.add_argument("--repo", default=DEFAULT_EXLLAMA_REPO, help=f"GitHub repo (default: {DEFAULT_EXLLAMA_REPO})")
    p.add_argument("--ref", default=DEFAULT_EXLLAMA_REF, help=f"Git ref/branch/tag (default: {DEFAULT_EXLLAMA_REF})")
    p.add_argument("--engine", default=DEFAULT_ENGINE_NAME, choices=["exllama", "llama.cpp", "beellama"], help="Engine name (default: exllama)")
    p.add_argument("--cuda", dest="cuda", default="auto", choices=["auto", "on", "off", "true", "false", "1", "0"], help="CUDA mode (default: auto)")
    p.add_argument("--device", default="auto", help="CUDA device selector: auto or e.g. '0' or '0,1'")
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without changing disk")
    p.add_argument("--check", action="store_true", help="Check exllama installation status and exit")
    return p


def _check_status(install_root: Path) -> int:
    exllama_root = _default_exllama_install_root(install_root)
    venv_python = _venv_python(exllama_root)
    wrapper = exllama_root / "bin" / "llama-server-exllama"
    print(f"Install root: {install_root}")
    print(f"ExLlama root: {exllama_root}")
    print(f"Venv python: {venv_python}  exists={venv_python.exists()}")
    print(f"Wrapper: {wrapper}  exists={wrapper.exists()}")
    print(f"Wrapper alias: {exllama_root / 'bin' / 'exllama-server'}  exists={(exllama_root / 'bin' / 'exllama-server').exists()}")
    print(f"HEIMDALL_GATEWAY_PYTHONPATH={_resolve_pythonpath_env()!r}")
    print(f"CUDA GPU detected: {detect_nvidia_gpu()}  device_count={detect_cuda_device_count()}")
    print(f"Engine: {DEFAULT_ENGINE_NAME}  (use server_overrides.engine='exllama' per model)")
    print(f"ExLlamaV3: {DEFAULT_EXLLAMA_REPO} v{EXLLAMAV3_VERSION}")
    if venv_python.exists():
        try:
            r = subprocess.run([str(venv_python), "-c", "import sys; print(sys.version)"], capture_output=True, text=True, timeout=5)
            print(f"venv python version: {r.stdout.strip()}")
        except Exception as e:
            print(f"venv check failed: {e}")
        for mod in ["exllamav3", "torch", "safetensors"]:
            try:
                r = subprocess.run([str(venv_python), "-c", f"import {mod}; v = getattr({mod}, '__version__', None) or getattr(__import__('{mod}.version', fromlist=['__version__']), '__version__', 'ok'); print(v)"], capture_output=True, text=True, timeout=5)
                status = r.stdout.strip() or r.stderr.strip()[:200]
                print(f"  {mod}: {status or 'not installed'}")
            except Exception:
                print(f"  {mod}: not installed")
        # Check CUDA ext
        try:
            torch_lib = venv_python.parent.parent / "lib" / "python3.12" / "site-packages" / "torch" / "lib"
            env = os.environ.copy()
            if torch_lib.exists():
                env["LD_LIBRARY_PATH"] = str(torch_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
            r = subprocess.run([str(venv_python), "-c", "from exllamav3.ext import exllamav3_ext; print('CUDA ext: OK')"],
                               capture_output=True, text=True, timeout=10, env=env)
            status = r.stdout.strip() or r.stderr.strip()[:200]
            print(f"  exllamav3 CUDA ext: {status}")
        except Exception:
            print(f"  exllamav3 CUDA ext: not loaded")
    ready = exllama_root / ".exl3-ready"
    if ready.exists():
        try:
            print(f"EXL3 ready marker: {ready.read_text().strip()}")
        except Exception:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    install_root = _resolve_install_root(args.install_root)
    if args.check:
        return _check_status(install_root)

    cuda_mode = str(args.cuda).lower()
    if cuda_mode in ("auto",):
        enable_cuda = None
    elif cuda_mode in ("on", "true", "1"):
        enable_cuda = True
    else:
        enable_cuda = False

    device = None if args.device == "auto" else args.device

    if args.engine != DEFAULT_ENGINE_NAME:
        print(f"[*] Engine set to {args.engine!r}; exllama wrapper still installed as exllama engine.", file=sys.stderr)

    try:
        wrapper = build_exllama(
            repo=args.repo,
            ref=args.ref,
            install_root=install_root,
            python_exec=args.python_exec,
            enable_cuda=enable_cuda,
            cuda_device=device,
            dry_run=bool(args.dry_run),
        )
        if args.dry_run:
            print(f"[dry-run] exllama install dry-run complete. Wrapper would be at: {wrapper}")
            print(f"[dry-run] HEIMDALL_GATEWAY_PYTHONPATH={_resolve_pythonpath_env()!r} (respected)")
            print(f"[dry-run] ExLlamaV3 v{EXLLAMAV3_VERSION}, CUDA={'auto' if enable_cuda is None else enable_cuda}, engine={args.engine}")
        else:
            print(f"[*] exllama install complete: {wrapper}")
            print(f"[*] Engine '{DEFAULT_ENGINE_NAME}' ready. Set server_overrides.engine='exllama' per model to use it.")
            print(f"[*] HEIMDALL_GATEWAY_PYTHONPATH={_resolve_pythonpath_env()!r} (wrapper exports it)")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[!] Command failed: {e}", file=sys.stderr)
        return e.returncode or 1
    except Exception as e:
        print(f"[!] exllama install failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
