"""Beellama.cpp installer - mirrors install.py model for llama.cpp.

Installs Anbeeld/beellama.cpp as alternative engine selectable per-model
via server_overrides.engine = 'beellama' (fallback to llama.cpp).

Build: cmake -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON (same as llama.cpp)
Install prefix: <install_root>/beellama (alongside llama.cpp)
Binary: /opt/beellama/llama-server-beellama (or user install_root)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Reuse helpers from install.py
from llamacpp_stack.install import (
    _build_cmake_args_from_config,
    _bundle_llamacpp_cmake_flags_path,
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

DEFAULT_BEELLAMA_REPO = "Anbeeld/beellama.cpp"
DEFAULT_BEELLAMA_REF = "main"

_DRIVER_MAX_CUDA = {
    525: 12.0, 535: 12.2, 545: 12.3, 550: 12.4, 555: 12.5, 560: 12.6,
}


def _detect_driver_max_cuda() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if not out:
            return 999.0
        major_minor = out[0].split(".")
        driver_major = int(major_minor[0])
        for drv_ver in sorted(_DRIVER_MAX_CUDA.keys(), reverse=True):
            if driver_major >= drv_ver:
                return _DRIVER_MAX_CUDA[drv_ver]
    except Exception:
        pass
    return 999.0


def _nvcc_cuda_version(nvcc_path: str) -> float:
    try:
        out = subprocess.run(
            [nvcc_path, "--version"], check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            if "release" in line.lower():
                parts = line.split("release")[-1].strip().split(",")[0].strip().split(".")
                return float(f"{parts[0]}.{parts[1]}")
    except Exception:
        pass
    return 999.0


def _pick_compatible_nvcc(system_nvcc, pip_nvcc, driver_max_cuda: float):
    candidates = []
    if system_nvcc:
        v = _nvcc_cuda_version(str(system_nvcc))
        candidates.append((system_nvcc, v, "system"))
    if pip_nvcc:
        v = _nvcc_cuda_version(str(pip_nvcc))
        candidates.append((pip_nvcc, v, "pip"))
    compatible = [(p, v, s) for p, v, s in candidates if v <= driver_max_cuda]
    if compatible:
        best = min(compatible, key=lambda x: x[1])
        print(f"[*] Using {best[2]} nvcc CUDA {best[1]} (driver supports ≤{driver_max_cuda})")
        return best[0]
    if candidates:
        print(f"[!] Warning: all nvcc versions ({', '.join(f'{s} CUDA {v}' for _, v, s in candidates)}) exceed driver max CUDA {driver_max_cuda}")
        print(f"[!] Binaries may fail to run on this driver. Upgrade driver or use CUDA ≤{driver_max_cuda}.")
        return candidates[0][0]
    return None


def _default_beellama_install_root(install_root: Path) -> Path:
    return install_root / "beellama"


def build_beellama(
    repo: str = DEFAULT_BEELLAMA_REPO,
    ref: str = DEFAULT_BEELLAMA_REF,
    install_root: Path | None = None,
    python_exec: str | None = None,
) -> Path:
    """Clone and build beellama.cpp, return binary path."""
    if install_root is None:
        # reuse same layout as llama.cpp: user vs system
        from llamacpp_stack.install import InstallLayout, detect_existing_mode

        # fallback to user install root
        install_root = Path.home() / ".local" / "opt" / "heimdall-gateway"
        if os.geteuid() == 0:
            install_root = Path("/opt") / "heimdall-gateway"

    beellama_root = _default_beellama_install_root(install_root)
    src_dir = beellama_root / "src"
    build_dir = beellama_root / "build"
    bin_path = beellama_root / "bin" / "llama-server-beellama"

    if not src_dir.exists():
        print(f"[*] Cloning {repo}@{ref} (shallow) -> {src_dir}")
        clone_args = ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(src_dir)]
        if ref != "main":
            clone_args.extend(["--branch", ref])
        subprocess.run(clone_args, check=True)
    else:
        print(f"[*] Updating {src_dir} to {ref}")
        subprocess.run(["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", ref], check=False)
        subprocess.run(["git", "-C", str(src_dir), "checkout", ref], check=False)

    # Prepare CUDA env like install.py does
    python_exec = python_exec or sys.executable
    # Detect driver CUDA compatibility and pick the right nvcc
    driver_max_cuda = _detect_driver_max_cuda()
    system_nvcc = locate_nvcc()
    pip_nvcc = locate_nvcc_for_python(python_exec)
    nvcc = _pick_compatible_nvcc(system_nvcc, pip_nvcc, driver_max_cuda)
    cuda_root = locate_cuda_root_for_python(python_exec)
    if nvcc and system_nvcc and str(Path(nvcc).resolve()) == str(Path(system_nvcc).resolve()):
        system_cuda = Path(nvcc).parent.parent
        if system_cuda.exists():
            cuda_root = system_cuda
    nccl_root = locate_nccl_root_for_python(python_exec)
    _export_nvcc_path(nvcc)
    _export_cuda_root(cuda_root)
    _export_nccl_root(nccl_root)

    enable_cuda = detect_nvidia_gpu()
    arch = None
    if enable_cuda:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()
            if out:
                arch = out[0].replace(".", "") + "-real"
            else:
                arch = "89-real"
        except Exception:
            arch = "89-real"

    build_dir.mkdir(parents=True, exist_ok=True)
    beellama_lib_dir = beellama_root / "lib"
    beellama_lib_dir.mkdir(parents=True, exist_ok=True)
    cmake_bin = Path.home() / ".local/opt/heimdall-gateway/venv/lib/python3.12/site-packages/cmake/data/bin/cmake"
    if not cmake_bin.exists():
        cmake_bin = Path("cmake")
    venv_cuda_bin = str(Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin")
    orig_path = os.environ.get("PATH", "")
    filtered_path = ":".join(p for p in orig_path.split(":") if "linuxbrew" not in p and "homebrew" not in p)
    if Path(venv_cuda_bin).exists():
        os.environ["PATH"] = f"{venv_cuda_bin}:/usr/lib/nvidia-cuda-toolkit/bin:/usr/bin:" + filtered_path
    else:
        os.environ["PATH"] = "/usr/lib/nvidia-cuda-toolkit/bin:/usr/bin:" + filtered_path
    # Fix CUDA 13.3 header mismatch (pip packages have 13.0 runtime vs 13.3 nvcc)
    existing = os.environ.get("CMAKE_CUDA_FLAGS", "")
    if "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK" not in existing:
        os.environ["CMAKE_CUDA_FLAGS"] = (existing + " -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK").strip()
    pip_cuda_lib = Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
    pip_nccl_lib = Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "nccl" / "lib"
    rpath_list = [str(beellama_lib_dir), str(build_dir / "bin")]
    if pip_cuda_lib.exists():
        rpath_list.append(str(pip_cuda_lib))
    if pip_nccl_lib.exists():
        rpath_list.append(str(pip_nccl_lib))
    # Also add cublas etc from nvidia packages if needed
    for extra in (Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia").glob("*/lib"):
        if str(extra) not in rpath_list and extra.exists():
            rpath_list.append(str(extra))
    cmake_args = _build_cmake_args_from_config(
        src_dir, build_dir, enable_cuda=enable_cuda, enable_tls=False, arch=arch,
        cuda_toolkit_root=cuda_root, nccl_root=nccl_root, nvcc_compiler=Path(nvcc) if nvcc else None,
        rpath_dirs=rpath_list,
    )
    cmake_args[0] = str(cmake_bin)
    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])
    if shutil.which("ccache"):
        cmake_args.extend(["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                           "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"])
    for _flag, _val in (("GGML_NATIVE", "ON"), ("GGML_CUDA_FA", "ON"), ("GGML_CUDA_F16", "ON")):
        try:
            from llamacpp_stack.install import source_tree_supports_flag
            if source_tree_supports_flag(src_dir, _flag):
                cmake_args.append(f"-D{_flag}={_val}")
            else:
                cmake_args.append(f"-D{_flag}={_val}")
        except Exception:
            cmake_args.append(f"-D{_flag}={_val}")
    print(f"[*] Configuring beellama: {' '.join(cmake_args)}")
    subprocess.run(cmake_args, check=True)
    build_jobs = max(1, os.cpu_count() or 4)
    make_bin = shutil.which("make") or "make"
    print(f"[*] Building beellama ({build_jobs} jobs, ~5-10min)...")
    subprocess.run([make_bin, "-C", str(build_dir), "-j", str(build_jobs)], check=True)
    print("[*] Installing beellama libs and binary with rpath (like llama.cpp)...")
    subprocess.run(["cmake", "--install", str(build_dir), "--prefix", str(beellama_root)], check=False)
    # Ensure binary and libs have rpath and are in place
    src_bin = build_dir / "bin" / "llama-server"
    if not src_bin.exists():
        src_bin = build_dir / "tools" / "server" / "llama-server"
    if src_bin.exists():
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_bin, bin_path)
        for lib in (build_dir / "bin").glob("libggml*.so*"):
            try:
                shutil.copy2(lib, beellama_lib_dir / lib.name)
            except Exception:
                pass
        try:
            env_path = Path.home() / ".config" / "heimdall-gateway" / "heimdall-gateway.env"
            if env_path.exists():
                text = env_path.read_text(encoding="utf-8")
                needed = f"{beellama_lib_dir}:{build_dir / 'bin'}"
                if needed not in text:
                    with env_path.open("a", encoding="utf-8") as f:
                        f.write(f"\nLD_LIBRARY_PATH={needed}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n")
                    print(f"[*] Updated {env_path} with beellama LD_LIBRARY_PATH")
        except Exception as e:
            print(f"[!] Could not update env: {e}")
        print(f"[*] Installed {bin_path} with rpath {beellama_lib_dir}")
        return bin_path
    raise FileNotFoundError(f"Built binary not found at {src_bin}")
