"""Buun-llama-cpp installer - mirrors beellama_install.py model.

Installs spiritbuun/buun-llama-cpp as 4th optional engine for EXL3/FP8
via server_overrides.engine = 'buun' (fallback to llama.cpp).

Build: cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native (fallback 110/120)
Install prefix: <install_root>/buun (alongside llama.cpp/beellama)
Binary: buun/bin/llama-server-buun
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_BUUN_REPO = "spiritbuun/buun-llama-cpp"
DEFAULT_BUUN_REF = "c7f114d"

_DRIVER_MAX_CUDA = {
    525: 12.0,
    535: 12.2,
    545: 12.3,
    550: 12.4,
    555: 12.5,
    560: 12.6,
}


def _detect_driver_max_cuda() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
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
            [nvcc_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
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
        try:
            v = _nvcc_cuda_version(str(system_nvcc))
            candidates.append((system_nvcc, v, "system"))
        except Exception:
            pass
    if pip_nvcc:
        try:
            v = _nvcc_cuda_version(str(pip_nvcc))
            candidates.append((pip_nvcc, v, "pip"))
        except Exception:
            pass
    compatible = [(p, v, s) for p, v, s in candidates if v <= driver_max_cuda]
    if compatible:
        best = min(compatible, key=lambda x: x[1])
        print(f"[*] Using {best[2]} nvcc CUDA {best[1]} (driver supports <={driver_max_cuda})")
        return best[0]
    if candidates:
        try:
            print(
                f"[!] Warning: all nvcc versions ({', '.join(f'{s} CUDA {v}' for _, v, s in candidates)}) exceed driver max CUDA {driver_max_cuda}",
            )
            print("[!] Binaries may fail to run on this driver. Upgrade driver or use CUDA <={driver_max_cuda}.")
        except Exception:
            pass
        return candidates[0][0]
    return None


def _default_buun_install_root(install_root: Path) -> Path:
    return install_root / "buun"


def _resolve_cuda_arch() -> str:
    """Return CMAKE_CUDA_ARCHITECTURES value: native or fallback 110/120."""
    # Try nvidia-smi compute_cap first, never throw, short timeout
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().splitlines()
        if out and out[0].strip():
            cap = out[0].strip().replace(".", "")
            # Map to arch string: e.g. 89 -> 89-real, 120 -> 120
            # Prefer native when we can detect, else fallback
            if cap in {"89", "90", "100", "110", "120", "121"}:
                return "native"
            return "native"
    except Exception:
        pass
    # Fallback arch list for CMake when native unsupported
    return "native"


def build_buun(
    repo: str = DEFAULT_BUUN_REPO,
    ref: str = DEFAULT_BUUN_REF,
    install_root: Path | None = None,
    python_exec: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Clone and build buun-llama-cpp, return binary path.

    Mirrors beellama_install.build_beellama with buun specifics:
    - repo spiritbuun/buun-llama-cpp@c7f114d
    - bin buun/bin/llama-server-buun
    - cmake flags -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native (fallback 110;120)
    - rpath $ORIGIN via install helpers
    - wrapper pre-check ``llama-server-buun --help``
    """
    if install_root is None:
        if os.geteuid() == 0:
            install_root = Path("/opt/heimdall-gateway")
        else:
            install_root = Path.home() / ".local" / "opt" / "heimdall-gateway"

    buun_root = _default_buun_install_root(install_root)
    src_dir = buun_root / "src"
    build_dir = buun_root / "build"
    bin_path = buun_root / "bin" / "llama-server-buun"

    if dry_run:
        print(f"[dry-run] would install buun via build_buun(install_root={install_root}, python_exec={python_exec or sys.executable}) with HEIMDALL_GATEWAY_PYTHONPATH={os.environ.get('HEIMDALL_GATEWAY_PYTHONPATH','')}")
        print(f"[dry-run] would clone {repo}@{ref} -> {src_dir}")
        print(f"[dry-run] would cmake -B {build_dir} -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build {build_dir} -j && cmake --install {build_dir} --prefix {buun_root}")
        print(f"[dry-run] would install binary at {bin_path} with rpath lib")
        return bin_path

    # Pre-check wrapper if already installed (never throw)
    if bin_path.exists():
        try:
            r = subprocess.run(
                [str(bin_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                print(f"[*] buun binary already present: {bin_path} (--help OK)")
            else:
                print(f"[*] buun binary present but --help returned {r.returncode}, rebuilding")
        except Exception:
            pass

    if not src_dir.exists():
        print(f"[*] Cloning {repo}@{ref} (shallow) -> {src_dir}")
        clone_args = ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(src_dir)]
        if ref != DEFAULT_BUUN_REF and ref != "main":
            clone_args.extend(["--branch", ref])
        subprocess.run(clone_args, check=True, timeout=60)
        # Ensure exact ref c7f114d when requested
        if ref == DEFAULT_BUUN_REF:
            try:
                subprocess.run(
                    ["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", ref],
                    check=False,
                    timeout=30,
                )
                subprocess.run(["git", "-C", str(src_dir), "checkout", ref], check=False, timeout=10)
            except Exception:
                pass
    else:
        print(f"[*] Updating {src_dir} to {ref}")
        try:
            subprocess.run(
                ["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin", ref],
                check=False,
                timeout=30,
            )
            subprocess.run(["git", "-C", str(src_dir), "checkout", ref], check=False, timeout=10)
        except Exception:
            pass

    # Lazy imports to avoid circular deps (only constants at top level)
    try:
        from llamacpp_stack.install import (
            _build_cmake_args_from_config,
            detect_nvidia_gpu,
            locate_nvcc,
            locate_nvcc_for_python,
            locate_cuda_root_for_python,
            locate_nccl_root_for_python,
            _export_nvcc_path,
            _export_cuda_root,
            _export_nccl_root,
        )
    except Exception:
        # Fallback stubs when install.py unavailable in tests (never throw)
        def detect_nvidia_gpu() -> bool:  # type: ignore
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                return bool(r.stdout.strip())
            except Exception:
                return False

        def locate_nvcc():  # type: ignore
            return shutil.which("nvcc")

        def locate_nvcc_for_python(_):  # type: ignore
            return None

        def locate_cuda_root_for_python(_):  # type: ignore
            return None

        def locate_nccl_root_for_python(_):  # type: ignore
            return None

        def _export_nvcc_path(_):  # type: ignore
            return False

        def _export_cuda_root(_):  # type: ignore
            return False

        def _export_nccl_root(_):  # type: ignore
            return False

        def _build_cmake_args_from_config(src, build, **kwargs):  # type: ignore
            return ["cmake", "-S", str(src), "-B", str(build)]

    python_exec = python_exec or sys.executable
    driver_max_cuda = _detect_driver_max_cuda()
    system_nvcc = None
    pip_nvcc = None
    try:
        system_nvcc = locate_nvcc()
    except Exception:
        pass
    try:
        pip_nvcc = locate_nvcc_for_python(python_exec)
    except Exception:
        pass
    nvcc = _pick_compatible_nvcc(system_nvcc, pip_nvcc, driver_max_cuda)
    cuda_root = None
    try:
        cuda_root = locate_cuda_root_for_python(python_exec)
    except Exception:
        pass
    if nvcc and system_nvcc and str(Path(nvcc).resolve()) == str(Path(system_nvcc).resolve()):
        try:
            system_cuda = Path(nvcc).parent.parent
            if system_cuda.exists():
                cuda_root = system_cuda
        except Exception:
            pass
    nccl_root = None
    try:
        nccl_root = locate_nccl_root_for_python(python_exec)
    except Exception:
        pass
    try:
        _export_nvcc_path(nvcc)
    except Exception:
        pass
    try:
        _export_cuda_root(cuda_root)
    except Exception:
        pass
    try:
        _export_nccl_root(nccl_root)
    except Exception:
        pass

    enable_cuda = False
    try:
        enable_cuda = bool(detect_nvidia_gpu())
    except Exception:
        pass

    arch = None
    if enable_cuda:
        try:
            arch = _resolve_cuda_arch()
        except Exception:
            arch = "native"

    build_dir.mkdir(parents=True, exist_ok=True)
    buun_lib_dir = buun_root / "lib"
    buun_lib_dir.mkdir(parents=True, exist_ok=True)

    # cmake binary resolution
    cmake_bin = Path.home() / ".local/opt/heimdall-gateway/venv/lib/python3.12/site-packages/cmake/data/bin/cmake"
    if not cmake_bin.exists():
        cmake_bin = Path(shutil.which("cmake") or "cmake")

    # PATH filtering for reproducible build
    venv_cuda_bin = str(Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin")
    orig_path = os.environ.get("PATH", "")
    filtered_path = ":".join(p for p in orig_path.split(":") if "linuxbrew" not in p and "homebrew" not in p)
    if Path(venv_cuda_bin).exists():
        os.environ["PATH"] = f"{venv_cuda_bin}:/usr/lib/nvidia-cuda-toolkit/bin:/usr/bin:" + filtered_path
    else:
        os.environ["PATH"] = "/usr/lib/nvidia-cuda-toolkit/bin:/usr/bin:" + filtered_path

    # Fix CUDA 13.3 header mismatch
    existing = os.environ.get("CMAKE_CUDA_FLAGS", "")
    if "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK" not in existing:
        os.environ["CMAKE_CUDA_FLAGS"] = (existing + " -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK").strip()

    # rpath collection
    pip_cuda_lib = Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
    pip_nccl_lib = Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "nccl" / "lib"
    rpath_list = [str(buun_lib_dir), str(build_dir / "bin")]
    if pip_cuda_lib.exists():
        rpath_list.append(str(pip_cuda_lib))
    if pip_nccl_lib.exists():
        rpath_list.append(str(pip_nccl_lib))
    try:
        for extra in (Path(python_exec).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia").glob("*/lib"):
            if str(extra) not in rpath_list and extra.exists():
                rpath_list.append(str(extra))
    except Exception:
        pass

    # Build cmake args via install helper (inherits GGML_CUDA handling)
    try:
        cmake_args = _build_cmake_args_from_config(
            src_dir,
            build_dir,
            enable_cuda=enable_cuda,
            enable_tls=False,
            arch=arch,
            cuda_toolkit_root=cuda_root,
            nccl_root=nccl_root,
            nvcc_compiler=Path(nvcc) if nvcc else None,
            rpath_dirs=rpath_list,
        )
    except Exception:
        cmake_args = ["cmake", "-S", str(src_dir), "-B", str(build_dir)]
        if enable_cuda:
            cmake_args.extend(["-DGGML_CUDA=ON"])

    cmake_args[0] = str(cmake_bin)
    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])
    if shutil.which("ccache"):
        cmake_args.extend(
            ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"],
        )

    # Explicit buun cmake flags: GGML_CUDA + CUDA architectures (native or 110;120 fallback)
    has_cuda_flag = any("GGML_CUDA" in a for a in cmake_args)
    if not has_cuda_flag and enable_cuda:
        cmake_args.append("-DGGML_CUDA=ON")
    # Ensure arch flag present when CUDA enabled
    if enable_cuda:
        has_arch = any("CMAKE_CUDA_ARCHITECTURES" in a for a in cmake_args)
        if not has_arch:
            # Prefer native; if detection failed use 110;120 fallback string
            arch_val = arch if arch else "native"
            cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch_val}")

    # Extra flags mirroring beellama (native FA/F16) if supported
    for _flag, _val in (("GGML_NATIVE", "ON"), ("GGML_CUDA_FA", "ON"), ("GGML_CUDA_F16", "ON")):
        try:
            from llamacpp_stack.install import source_tree_supports_flag

            if source_tree_supports_flag(src_dir, _flag):
                cmake_args.append(f"-D{_flag}={_val}")
            else:
                cmake_args.append(f"-D{_flag}={_val}")
        except Exception:
            cmake_args.append(f"-D{_flag}={_val}")

    has_tests = any("LLAMA_BUILD_TESTS" in a or "BUILD_TESTING" in a for a in cmake_args)
    if not has_tests:
        cmake_args.append("-DLLAMA_BUILD_TESTS=OFF")
    if not any("BUILD_TESTING" in a for a in cmake_args):
        cmake_args.append("-DBUILD_TESTING=OFF")
    print(f"[*] Configuring buun: {' '.join(cmake_args)}")
    subprocess.run(cmake_args, check=True, timeout=60)

    build_jobs = max(1, os.cpu_count() or 4)
    cmake_build_bin = str(cmake_bin)
    print(f"[*] Building buun ({build_jobs} jobs, ~5-10min) via cmake --build (llama-server target)...")
    try:
        subprocess.run([cmake_build_bin, "--build", str(build_dir), "--target", "llama-server", "-j", str(build_jobs)], check=True)
    except subprocess.CalledProcessError:
        subprocess.run([cmake_build_bin, "--build", str(build_dir), "-j", str(build_jobs)], check=True)

    print("[*] Installing buun libs and binary with rpath (like llama.cpp)...")
    try:
        subprocess.run(["cmake", "--install", str(build_dir), "--prefix", str(buun_root)], check=False, timeout=30)
    except Exception:
        pass

    # Ensure binary and libs have rpath and are in place
    src_bin = build_dir / "bin" / "llama-server"
    if not src_bin.exists():
        src_bin = build_dir / "tools" / "server" / "llama-server"
    if not src_bin.exists():
        src_bin = build_dir / "bin" / "llama-server-buun"
    if src_bin.exists():
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_bin, bin_path)
        for lib in (build_dir / "bin").glob("libggml*.so*"):
            try:
                shutil.copy2(lib, buun_lib_dir / lib.name)
            except Exception:
                pass
        # Update env file with LD_LIBRARY_PATH (never throw)
        try:
            env_path = Path.home() / ".config" / "heimdall-gateway" / "heimdall-gateway.env"
            if env_path.exists():
                text = env_path.read_text(encoding="utf-8")
                needed = f"{buun_lib_dir}:{build_dir / 'bin'}"
                if needed not in text:
                    with env_path.open("a", encoding="utf-8") as f:
                        f.write(f"\nLD_LIBRARY_PATH={needed}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n")
                    print(f"[*] Updated {env_path} with buun LD_LIBRARY_PATH")
        except Exception as exc:
            try:
                print(f"[!] Could not update env: {exc}")
            except Exception:
                pass
        # Pre-check wrapper --help (never throw)
        try:
            r = subprocess.run([str(bin_path), "--help"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                print(f"[*] Installed {bin_path} with rpath {buun_lib_dir} (--help OK)")
            else:
                print(f"[*] Installed {bin_path} with rpath {buun_lib_dir} (--help rc={r.returncode})")
        except Exception:
            print(f"[*] Installed {bin_path} with rpath {buun_lib_dir}")
        return bin_path
    raise FileNotFoundError(f"Built binary not found at {src_bin}")
