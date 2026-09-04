"""Registry for updatable Heimdall Gateway dependencies.

Each dependency (llama.cpp, llama-swap, vLLM, etc.) registers itself via
``register_dependency`` so ``heimdall-gateway update`` can discover and
update it without hard-coding the list in the CLI layer.

Adding support for a new technology:

    from llamacpp_stack.dependencies import register_dependency, InstallLayout

    @register_dependency("my-tech", "My new backend", aliases=["mytech"])
    def update_my_tech(layout, *, dry_run: bool = False, force: bool = False, **kwargs):
        # fetch latest, compare with current, install if needed
        return {"name": "my-tech", "old": old_version, "new": new_version, "action": "updated|skipped|dry-run"}

The registry is intentionally import-side-effect free for the concrete
updaters: ``dependencies`` only imports heavy ``install`` helpers lazily
inside each updater so CLI startup remains fast and avoids circular imports.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Registry core
# ---------------------------------------------------------------------------

@dataclass
class Dependency:
    name: str
    description: str
    aliases: tuple[str, ...] = ()
    updater: Callable[..., dict] | None = None

    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


REGISTRY: dict[str, Dependency] = {}
_ALIAS_MAP: dict[str, str] = {}  # lower alias -> canonical name


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def register_dependency(name: str, description: str, *, aliases: tuple[str, ...] | list[str] = ()) -> Callable:
    """Decorator to register an updater for a dependency.

    Example::

        @register_dependency("llama.cpp", "llama.cpp engine", aliases=["llamacpp"])
        def update_llamacpp(layout, *, dry_run=False, force=False, **kw):
            ...

    Future techs only need to add a module that imports and uses this
    decorator; ``heimdall-gateway update`` will automatically include them.
    """

    def decorator(func: Callable) -> Callable:
        canonical = str(name).strip()
        norm_canonical = _normalize_key(canonical)
        alias_tuple = tuple(str(a).strip() for a in (aliases or ()) if str(a).strip())
        dep = Dependency(name=canonical, description=description, aliases=alias_tuple, updater=func)
        REGISTRY[canonical] = dep
        # fill alias map (both canonical and aliases map to canonical)
        _ALIAS_MAP[norm_canonical] = canonical
        for alias in alias_tuple:
            _ALIAS_MAP[_normalize_key(alias)] = canonical
        # keep original updater attribute for introspection
        func._dependency_name = canonical  # type: ignore[attr-defined]
        return func

    return decorator


def all_dependencies() -> list[Dependency]:
    return list(REGISTRY.values())


def get_dependency(name: str) -> Dependency | None:
    key = _normalize_key(name)
    canonical = _ALIAS_MAP.get(key)
    if canonical is None:
        return None
    return REGISTRY.get(canonical)


def resolve_requested(names: list[str] | None) -> list[Dependency]:
    if not names:
        return all_dependencies()
    resolved: list[Dependency] = []
    seen: set[str] = set()
    for raw in names:
        # allow comma-separated lists: "llamacpp,vllm"
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        for part in parts:
            dep = get_dependency(part)
            if dep is None:
                available = ", ".join(d.name for d in all_dependencies())
                raise RuntimeError(f"Unknown dependency '{part}'. Available: {available}")
            if dep.name not in seen:
                seen.add(dep.name)
                resolved.append(dep)
    return resolved


def list_dependencies_text() -> str:
    lines = ["Registered updatable dependencies:"]
    for dep in sorted(all_dependencies(), key=lambda d: d.name.lower()):
        alias_str = f" (aliases: {', '.join(dep.aliases)})" if dep.aliases else ""
        lines.append(f"  - {dep.name}: {dep.description}{alias_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers shared by built-in updaters
# ---------------------------------------------------------------------------

def _detect_layout_from_args(args) -> object:
    """Build an InstallLayout from CLI args or from existing install."""
    try:
        from llamacpp_stack.install import choose_layout, detect_existing_mode
    except Exception as exc:
        raise RuntimeError(f"Could not load install layout helpers: {exc}") from exc

    public_host = str(getattr(args, "public_host", "127.0.0.1") or "127.0.0.1")
    public_port = getattr(args, "public_port", None)
    # detect mode from existing install if not forced
    mode = detect_existing_mode()
    return choose_layout(mode, public_host, public_port, models_dir=None, args=args)


def _read_manifest(layout) -> dict:
    try:
        from llamacpp_stack.install import read_install_manifest

        return read_install_manifest(layout)
    except Exception:
        return {}


def _run_capture(cmd: list[str], timeout: float = 8.0) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Built-in updaters
# ---------------------------------------------------------------------------

@register_dependency("llama.cpp", "llama.cpp inference engine (llama-server)", aliases=["llamacpp", "llama-cpp", "llamacpp-stack"])
def update_llamacpp(layout, *, dry_run: bool = False, force: bool = False, **kwargs) -> dict:
    import os as _os

    from llamacpp_stack.install import (
        DEFAULT_LLAMA_CPP_REPO,
        LLAMA_CPP_REF_ENV,
        LEGACY_LLAMA_CPP_REF_ENV,
        LLAMA_CPP_REF_PROMPTED_ENV,
        LEGACY_LLAMA_CPP_REF_PROMPTED_ENV,
        choose_llamacpp_linux_asset,
        detect_existing_llama_cpp_mode,
        dry_run_release_placeholder,
        install_release_asset,
        latest_release,
        build_llama_cpp_from_source,
        detect_nvidia_gpu,
        locate_nvcc,
        write_manifest,
        _link_stable_binary,
        read_install_manifest,
    )

    manifest = read_install_manifest(layout)
    current = str(manifest.get("llama_cpp_tag") or "unknown")
    strategy = str(manifest.get("llama_cpp_strategy") or detect_existing_llama_cpp_mode(layout))
    if strategy == "native":
        return {"name": "llama.cpp", "old": current, "new": current, "action": "skipped", "reason": "native mode: system package manages llama.cpp"}

    explicit_ref = str(kwargs.get("llama_cpp_ref") or _os.environ.get(LLAMA_CPP_REF_ENV) or _os.environ.get(LEGACY_LLAMA_CPP_REF_ENV) or "").strip()
    if not explicit_ref:
        explicit_ref = str(kwargs.get("llama_cpp_ref_arg") or "").strip()
    prompted_ref = ""
    if not explicit_ref and not dry_run and sys.stdin.isatty():
        try:
            from llamacpp_stack.install import prompt_choice

            if not _os.environ.get(LLAMA_CPP_REF_PROMPTED_ENV) and not _os.environ.get(LEGACY_LLAMA_CPP_REF_PROMPTED_ENV):
                source_choice = prompt_choice(
                    "Which llama.cpp source version should be built?",
                    [
                        ("latest", "use the latest llama.cpp release (default)"),
                        ("commit", "build a specific git commit/tag/ref"),
                    ],
                    default="latest",
                )
                _os.environ[LLAMA_CPP_REF_PROMPTED_ENV] = "1"
                if source_choice == "commit":
                    while True:
                        raw = input("llama.cpp commit/tag/ref: ").strip()
                        if raw:
                            prompted_ref = raw
                            break
                        print("Please enter a non-empty commit/tag/ref, or press Ctrl+C to cancel.")
            explicit_ref = prompted_ref
        except Exception:
            pass

    if explicit_ref:
        release = {"tag_name": explicit_ref, "source_kind": "ref", "assets": []}
        target_tag = explicit_ref
        force_source = True
    else:
        try:
            release = latest_release(DEFAULT_LLAMA_CPP_REPO)
        except Exception:
            try:
                release = dry_run_release_placeholder(DEFAULT_LLAMA_CPP_REPO)
            except Exception as exc:
                return {"name": "llama.cpp", "old": current, "new": current, "action": "error", "reason": f"could not fetch latest release: {exc}"}
        target_tag = str(release.get("tag_name") or current)
        force_source = False
        if not force and target_tag == current and current != "unknown":
            return {"name": "llama.cpp", "old": current, "new": target_tag, "action": "skipped", "reason": "already at latest"}

    if dry_run:
        return {"name": "llama.cpp", "old": current, "new": target_tag, "action": "dry-run"}

    try:
        if force_source or strategy.startswith("source"):
            # rebuild from source tag
            enable_cuda = detect_nvidia_gpu() and locate_nvcc() is not None
            llama_server_real = build_llama_cpp_from_source(release, layout.install_root, enable_tls=False, dry_run=False, python_exec=sys.executable, enable_cuda=enable_cuda)
            _link_stable_binary(llama_server_real, layout.install_root / "llama-server", dry_run=False)
        else:
            asset = choose_llamacpp_linux_asset(release)
            if asset is None:
                return {"name": "llama.cpp", "old": current, "new": target_tag, "action": "error", "reason": "no prebuilt asset for this platform"}
            cpp_root = install_release_asset(asset, layout.install_root, dry_run=False)
            # link using install helper
            from llamacpp_stack.install import _find_executable

            llama_server_real = _find_executable(cpp_root, "llama-server")
            _link_stable_binary(llama_server_real, layout.install_root / "llama-server", dry_run=False)

        # update manifest
        llamaswap_tag = str(manifest.get("llamaswap_tag") or "unknown")
        backend = str(manifest.get("backend") or "auto")
        write_manifest(layout, target_tag, llamaswap_tag, strategy, backend, dry_run=False)
        return {"name": "llama.cpp", "old": current, "new": target_tag, "action": "updated"}
    except Exception as exc:
        return {"name": "llama.cpp", "old": current, "new": target_tag, "action": "error", "reason": str(exc)}


@register_dependency("llama-swap", "llama-swap router/proxy", aliases=["llamaswap", "swap", "llama-swap-router"])
def update_llamaswap(layout, *, dry_run: bool = False, force: bool = False, **kwargs) -> dict:
    from llamacpp_stack.install import (
        DEFAULT_LLAMASWAP_REPO,
        choose_llamaswap_asset,
        dry_run_release_placeholder,
        install_release_asset,
        latest_release,
        write_manifest,
        _link_stable_binary,
        read_install_manifest,
    )

    manifest = read_install_manifest(layout)
    current = str(manifest.get("llamaswap_tag") or "unknown")
    llama_cpp_tag = str(manifest.get("llama_cpp_tag") or "unknown")
    strategy = str(manifest.get("llama_cpp_strategy") or "unknown")
    backend = str(manifest.get("backend") or "auto")

    try:
        release = latest_release(DEFAULT_LLAMASWAP_REPO)
    except Exception:
        try:
            release = dry_run_release_placeholder(DEFAULT_LLAMASWAP_REPO)
        except Exception as exc:
            return {"name": "llama-swap", "old": current, "new": current, "action": "error", "reason": f"could not fetch latest release: {exc}"}

    target_tag = str(release.get("tag_name") or current)
    if not force and target_tag == current and current != "unknown":
        return {"name": "llama-swap", "old": current, "new": target_tag, "action": "skipped", "reason": "already at latest"}

    if dry_run:
        return {"name": "llama-swap", "old": current, "new": target_tag, "action": "dry-run"}

    try:
        asset = choose_llamaswap_asset(release)
        swap_root = install_release_asset(asset, layout.install_root, dry_run=False)
        from llamacpp_stack.install import _find_executable

        swap_real = _find_executable(swap_root, "llama-swap")
        _link_stable_binary(swap_real, layout.install_root / "llama-swap", dry_run=False)
        write_manifest(layout, llama_cpp_tag, target_tag, strategy, backend, dry_run=False)
        return {"name": "llama-swap", "old": current, "new": target_tag, "action": "updated"}
    except Exception as exc:
        return {"name": "llama-swap", "old": current, "new": target_tag, "action": "error", "reason": str(exc)}


@register_dependency("vllm", "vLLM backend (HF-native models)", aliases=["vLLM", "vllm-beta"])
def update_vllm(layout, *, dry_run: bool = False, force: bool = False, **kwargs) -> dict:
    runtime_python = layout.runtime_venv / "bin" / "python"
    current = "unknown"
    if runtime_python.exists():
        out = _run_capture([str(runtime_python), "-c", "import importlib.metadata; print(importlib.metadata.version('vllm'))"])
        if out:
            current = out.splitlines()[0].strip()

    if dry_run:
        # try to fetch latest via pip index without installing
        return {"name": "vllm", "old": current, "new": "latest", "action": "dry-run"}

    try:
        from llamacpp_stack.install import ensure_runtime_python, sync_cuda_runtime, sync_nccl_runtime, resolve_uv_executable

        uv_bin = resolve_uv_executable()
        if uv_bin is None:
            return {"name": "vllm", "old": current, "new": current, "action": "error", "reason": "uv not found, cannot upgrade vLLM"}

        # ensure_runtime_python reinstalls venv with latest vllm
        # but we want to upgrade in place if venv exists for speed
        if runtime_python.exists():
            # try pip upgrade in existing venv
            result = subprocess.run(
                [uv_bin, "pip", "install", "--python", str(runtime_python), "--upgrade", "vllm", "--torch-backend=auto"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            # fallback to full venv recreation on failure
            if result.returncode != 0:
                ensure_runtime_python(layout, dry_run=False, skip_pip_install=False)
        else:
            ensure_runtime_python(layout, dry_run=False, skip_pip_install=False)

        # sync cuda/nccl runtime after upgrade
        try:
            python_exec = str(runtime_python) if runtime_python.exists() else sys.executable
            sync_cuda_runtime(layout, python_exec, dry_run=False)
            sync_nccl_runtime(layout, python_exec, dry_run=False)
        except Exception:
            pass

        new_ver = _run_capture([str(runtime_python), "-c", "import importlib.metadata; print(importlib.metadata.version('vllm'))"]) or "unknown"
        action = "updated" if new_ver != current else "skipped"
        return {"name": "vllm", "old": current, "new": new_ver, "action": action}
    except Exception as exc:
        return {"name": "vllm", "old": current, "new": current, "action": "error", "reason": str(exc)}


@register_dependency(
    "heimdall-gateway",
    "Heimdall Gateway Python package",
    aliases=["heimdall", "gateway", "heimdall-gateway-manager"],
)
def update_heimdall_gateway(layout, *, dry_run: bool = False, force: bool = False, **kwargs) -> dict:
    try:
        from importlib.metadata import version as pkg_version
        current = pkg_version("heimdall-gateway")
    except Exception:
        current = "unknown"

    if dry_run:
        return {"name": "heimdall-gateway", "old": current, "new": "latest", "action": "dry-run"}

    try:
        from llamacpp_stack.install import resolve_uv_executable

        runtime_python = layout.runtime_venv / "bin" / "python"
        uv_bin = resolve_uv_executable()

        # Prefer upgrading the runtime venv's copy; also upgrade host env if running from host
        targets: list[list[str]] = []
        if uv_bin and runtime_python.exists():
            targets.append([uv_bin, "pip", "install", "--python", str(runtime_python), "--upgrade", "heimdall-gateway"])
        # Also upgrade the current python environment (host) when not inside runtime venv
        # Use pip/uv if available
        if uv_bin:
            targets.append([uv_bin, "pip", "install", "--python", sys.executable, "--upgrade", "heimdall-gateway"])
        else:
            targets.append([sys.executable, "-m", "pip", "install", "--upgrade", "heimdall-gateway"])

        last_error = None
        for cmd in targets:
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            except subprocess.CalledProcessError as exc:
                last_error = exc.stderr or str(exc)
                continue
        # Re-sync python package into install root after upgrade
        try:
            from llamacpp_stack.install import ensure_runtime_python

            # copy updated package tree into python_root without recreating venv
            source_pkg = Path(__file__).resolve().parent
            target_pkg = layout.python_root / "llamacpp_stack"
            # If upgrade succeeded we still refresh the python_root copy
            if target_pkg.exists():
                import shutil

                if target_pkg.is_symlink():
                    target_pkg.unlink()
                elif target_pkg.exists():
                    import shutil as _shutil

                    _shutil.rmtree(target_pkg)
                import shutil as _shutil2

                _shutil2.copytree(source_pkg, target_pkg, ignore=_shutil2.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        except Exception:
            pass

        try:
            from importlib.metadata import version as pkg_version2

            new_ver = pkg_version2("heimdall-gateway")
        except Exception:
            new_ver = current

        if last_error and new_ver == current:
            return {"name": "heimdall-gateway", "old": current, "new": new_ver, "action": "error", "reason": last_error}
        action = "updated" if new_ver != current else "skipped"
        return {"name": "heimdall-gateway", "old": current, "new": new_ver, "action": action}
    except Exception as exc:
        return {"name": "heimdall-gateway", "old": current, "new": current, "action": "error", "reason": str(exc)}


# ---------------------------------------------------------------------------
# Public entrypoint for CLI
# ---------------------------------------------------------------------------

def _prompt_interactive_dependency_selection(all_deps: list[Dependency]) -> list[Dependency]:
    print("\nAvailable dependencies:")
    for idx, dep in enumerate(all_deps, 1):
        print(f"  {idx}. {dep.name:<18} {dep.description}")
    print()
    print("Enter 'all' or comma-separated numbers/names (e.g. 1,3 or llama.cpp,vllm)")
    while True:
        raw = input("Selection [all]: ").strip()
        if not raw or raw.lower() == "all":
            return all_deps
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        selected: list[Dependency] = []
        seen: set[str] = set()
        ok = True
        for p in parts:
            dep = None
            if p.isdigit():
                idx = int(p)
                if 1 <= idx <= len(all_deps):
                    dep = all_deps[idx - 1]
                else:
                    print(f"Invalid number: {p}")
                    ok = False
                    break
            else:
                dep = get_dependency(p)
                if dep is None:
                    print(f"Unknown dependency: {p}. Available: {', '.join(d.name for d in all_deps)}")
                    ok = False
                    break
            if dep is not None and dep.name not in seen:
                selected.append(dep)
                seen.add(dep.name)
        if ok and selected:
            return selected
        print("Please enter valid dependency names or numbers.")


def handle_dependency_update(args) -> int:
    if bool(getattr(args, "list_deps", False)) or bool(getattr(args, "list", False)):
        print(list_dependencies_text())
        return 0

    dry_run = bool(getattr(args, "dry_run", False) or getattr(args, "check", False))
    force = bool(getattr(args, "force", False))
    yes = bool(getattr(args, "yes", False))
    verbose = bool(getattr(args, "verbose", False))

    requested_raw: list[str] = []
    for attr in ("only", "component", "components", "deps_only"):
        val = getattr(args, attr, None)
        if val:
            if isinstance(val, list):
                requested_raw.extend(val)
            elif isinstance(val, str):
                requested_raw.append(val)
    if not requested_raw and bool(getattr(args, "deps", False)):
        repo_vals = getattr(args, "repo", None) or []
        if isinstance(repo_vals, list) and repo_vals:
            dep_candidates = [r for r in repo_vals if "/" not in r and ":" not in r]
            if dep_candidates and len(dep_candidates) == len(repo_vals):
                requested_raw.extend(dep_candidates)

    if not requested_raw and sys.stdin.isatty() and not yes:
        try:
            all_deps = all_dependencies()
            selected = _prompt_interactive_dependency_selection(all_deps)
            requested_raw = [d.name for d in selected]
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        except Exception:
            pass

    try:
        deps = resolve_requested(requested_raw if requested_raw else None)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Detect layout (may fail if no install yet)
    try:
        layout = _detect_layout_from_args(args)
    except Exception as exc:
        print(f"[!] No existing Heimdall Gateway installation detected: {exc}", file=sys.stderr)
        print("Run 'heimdall-gateway install --mode user --backend auto' first.", file=sys.stderr)
        return 1

    # Verify install exists
    manifest = _read_manifest(layout)
    if not manifest and not dry_run:
        print(f"[!] No install manifest at {layout.state_dir / 'install-manifest.json'}", file=sys.stderr)
        print("Run install first, or use --dry-run/--check to preview.", file=sys.stderr)
        if not yes:
            return 1

    if not dry_run and not yes and not force:
        names = ", ".join(d.name for d in deps)
        # interactive confirmation unless --yes or non-tty
        try:
            if sys.stdin.isatty():
                from llamacpp_stack.install import prompt_bool

                if not prompt_bool(f"Update {len(deps)} component(s) ({names}) now?", default=True):
                    print("Aborted.")
                    return 0
        except Exception:
            pass

    llama_cpp_ref_arg = getattr(args, "llama_cpp_ref", None)
    results: list[dict] = []
    for dep in deps:
        if verbose:
            print(f"[*] Updating {dep.name} ...")
        assert dep.updater is not None
        try:
            res = dep.updater(layout, dry_run=dry_run, force=force, verbose=verbose, llama_cpp_ref=llama_cpp_ref_arg, llama_cpp_ref_arg=llama_cpp_ref_arg)
        except Exception as exc:
            res = {"name": dep.name, "old": "unknown", "new": "unknown", "action": "error", "reason": str(exc)}
        results.append(res)
        action = res.get("action", "unknown")
        old = res.get("old", "?")
        new = res.get("new", "?")
        reason = res.get("reason", "")
        if action == "updated":
            print(f"[✓] {dep.name}: {old} -> {new} (updated)")
        elif action == "dry-run":
            print(f"[dry-run] {dep.name}: {old} -> {new} would be updated")
        elif action == "skipped":
            suffix = f" ({reason})" if reason else ""
            print(f"[-] {dep.name}: {old} (skipped{suffix})")
        else:
            suffix = f": {reason}" if reason else ""
            print(f"[!] {dep.name}: {old} -> {new} ({action}{suffix})", file=sys.stderr)

    # Summary + optional service restart
    updated = [r for r in results if r.get("action") == "updated"]
    failed = [r for r in results if r.get("action") == "error"]
    if updated and not dry_run:
        print(f"\nUpdated {len(updated)}/{len(results)} component(s).")
        # Offer to restart services so new binaries take effect
        try:
            from llamacpp_stack.install import restart_systemd_units

            restart_systemd_units(layout, dry_run=False)
        except Exception as exc:
            print(f"[!] Could not restart services automatically: {exc}", file=sys.stderr)
            print(f"Restart manually: systemctl --user restart heimdall-gateway-manager heimdall-gateway-router", file=sys.stderr)
    elif dry_run:
        print(f"\nDry-run complete: {len(results)} component(s) checked.")
    else:
        print(f"\nNo updates applied ({len(results)} checked).")

    return 1 if failed else 0


try:
    import importlib as _importlib

    for _extra in ("llamacpp_stack.contrib_dependencies", "llamacpp_stack.plugins.dependencies"):
        try:
            _importlib.import_module(_extra)
        except ImportError:
            continue
        except Exception:
            continue
except Exception:
    pass
