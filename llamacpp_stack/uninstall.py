from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import pwd
import sys
from pathlib import Path

try:
    from llamacpp_stack.install import (
        CLI_COMMAND,
        DEFAULT_SERVICE_USER,
        LEGACY_CLI_COMMAND,
        LEGACY_MANAGER_SERVICE_NAME,
        LEGACY_SWAP_SERVICE_NAME,
        MANAGER_SERVICE_NAME,
        MANAGER_WRAPPER_NAME,
        PRODUCT_SLUG,
        SWAP_SERVICE_NAME,
        SWAP_WRAPPER_NAME,
        InstallLayout,
        choose_layout,
        detect_existing_mode,
        env_paths_for_mode,
        existing_models_dir,
        existing_public_host,
        legacy_layout_paths,
        _sudo_prefix,
    )

    try:
        from llamacpp_stack.install import heimdall_legacy_layout_paths
    except ImportError:
        heimdall_legacy_layout_paths = None  # type: ignore[assignment]
except ImportError:
    from install import (
        CLI_COMMAND,
        DEFAULT_SERVICE_USER,
        LEGACY_CLI_COMMAND,
        LEGACY_MANAGER_SERVICE_NAME,
        LEGACY_SWAP_SERVICE_NAME,
        MANAGER_SERVICE_NAME,
        MANAGER_WRAPPER_NAME,
        PRODUCT_SLUG,
        SWAP_SERVICE_NAME,
        SWAP_WRAPPER_NAME,
        InstallLayout,
        choose_layout,
        detect_existing_mode,
        env_paths_for_mode,
        existing_models_dir,
        existing_public_host,
        legacy_layout_paths,
        _sudo_prefix,
    )

    try:
        from install import heimdall_legacy_layout_paths  # type: ignore[import-not-found]
    except ImportError:
        heimdall_legacy_layout_paths = None  # type: ignore[assignment]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _remove_path(path: Path, dry_run: bool, use_sudo: bool = False) -> None:
    if dry_run:
        exists = path.exists() or path.is_symlink()
        # System-mode paths often do not exist on user-mode installs; stay quiet.
        if not exists and use_sudo and _sudo_prefix():
            return
        prefix = "[dry-run] would remove"
        if use_sudo and _sudo_prefix():
            prefix += " (with sudo)"
        suffix = "" if exists else " (not present, would skip)"
        print(f"{prefix} {path}{suffix}")
        return
    if not path.exists() and not path.is_symlink():
        return
    # System paths need sudo when not root
    sudo = _sudo_prefix() if use_sudo and _sudo_prefix() else []
    try:
        if path.is_dir() and not path.is_symlink():
            if sudo:
                _run(sudo + ["rm", "-rf", str(path)], check=False)
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            else:
                shutil.rmtree(path, ignore_errors=True)
        else:
            if sudo:
                _run(sudo + ["rm", "-f", str(path)], check=False)
                if path.exists() or path.is_symlink():
                    path.unlink(missing_ok=True)
            else:
                path.unlink(missing_ok=True)
    except PermissionError:
        # Retry with sudo if not already
        if not sudo and _sudo_prefix():
            _remove_path(path, dry_run=False, use_sudo=True)
        else:
            print(f"[!] Permission denied removing {path} (try with sudo)", file=sys.stderr)
    except Exception as exc:
        print(f"[!] Could not remove {path}: {exc}", file=sys.stderr)


def _remove_dir_preserve_models(directory: Path, models_dir: Path | None, dry_run: bool, use_sudo: bool) -> None:
    """Remove a state/install dir but keep the models subtree when it lives inside it."""
    if models_dir is None:
        _remove_path(directory, dry_run, use_sudo=use_sudo)
        return
    if dry_run:
        exists = directory.exists() or directory.is_symlink()
        if not exists:
            if not (use_sudo and _sudo_prefix()):
                prefix = "[dry-run] would remove"
                if use_sudo and _sudo_prefix():
                    prefix += " (with sudo)"
                print(f"{prefix} {directory} (not present, would skip)")
            return
        try:
            inside = _is_subpath(models_dir, directory)
        except Exception:
            inside = False
        if inside and models_dir.resolve() != directory.resolve():
            print(f"[dry-run] would remove contents of {directory} except {models_dir}")
            try:
                for child in sorted(directory.iterdir()):
                    try:
                        if child.resolve() == models_dir.resolve() or _is_subpath(models_dir, child):
                            print(f"[dry-run]   keeping {child} (contains models)")
                            continue
                    except Exception:
                        pass
                    print(f"[dry-run]   would remove {child}")
            except Exception:
                pass
        else:
            prefix = "[dry-run] would remove"
            if use_sudo and _sudo_prefix():
                prefix += " (with sudo)"
            print(f"{prefix} {directory}")
        return
    if not directory.exists() and not directory.is_symlink():
        return
    try:
        inside = _is_subpath(models_dir, directory)
    except Exception:
        inside = False
    if inside and models_dir.resolve() != directory.resolve():
        try:
            for child in list(directory.iterdir()):
                try:
                    if child.resolve() == models_dir.resolve() or _is_subpath(models_dir, child):
                        continue
                except Exception:
                    pass
                _remove_path(child, dry_run=False, use_sudo=use_sudo)
            try:
                if not any(directory.iterdir()):
                    _remove_path(directory, dry_run=False, use_sudo=use_sudo)
            except Exception:
                pass
        except Exception as exc:
            print(f"[!] Could not clean {directory}: {exc}", file=sys.stderr)
    else:
        _remove_path(directory, dry_run, use_sudo=use_sudo)


def _looks_like_model_file(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.endswith((".gguf", ".gguf-split"))
        or lowered.startswith("consolidated.") and lowered.endswith(".bin")
        or lowered.endswith(".safetensors")
        or lowered.endswith(".pt") and lowered.startswith(("model", "gguf"))
    )


def _has_model_files(path: Path, max_dirs: int = 5000) -> bool:
    """Detect GGUF/llama.cpp model artifacts inside a directory (bounded walk)."""
    if not path.is_dir():
        return False
    visited = 0
    for root, dirs, files in os.walk(path):
        if any(_looks_like_model_file(f) for f in files):
            return True
        visited += 1
        if visited > max_dirs:
            return False
    return False


def _legacy_removal_targets(mode: str, preserve_models: bool) -> list[tuple[Path, Path | None]]:
    """Legacy (pre-Heimdall) + heimdall-gateway paths that the installer historically created.

    Returns (path, models_to_preserve_inside) tuples. Legacy state/install
    dirs are only removed when they do not contain model artifacts; config
    and run dirs never contain models. Covers both 1st-gen (llamacpp-superserver)
    and 2nd-gen (heimdall-gateway) layouts.
    """
    targets: list[tuple[Path, Path | None]] = []

    # Collect from both legacy families: llamacpp-superserver (1st-gen) and
    # heimdall-gateway (2nd-gen). Each provides config/state/install/run dirs.
    legacy_dicts: list[dict[str, Path]] = []
    try:
        legacy_dicts.append(legacy_layout_paths(mode))
    except Exception:
        pass
    # heimdall legacy: use helper if available, else fallback to hardcoded paths
    if heimdall_legacy_layout_paths is not None:
        try:
            legacy_dicts.append(heimdall_legacy_layout_paths(mode))
        except Exception:
            pass
    else:
        # Fallback hardcoded heimdall paths when helper unavailable
        if mode == "system":
            legacy_dicts.append({
                "config_dir": Path("/etc/heimdall-gateway"),
                "state_dir": Path("/var/lib/heimdall-gateway"),
                "install_root": Path("/opt/heimdall-gateway"),
                "run_dir": Path("/run/heimdall-gateway"),
                "systemd_dir": Path("/etc/systemd/system"),
                "bin_dir": Path("/usr/local/bin"),
            })
        else:
            legacy_dicts.append({
                "config_dir": Path.home() / ".config/heimdall-gateway",
                "state_dir": Path.home() / ".local/state/heimdall-gateway",
                "install_root": Path.home() / ".local/opt/heimdall-gateway",
                "run_dir": Path.home() / ".local/run/heimdall-gateway",
                "systemd_dir": Path.home() / ".config/systemd/user",
                "bin_dir": Path.home() / ".local/bin",
            })

    for legacy in legacy_dicts:
        # Config dirs: always safe to remove (env files, server json, legacy unit
        # copies). They never contain models.
        targets.append((legacy["config_dir"], None))
        if "alt_config_dir" in legacy:
            targets.append((legacy["alt_config_dir"], None))
        # Run dirs: sockets/pids only.
        targets.append((legacy["run_dir"], None))
        # State and install roots: skip absent paths; keep when they hold model artifacts.
        for key in ("state_dir", "install_root"):
            path = legacy[key]
            if not (path.exists() or path.is_symlink()):
                continue
            if preserve_models and _has_model_files(path):
                print(f"[i] Keeping {path} (contains model artifacts). Remove manually if desired.")
                continue
            targets.append((path, None))
    # Deduplicate (same path can appear for config/run keys across modes/families)
    seen: set[Path] = set()
    unique: list[tuple[Path, Path | None]] = []
    for path, keep in targets:
        if path in seen:
            continue
        seen.add(path)
        unique.append((path, keep))
    return unique


def _uv_tool_installed() -> bool:
    """True when the CLI is installed via 'uv tool install' (either name)."""
    if not _uv_tool_executable_path():
        return False
    uv = shutil.which("uv")
    if uv is None:
        return False
    try:
        result = _run([uv, "tool", "list"], check=False)
        if result.returncode != 0:
            return False
    except Exception:
        return False
    # Text output looks like:
    #   llm-server v0.1.0 / heimdall-gateway v0.1.0
    #   - llm-server / - heimdall-gateway
    for line in result.stdout.splitlines():
        name = line.strip().lstrip("-").strip()
        if name.startswith(f"{PRODUCT_SLUG} ") or name.startswith("heimdall-gateway "):
            return True
    return False


def _uv_tool_executable_path() -> bool:
    """Detect uv-tool installs without querying uv (fast pre-check)."""
    candidates = [
        Path.home() / ".local" / "bin" / CLI_COMMAND,
        Path("/usr/local/bin") / CLI_COMMAND,
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        # uv tool venvs live under <...>/uv/tools/<tool-name>/bin/<exe>
        parts = resolved.parts
        if "tools" in parts and parts[-2] == "bin":
            idx = parts.index("tools")
            if idx >= 1 and parts[idx - 1] == "uv" and idx + 2 <= len(parts) - 1:
                return True
    return False


def _remove_uv_tool(dry_run: bool) -> None:
    if not _uv_tool_installed():
        return
    if dry_run:
        print("[dry-run] would run: uv tool uninstall llm-server; uv tool uninstall heimdall-gateway (either name)")
        return
    uv = shutil.which("uv")
    if uv is None:
        return
    successes: list[str] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for tool_name in (PRODUCT_SLUG, "heimdall-gateway"):
        result = _run([uv, "tool", "uninstall", tool_name], check=False)
        last_result = result
        if result.returncode == 0:
            print(f"[i] Removed the '{tool_name}' uv tool install.")
            successes.append(tool_name)
        else:
            out = (result.stderr.strip() or result.stdout.strip())
            if "No tools" in out or "not found" in out.lower() or "No package" in out:
                continue
    if not successes and last_result is not None:
        err = (last_result.stderr.strip() or last_result.stdout.strip())
        if err:
            print(f"[i] Could not remove the uv tool install (continuing): {err}")


def _remove_service_user(layout: InstallLayout, dry_run: bool) -> None:
    if layout.mode != "system":
        return
    user = layout.service_user or DEFAULT_SERVICE_USER
    try:
        pwd.getpwnam(user)
    except KeyError:
        return
    if user == pwd.getpwuid(os.geteuid()).pw_name:
        return  # never remove the account we are running as
    cmd = _sudo_prefix() + ["userdel", user]
    if dry_run:
        print(f"[dry-run] would run: {' '.join(cmd)}")
        return
    result = _run(cmd, check=False)
    if result.returncode != 0:
        print(f"[i] Could not remove service user '{user}' (continuing): {result.stderr.strip() or result.stdout.strip()}")


def _ufw_remove_rules(layout: InstallLayout, dry_run: bool) -> None:
    if layout.mode != "system" or layout.public_host != "0.0.0.0":
        return
    if shutil.which("ufw") is None:
        return
    ports = set()
    if layout.public_port:
        ports.update({layout.public_port, layout.public_port - 1})
    api_port = _env_file_value(layout, ("HEIMDALL_GATEWAY_API_PORT",))
    if api_port:
        ports.add(api_port)
    if not ports:
        return
    sudo = _sudo_prefix()
    for port in sorted(ports):
        cmd = sudo + ["ufw", "delete", "allow", f"{port}/tcp"]
        if dry_run:
            print(f"[dry-run] would run: {' '.join(cmd)} (if rule exists)")
            continue
        result = _run(cmd, check=False)
        if result.returncode != 0:
            print(f"[i] ufw rule {port}/tcp not present or could not be removed (continuing).")


def _env_file_value(layout: InstallLayout, keys: tuple[str, ...]) -> int | None:
    for env_path in env_paths_for_mode(layout.mode):
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                clean = line.strip()
                for key in keys:
                    if clean.startswith(f"{key}="):
                        raw = clean.split("=", 1)[1].strip()
                        if raw.isdigit():
                            return int(raw)
        except Exception:
            continue
    return None


def uninstall_systemd_units(layout: InstallLayout, dry_run: bool) -> None:
    if shutil.which("systemctl") is None:
        if dry_run:
            print("[dry-run] systemctl not available; would skip service stop/disable.")
        return

    all_service_names = [
        MANAGER_SERVICE_NAME,
        SWAP_SERVICE_NAME,
        LEGACY_MANAGER_SERVICE_NAME,
        LEGACY_SWAP_SERVICE_NAME,
        "heimdall-gateway-manager.service",
        "heimdall-gateway-router.service",
    ]

    if layout.mode == "user":
        base_systemctl = ["systemctl", "--user"]
        unit_dir = Path.home() / ".config/systemd/user"
        use_sudo = False
    else:
        base_systemctl = _sudo_prefix() + ["systemctl"] if _sudo_prefix() else ["systemctl"]
        unit_dir = Path("/etc/systemd/system")
        use_sudo = True

    # Discover additional LLM Server (incl. legacy Heimdall)/legacy + llm-server units dynamically
    for pattern in ("*llm-server*", "*heimdall*", "*llamacpp-superserver*", "*llamaswap*"):
        try:
            result = _run(base_systemctl + ["list-units", "--all", "--full", "--no-legend", pattern], check=False)
            for line in result.stdout.splitlines():
                parts = line.split()
                if parts:
                    unit_name = parts[0]
                    if unit_name.endswith(".service") and unit_name not in all_service_names:
                        all_service_names.append(unit_name)
        except Exception:
            pass

    # Stop and disable
    if all_service_names:
        stop_cmd = base_systemctl + ["stop"] + all_service_names
        if dry_run:
            print(f"[dry-run] would run {' '.join(stop_cmd)}")
        else:
            _run(stop_cmd, check=False)

        disable_cmd = base_systemctl + ["disable"] + all_service_names
        if dry_run:
            print(f"[dry-run] would run {' '.join(disable_cmd)}")
        else:
            _run(disable_cmd, check=False)

    # Remove unit files and symlinks
    for path in (
        unit_dir / MANAGER_SERVICE_NAME,
        unit_dir / SWAP_SERVICE_NAME,
        unit_dir / LEGACY_MANAGER_SERVICE_NAME,
        unit_dir / LEGACY_SWAP_SERVICE_NAME,
        unit_dir / "heimdall-gateway-manager.service",
        unit_dir / "heimdall-gateway-router.service",
        unit_dir / "default.target.wants" / MANAGER_SERVICE_NAME,
        unit_dir / "default.target.wants" / SWAP_SERVICE_NAME,
        unit_dir / "default.target.wants" / LEGACY_MANAGER_SERVICE_NAME,
        unit_dir / "default.target.wants" / LEGACY_SWAP_SERVICE_NAME,
        unit_dir / "default.target.wants" / "heimdall-gateway-manager.service",
        unit_dir / "default.target.wants" / "heimdall-gateway-router.service",
        unit_dir / "multi-user.target.wants" / MANAGER_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / SWAP_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / LEGACY_MANAGER_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / LEGACY_SWAP_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / "heimdall-gateway-manager.service",
        unit_dir / "multi-user.target.wants" / "heimdall-gateway-router.service",
    ):
        _remove_path(path, dry_run, use_sudo=use_sudo)

    # Scan and remove any residual *heimdall*/*llm-server* unit files not in canonical list
    for pattern in ("*heimdall*", "*llm-server*"):
        try:
            for candidate in unit_dir.glob(pattern):
                if candidate.name.endswith(".service") or candidate.name.endswith(".wants"):
                    _remove_path(candidate, dry_run, use_sudo=use_sudo)
            for candidate in (unit_dir / "default.target.wants").glob(pattern):
                _remove_path(candidate, dry_run, use_sudo=use_sudo)
            for candidate in (unit_dir / "multi-user.target.wants").glob(pattern):
                _remove_path(candidate, dry_run, use_sudo=use_sudo)
        except Exception:
            pass

    reload_cmd = base_systemctl + ["daemon-reload"]
    if dry_run:
        print(f"[dry-run] would run {' '.join(reload_cmd)}")
    else:
        _run(reload_cmd, check=False)


def _layout_for_mode(mode: str) -> InstallLayout:
    # Use existing host/port if available so the plan reflects the real install.
    public_host = existing_public_host(mode) or "127.0.0.1"
    return choose_layout(mode, public_host, None, models_dir=None, args=None)


def _has_sudo_access() -> bool:
    """Return True if we can run sudo non-interactively or are already root."""
    if os.geteuid() == 0:
        return True
    if shutil.which("sudo") is None:
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _systemd_unit_exists(mode: str) -> bool:
    """Check if the LLM Server (legacy Heimdall) manager service exists via systemctl."""
    if shutil.which("systemctl") is None:
        return False
    try:
        if mode == "system":
            # System service: systemctl cat or status
            cat = subprocess.run(
                ["systemctl", "cat", MANAGER_SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if cat.returncode == 0:
                return True
            status = subprocess.run(
                ["systemctl", "status", MANAGER_SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # returncode 4 = not found; 0/3 = exists (active/inactive)
            if status.returncode != 4 and ("Loaded:" in status.stdout or "Loaded:" in status.stderr):
                return True
        else:
            cat = subprocess.run(
                ["systemctl", "--user", "cat", MANAGER_SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if cat.returncode == 0:
                return True
            status = subprocess.run(
                ["systemctl", "--user", "status", MANAGER_SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if status.returncode != 4 and ("Loaded:" in status.stdout or "Loaded:" in status.stderr):
                return True
    except Exception:
        return False
    return False


def _system_install_present() -> bool:
    """Explicit check for system install traces as described in task spec."""
    system_paths = [
        Path("/opt/llm-server"),
        Path("/opt/heimdall-gateway"),
        Path("/etc/llm-server"),
        Path("/etc/heimdall-gateway"),
        Path("/var/lib/llm-server"),
        Path("/var/lib/heimdall-gateway"),
    ]
    if any(p.exists() or p.is_symlink() for p in system_paths):
        return True
    if _systemd_unit_exists("system"):
        return True
    return False


def _detect_modes_to_uninstall(requested_mode: str | None) -> list[str]:
    if requested_mode:
        return [requested_mode]
    # No mode specified: uninstall any mode that has traces
    modes: list[str] = []
    for mode in ("user", "system"):
        layout = _layout_for_mode(mode)
        traces = [
            layout.install_root,
            layout.state_dir,
            layout.config_dir,
            layout.run_dir,
        ]
        # Explicit system paths per task spec (only for system mode) — both families
        if mode == "system":
            traces.extend([
                Path("/opt/llm-server"),
                Path("/opt/heimdall-gateway"),
                Path("/etc/llm-server"),
                Path("/etc/heimdall-gateway"),
                Path("/var/lib/llm-server"),
                Path("/var/lib/heimdall-gateway"),
            ])
        legacy = legacy_layout_paths(mode)
        legacy_traces = [
            legacy["config_dir"],
            legacy["state_dir"],
            legacy["install_root"],
            legacy["run_dir"],
            legacy["systemd_dir"] / MANAGER_SERVICE_NAME,
            legacy["bin_dir"] / CLI_COMMAND,
            legacy["systemd_dir"] / "heimdall-gateway-manager.service",
            legacy["systemd_dir"] / "heimdall-gateway-router.service",
            legacy["bin_dir"] / "heimdall-gateway",
            legacy["bin_dir"] / "heimdall-gateway-manager-start",
            legacy["bin_dir"] / "heimdall-gateway-router-start",
        ]
        if "alt_config_dir" in legacy:
            legacy_traces.append(legacy["alt_config_dir"])
        # Also check heimdall legacy layout directly
        if heimdall_legacy_layout_paths is not None:
            try:
                heimdall_legacy = heimdall_legacy_layout_paths(mode)
                legacy_traces.extend([
                    heimdall_legacy["config_dir"],
                    heimdall_legacy["state_dir"],
                    heimdall_legacy["install_root"],
                    heimdall_legacy["run_dir"],
                    heimdall_legacy["systemd_dir"] / "heimdall-gateway-manager.service",
                    heimdall_legacy["systemd_dir"] / "heimdall-gateway-router.service",
                    heimdall_legacy["bin_dir"] / "heimdall-gateway",
                ])
            except Exception:
                pass
        else:
            # Fallback hardcoded heimdall paths
            if mode == "system":
                legacy_traces.extend([
                    Path("/etc/heimdall-gateway"),
                    Path("/var/lib/heimdall-gateway"),
                    Path("/opt/heimdall-gateway"),
                    Path("/run/heimdall-gateway"),
                    Path("/etc/systemd/system/heimdall-gateway-manager.service"),
                    Path("/usr/local/bin/heimdall-gateway"),
                ])
            else:
                legacy_traces.extend([
                    Path.home() / ".config/heimdall-gateway",
                    Path.home() / ".local/state/heimdall-gateway",
                    Path.home() / ".local/opt/heimdall-gateway",
                    Path.home() / ".local/run/heimdall-gateway",
                    Path.home() / ".config/systemd/user/heimdall-gateway-manager.service",
                    Path.home() / ".local/bin/heimdall-gateway",
                ])
        has_file_traces = any(p.exists() or p.is_symlink() for p in [*traces, *legacy_traces])
        has_systemd_traces = _systemd_unit_exists(mode)
        if has_file_traces or has_systemd_traces:
            modes.append(mode)
    if not modes:
        # Fallback: at least try the detected existing mode, or user
        detected = detect_existing_mode()
        if detected:
            return [detected]
        # Also check explicit system present even if detect_existing_mode missed
        if _system_install_present():
            return ["system"]
        return ["user"]
    return modes


def _collect_bin_targets(layout: InstallLayout) -> list[Path]:
    bin_dir = layout.bin_dir
    seen: set[str] = set()
    targets: list[Path] = []
    for name in (
        MANAGER_WRAPPER_NAME,
        SWAP_WRAPPER_NAME,
        CLI_COMMAND,
        LEGACY_CLI_COMMAND,
        "heimdall-gateway",
        "heimdall-gateway-manager-start",
        "heimdall-gateway-router-start",
        "llamacpp-manager-start",
        "llamaswap-start",
        "llamacpp-superserver",
        "llamacpp-server",
        "llamacpp-stack-install",
        "llamacpp-stack-uninstall",
        "vllm-server",
    ):
        if name not in seen:
            seen.add(name)
            targets.append(bin_dir / name)
    # Also handle alternative bin locations: /usr/local/bin, ~/.local/bin, ~/.local/opt/*/bin
    extra_bin_dirs: list[Path] = []
    if layout.mode == "user":
        extra_bin_dirs.append(Path.home() / ".local/bin")
        extra_bin_dirs.append(Path("/usr/local/bin"))
        try:
            opt_base = Path.home() / ".local/opt"
            if opt_base.exists():
                for child in opt_base.iterdir():
                    bin_candidate = child / "bin"
                    if bin_candidate.is_dir():
                        extra_bin_dirs.append(bin_candidate)
        except Exception:
            pass
        extra_bin_dirs.append(Path.home() / ".local/opt/llm-server/bin")
        extra_bin_dirs.append(Path.home() / ".local/opt/heimdall-gateway/bin")
    else:
        extra_bin_dirs.append(Path("/usr/local/bin"))
        extra_bin_dirs.append(Path.home() / ".local/bin")
    for extra_dir in extra_bin_dirs:
        if extra_dir == bin_dir:
            continue
        for name in (CLI_COMMAND, "heimdall-gateway", MANAGER_WRAPPER_NAME, SWAP_WRAPPER_NAME, "heimdall-gateway-manager-start", "heimdall-gateway-router-start"):
            key = f"{extra_dir}:{name}"
            if key not in seen:
                seen.add(key)
                targets.append(extra_dir / name)
    return targets


def _build_confirmation(modes: list[str], per_mode_info: list[tuple[InstallLayout, Path]], remove_models: bool) -> tuple[list[str], list[str]]:
    removals: list[str] = []
    for layout, mdir in per_mode_info:
        if remove_models:
            removals.append(str(layout.state_dir))
        else:
            removals.append(f"{layout.state_dir} (except models)")
        removals.extend([
            str(layout.install_root),
            str(layout.config_dir),
            str(layout.run_dir),
            str(layout.bin_dir / CLI_COMMAND),
            str(layout.bin_dir / MANAGER_WRAPPER_NAME),
            str(layout.bin_dir / SWAP_WRAPPER_NAME),
        ])
        for legacy_path, _keep in _legacy_removal_targets(layout.mode, not remove_models):
            removals.append(str(legacy_path))
        if layout.mode == "system":
            removals.append(f"/etc/systemd/system/{MANAGER_SERVICE_NAME}")
            removals.append(f"/etc/systemd/system/{SWAP_SERVICE_NAME}")
            removals.append("/etc/systemd/system/heimdall-gateway-manager.service")
            removals.append("/etc/systemd/system/heimdall-gateway-router.service")
            removals.append(f"service user '{layout.service_user}'")
            removals.append("ufw rules for the gateway ports (if present)")
        else:
            removals.append(f"~/.config/systemd/user/{MANAGER_SERVICE_NAME}")
            removals.append(f"~/.config/systemd/user/{SWAP_SERVICE_NAME}")
            removals.append("~/.config/systemd/user/heimdall-gateway-manager.service")
            removals.append("~/.config/systemd/user/heimdall-gateway-router.service")
    removals.append("'llm-server' + 'heimdall-gateway' uv tool installs (if installed via 'uv tool install')")
    kept: list[str] = [str(mdir) for _l, mdir in per_mode_info] if not remove_models else []

    seen = set()
    uniq: list[str] = []
    for r in removals:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq, kept


def uninstall_stack(args: argparse.Namespace) -> int:
    modes = _detect_modes_to_uninstall(getattr(args, "mode", None))
    dry_run: bool = bool(getattr(args, "dry_run", False))
    remove_models: bool = bool(getattr(args, "remove_models", False))
    assume_yes: bool = bool(getattr(args, "yes", False) or getattr(args, "keep_models", False))

    if not dry_run and "system" in modes:
        if not _has_sudo_access():
            print(
                "System install detected at /opt/heimdall-gateway. Uninstall requires sudo. Please run with sudo or ensure passwordless sudo.",
                file=sys.stderr,
            )
            return 1
        if os.geteuid() != 0 and sys.stdin.isatty():
            try:
                subprocess.run(["sudo", "-v"], timeout=10)
            except Exception:
                pass

    # Gather per-mode info for the plan/confirmation message
    per_mode_info: list[tuple[InstallLayout, Path]] = []
    for mode in modes:
        layout = _layout_for_mode(mode)
        # Prefer existing_models_dir if known, else layout.models_dir
        mdir = existing_models_dir(mode) or layout.models_dir
        per_mode_info.append((layout, mdir))

    uniq_removals, kept = _build_confirmation(modes, per_mode_info, remove_models)

    # Confirmation prompt (skipped in dry-run or with --yes)
    if not dry_run and not assume_yes:
        print("This will remove LLM Server installations from:")
        for r in uniq_removals:
            print(f"  - {r}")
        if kept:
            print(f"Models will be KEPT at: {', '.join(kept)}")
            print("If you want to delete the models afterwards, remove them manually, e.g.:")
            for path in kept:
                print(f"  rm -rf {path}")
        else:
            print("WARNING: models will also be removed (--remove-models).")
        print()
        try:
            if sys.stdin.isatty():
                answer = input("Continue with the uninstall? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("Aborted.")
                    return 1
            else:
                # Non-TTY without --yes: require explicit confirmation flag
                print("Non-interactive terminal: use --yes to confirm the uninstall.", file=sys.stderr)
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1

    if dry_run:
        models_paths = ", ".join(str(mdir) for _l, mdir in per_mode_info)
        if remove_models:
            print(f"[dry-run] Would remove LLM Server ({', '.join(modes)} mode) including models at {models_paths}")
        else:
            print(f"[dry-run] Would remove LLM Server ({', '.join(modes)} mode) keeping models at {models_paths}")

    # Perform removal per mode
    for layout, models_dir in per_mode_info:
        use_sudo = layout.mode == "system"
        print(f"{'[dry-run] ' if dry_run else ''}Uninstalling LLM Server ({layout.mode} mode)...")
        uninstall_systemd_units(layout, dry_run)

        if remove_models:
            _remove_path(models_dir, dry_run, use_sudo=use_sudo)

        _ufw_remove_rules(layout, dry_run)
        _remove_service_user(layout, dry_run)

        # Remove bin wrappers/symlinks
        for bin_target in _collect_bin_targets(layout):
            _remove_path(bin_target, dry_run, use_sudo=use_sudo)

        # Remove install_root (contains venv, cuda, etc.)
        _remove_path(layout.install_root, dry_run, use_sudo=use_sudo)

        # Remove config_dir
        _remove_path(layout.config_dir, dry_run, use_sudo=use_sudo)

        # Remove state_dir except models (unless --remove-models)
        preserve = None if remove_models else models_dir
        _remove_dir_preserve_models(layout.state_dir, preserve, dry_run, use_sudo)

        # Remove run_dir
        _remove_path(layout.run_dir, dry_run, use_sudo=use_sudo)

        # Legacy state/config traces (only if not already covered) - keep models there too
        for legacy_path, _keep in _legacy_removal_targets(layout.mode, not remove_models):
            _remove_dir_preserve_models(legacy_path, models_dir if not remove_models else None, dry_run, use_sudo)

    # uv tool install (single, shared across modes)
    _remove_uv_tool(dry_run)

    # Final message with manual models deletion instruction (dedup by path)
    reported: set[str] = set()
    for _, models_dir in per_mode_info:
        key = str(models_dir)
        if key in reported:
            continue
        reported.add(key)
        if dry_run:
            if remove_models:
                print(f"[dry-run] would remove models at {models_dir}")
            else:
                print(f"[dry-run] would keep models at {models_dir}. To delete manually: rm -rf {models_dir}")
        else:
            if remove_models:
                print(f"Removed models at {models_dir}.")
            else:
                print(f"Kept models at {models_dir}. To delete manually if desired: rm -rf {models_dir}")

    if dry_run:
        print("Dry-run complete. No changes were made.")
    else:
        suffix = " (models removed)" if remove_models else " (models preserved)"
        print(f"LLM Server ({', '.join(modes)} mode) uninstalled{suffix}.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uninstall LLM Server (removes all traces except models).",
        epilog=(
            "Models are never deleted automatically. Remove them manually with: "
            "rm -rf <models_dir>   (or pass --remove-models to delete them as part of the uninstall)"
        ),
    )
    parser.add_argument("--mode", choices=("system", "user"), help="Install mode to uninstall (default: auto-detect all existing).")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without removing anything.")
    parser.add_argument("--remove-models", action="store_true", help="Also remove the models directory (normally kept).")
    # Keep hidden compat for old flag
    parser.add_argument("--keep-models", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--public-host", help=argparse.SUPPRESS, default="127.0.0.1")
    parser.add_argument("--public-port", type=int, help=argparse.SUPPRESS, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return uninstall_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
