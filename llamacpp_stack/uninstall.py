from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

try:
    from llamacpp_stack.install import (
        CLI_COMMAND,
        LEGACY_CLI_COMMAND,
        LEGACY_MANAGER_SERVICE_NAME,
        LEGACY_SWAP_SERVICE_NAME,
        MANAGER_SERVICE_NAME,
        MANAGER_WRAPPER_NAME,
        SWAP_SERVICE_NAME,
        SWAP_WRAPPER_NAME,
        InstallLayout,
        choose_layout,
        detect_existing_mode,
        prompt_bool,
        _sudo_prefix,
    )
except ImportError:
    from install import (
        CLI_COMMAND,
        LEGACY_CLI_COMMAND,
        LEGACY_MANAGER_SERVICE_NAME,
        LEGACY_SWAP_SERVICE_NAME,
        MANAGER_SERVICE_NAME,
        MANAGER_WRAPPER_NAME,
        SWAP_SERVICE_NAME,
        SWAP_WRAPPER_NAME,
        InstallLayout,
        choose_layout,
        detect_existing_mode,
        prompt_bool,
        _sudo_prefix,
    )


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if dry_run:
        print(f"[dry-run] would remove {path}")
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


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
    ]

    base_systemctl = ["systemctl"]
    if layout.mode == "user":
        base_systemctl.append("--user")
        unit_dir = Path.home() / ".config/systemd/user"
    else:
        base_systemctl = _sudo_prefix() + ["systemctl"]
        unit_dir = Path("/etc/systemd/system")

    # 1. Dynamically find Heimdall units plus explicit legacy service names.
    for pattern in ("*heimdall*", "*llamacpp-superserver*", "*llamaswap*"):
        try:
            list_units_cmd = base_systemctl + ["list-units", "--all", "--full", "--no-legend", pattern]
            result = _run(list_units_cmd, check=False)
            for line in result.stdout.splitlines():
                parts = line.split()
                if parts:
                    unit_name = parts[0]
                    if unit_name.endswith(".service") and unit_name not in all_service_names:
                        all_service_names.append(unit_name)
        except Exception:
            pass

    # 2. Stop and disable all of them
    if all_service_names:
        # First stop them all to make sure port is freed
        stop_cmd = base_systemctl + ["stop"] + all_service_names
        if dry_run:
            print(f"[dry-run] would run {' '.join(stop_cmd)}")
        else:
            _run(stop_cmd, check=False)

        # Then disable them
        disable_cmd = base_systemctl + ["disable"] + all_service_names
        if dry_run:
            print(f"[dry-run] would run {' '.join(disable_cmd)}")
        else:
            _run(disable_cmd, check=False)

    # 3. Remove known unit files and symlinks
    for path in (
        unit_dir / MANAGER_SERVICE_NAME,
        unit_dir / SWAP_SERVICE_NAME,
        unit_dir / LEGACY_MANAGER_SERVICE_NAME,
        unit_dir / LEGACY_SWAP_SERVICE_NAME,
        unit_dir / "default.target.wants" / MANAGER_SERVICE_NAME,
        unit_dir / "default.target.wants" / SWAP_SERVICE_NAME,
        unit_dir / "default.target.wants" / LEGACY_MANAGER_SERVICE_NAME,
        unit_dir / "default.target.wants" / LEGACY_SWAP_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / MANAGER_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / SWAP_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / LEGACY_MANAGER_SERVICE_NAME,
        unit_dir / "multi-user.target.wants" / LEGACY_SWAP_SERVICE_NAME,
    ):
        _remove_path(path, dry_run)

    # 4. Reload daemon
    reload_cmd = base_systemctl + ["daemon-reload"]
    if dry_run:
        print(f"[dry-run] would run {' '.join(reload_cmd)}")
    else:
        _run(reload_cmd, check=False)


def uninstall_stack(args: argparse.Namespace) -> int:
    # 1. Detect mode more robustly
    resolved_mode = args.mode
    if not resolved_mode:
        resolved_mode = detect_existing_mode()

    if not resolved_mode:
        # Check systemctl for any Heimdall or legacy services
        for mode in ("user", "system"):
            base_cmd = ["systemctl"]
            if mode == "user":
                base_cmd.append("--user")
            try:
                result = subprocess.run(
                    base_cmd + ["list-units", "--all", "--full", "--no-legend", "*heimdall*"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout.strip():
                    resolved_mode = mode
                    break
            except Exception:
                continue

    if not resolved_mode:
        resolved_mode = "system" if (Path("/etc/heimdall-gateway").exists() or Path("/etc/llamacpp-superserver").exists()) else "user"

    layout = choose_layout(resolved_mode, args.public_host, args.public_port, args=args)
    uninstall_systemd_units(layout, args.dry_run)

    remove_models = not args.keep_models
    if not args.keep_models and not args.dry_run:
        remove_models = prompt_bool("Remove downloaded models too?", default=True)

    remove_binaries = True
    if not args.dry_run:
        remove_binaries = prompt_bool("Remove compiled binaries and installation root?", default=True)

    targets = [
        layout.run_dir,
    ]

    if remove_binaries:
        targets.extend([
            layout.install_root,
            layout.bin_dir / MANAGER_WRAPPER_NAME,
            layout.bin_dir / SWAP_WRAPPER_NAME,
            layout.bin_dir / CLI_COMMAND,
            layout.bin_dir / LEGACY_CLI_COMMAND,
            # Ensure older wrapper names are also cleaned up
            layout.bin_dir / "llamacpp-manager-start",
            layout.bin_dir / "llamaswap-start",
            layout.bin_dir / "llamacpp-superserver",
            layout.bin_dir / "llamacpp-server",
            layout.bin_dir / "llamacpp-stack-install",
            layout.bin_dir / "llamacpp-stack-uninstall",
        ])

    if not args.keep_models:
        targets.append(layout.config_dir)
        targets.append(layout.state_dir)
        targets.append(layout.models_dir)
    else:
        print(f"Keeping configuration ({layout.config_dir}), catalog ({layout.state_dir}) and models ({layout.models_dir})")

    for target in targets:
        _remove_path(target, args.dry_run)

    legacy_targets = [
        Path("/opt/llm"),
        Path("/etc/llamacpp"),
        Path("/var/lib/llamacpp"),
        Path("/run/llamacpp"),
        Path.home() / ".local/opt/llamacpp-stack",
        Path.home() / ".config/llamacpp",
        Path.home() / ".local/state/llamacpp",
        Path.home() / ".local/run/llamacpp",
    ]
    if not args.keep_models:
        legacy_targets.extend([
            Path("/etc/llamacpp-superserver"),
            Path("/var/lib/llamacpp-superserver"),
            Path("/run/llamacpp-superserver"),
            Path("/opt/llamacpp-superserver"),
            Path.home() / ".local/opt/llamacpp-superserver",
            Path.home() / ".local/state/llamacpp-superserver",
            Path.home() / ".config/llamacpp-superserver",
            Path.home() / ".local/run/llamacpp-superserver",
        ])
    for target in legacy_targets:
        _remove_path(target, args.dry_run)

    print(f"Heimdall Gateway ({resolved_mode} mode) uninstalled.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uninstall Heimdall Gateway.")
    parser.add_argument("--mode", choices=("system", "user"))
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--public-port", type=int)
    parser.add_argument("--keep-models", action="store_true", help="Keep the models directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return uninstall_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
