from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .install import InstallLayout, choose_layout, detect_existing_mode, prompt_bool


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
    if layout.mode == "system":
        reload_cmd = ["systemctl", "daemon-reload"]
        stop_cmd = ["systemctl", "disable", "--now", "llamacpp-manager.service", "llamaswap.service"]
        unit_dir = Path("/etc/systemd/system")
    else:
        reload_cmd = ["systemctl", "--user", "daemon-reload"]
        stop_cmd = ["systemctl", "--user", "disable", "--now", "llamacpp-manager.service", "llamaswap.service"]
        unit_dir = Path.home() / ".config/systemd/user"

    if dry_run:
        print(f"[dry-run] would run {' '.join(stop_cmd)}")
    else:
        _run(stop_cmd, check=False)

    for path in (
        unit_dir / "llamacpp-manager.service",
        unit_dir / "llamaswap.service",
        unit_dir / "default.target.wants" / "llamacpp-manager.service",
        unit_dir / "default.target.wants" / "llamaswap.service",
        unit_dir / "multi-user.target.wants" / "llamacpp-manager.service",
        unit_dir / "multi-user.target.wants" / "llamaswap.service",
    ):
        _remove_path(path, dry_run)

    if dry_run:
        print(f"[dry-run] would run {' '.join(reload_cmd)}")
    else:
        _run(reload_cmd, check=False)


def uninstall_stack(args: argparse.Namespace) -> int:
    resolved_mode = args.mode or detect_existing_mode() or ("system" if Path("/etc/llamacpp").exists() else "user")
    layout = choose_layout(resolved_mode, args.public_host, args.public_port)
    uninstall_systemd_units(layout, args.dry_run)

    remove_models = not args.keep_models
    if not args.keep_models and not args.dry_run:
        remove_models = prompt_bool("Remove downloaded models too?", default=True)

    targets = [
        layout.config_dir,
        layout.state_dir,
        layout.run_dir,
        layout.install_root,
        layout.bin_dir / "llamacpp-manager-start",
        layout.bin_dir / "llamaswap-start",
        layout.bin_dir / "llamacpp-server",
    ]
    if remove_models:
        targets.append(layout.models_dir)

    for target in targets:
        _remove_path(target, args.dry_run)

    print("llamacpp stack uninstalled.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uninstall llama.cpp + llama-swap stack.")
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
