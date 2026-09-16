#!/usr/bin/env python3
"""Packaged entrypoint for LLM Server.

The public executable is intentionally a single command, ``llm-server``.
Runtime CLI commands are handled by ``llamacpp_stack.cli`` while install and
uninstall are dispatched here so legacy helper scripts are no longer needed.
"""

import sys
from pathlib import Path

# Ensure parent dir is in path
pkg_dir = Path(__file__).resolve().parent.parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))


def _is_deprecated_alias() -> bool:
    prog = Path(sys.argv[0]).name if sys.argv else ""
    return prog == "heimdall-gateway"


def main(argv: list[str] | None = None) -> int:
    if _is_deprecated_alias():
        print("WARNING: heimdall-gateway is deprecated, use llm-server", file=sys.stderr)
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "install":
        from llamacpp_stack.install import main as install_main

        return int(install_main(args[1:]) or 0)
    if args and args[0] == "uninstall":
        from llamacpp_stack.uninstall import main as uninstall_main

        return int(uninstall_main(args[1:]) or 0)

    from llamacpp_stack.cli import main as cli_main

    return int(cli_main(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
