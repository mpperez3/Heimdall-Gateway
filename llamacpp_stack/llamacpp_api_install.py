#!/usr/bin/env python3
"""Packaged entrypoint for the llama.cpp stack CLI."""

import sys
from pathlib import Path

# Ensure parent dir is in path
pkg_dir = Path(__file__).resolve().parent.parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from llamacpp_stack.cli import main


if __name__ == "__main__":
    raise SystemExit(main() or 0)
