#!/usr/bin/env python3
"""Packaged entrypoint for the llama.cpp stack CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main() or 0)
