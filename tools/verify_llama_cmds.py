#!/usr/bin/env python3
"""Utility to render llama-server launch commands for models in test_catalog.json.
This script stubs heavy imports so it can run in minimal Python environments
inside the test Docker image.
"""
import sys
import types
from pathlib import Path
import json

# Stub out optional heavy deps used at module import time
sys.modules.setdefault('yaml', types.SimpleNamespace())
sys.modules.setdefault('requests', types.SimpleNamespace())
sys.modules.setdefault('huggingface_hub', types.SimpleNamespace(
    HfApi=lambda *a, **k: None,
    hf_hub_download=lambda *a, **k: None,
    snapshot_download=lambda *a, **k: None,
))

from llamacpp_stack.cli import ManagedModel, build_llama_server_command


def main():
    catalog_path = Path('tests/fixtures/test_catalog.json')
    if not catalog_path.exists():
        print('Missing tests/fixtures/test_catalog.json (run from repo root)', file=sys.stderr)
        sys.exit(2)
    models = json.loads(catalog_path.read_text(encoding='utf-8'))
    for m in models:
        model = ManagedModel(**m)
        cmd = build_llama_server_command(model, Path('/usr/local/bin/llama-server'), port='18090')
        print(model.model_id + ':', ' '.join(map(str, cmd)))


if __name__ == '__main__':
    main()
