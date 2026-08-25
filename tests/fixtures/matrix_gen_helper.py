
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
import yaml

# Mocking parts of cli.py to test render_llamaswap_config
sys.path.append('.')
from llamacpp_stack.cli import ManagedModel, render_llamaswap_config, _is_embedding_model, _is_small_model, _calculate_llama_swap_matrix

# Set file sizes manually for testing
def mock_stat(path):
    class Stat:
        st_size = 10 * 1024 * 1024 * 1024 # Default 10GB
        st_mode = 33188 # Regular file
    if "small" in str(path):
        Stat.st_size = 2 * 1024 * 1024 * 1024 # 2GB
    if "emb" in str(path):
        Stat.st_size = 500 * 1024 * 1024 # 0.5GB
    if str(path) in {".", ".."}:
        Stat.st_mode = 16877 # Directory
    return Stat()

import pathlib
pathlib.Path.stat = mock_stat

catalog = [
    ManagedModel(
        model_id="large-1",
        repo_id="repo/large-1",
        quant="Q4_K_M",
        filename="large-1.gguf",
        local_path="/models/large-1.gguf",
        n_gpu_layers=999
    ),
    ManagedModel(
        model_id="large-2",
        repo_id="repo/large-2",
        quant="Q4_K_M",
        filename="large-2.gguf",
        local_path="/models/large-2.gguf",
        n_gpu_layers=999
    ),
    ManagedModel(
        model_id="embedding-1",
        repo_id="repo/emb-1",
        quant=None,
        filename="emb-1.gguf",
        local_path="/models/emb-1.gguf",
        load_capabilities=["embedding"]
    ),
    ManagedModel(
        model_id="small-1",
        repo_id="repo/small-1",
        quant="Q4_K_M",
        filename="small-1.gguf",
        local_path="/models/small-1.gguf",
        n_gpu_layers=999
    )
]

class MockArgs:
    config = Path("test_matrix_config.yaml")
    llama_server = "/usr/bin/false"
    start_port = 18080

render_llamaswap_config(
    catalog,
    MockArgs.config,
    MockArgs.llama_server,
    MockArgs.start_port,
    replica_defaults={"enabled": True, "max": 2, "gpus_per_replica": 1}
)

with open("test_matrix_config.yaml", "r") as f:
    print(f.read())

