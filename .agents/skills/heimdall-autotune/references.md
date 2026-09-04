Source: jr2804/llama-cpp-optimizer (MIT, https://github.com/jr2804/llama-cpp-optimizer.git)
Retrieved: 2026-09-02 via npx skills add jr2804/llama-cpp-optimizer
License: MIT
Adapted for Heimdall Gateway catalog.json loop, parallel/tensor_split auto-tune, and beellama generic engine fix.

Changes vs base:
- Added Heimdall-specific catalog.json -> config.yaml sync via sync_config_from_server_config_for_startup
- Added claim_loading generic detection for *llama-server* (beellama)
- Added validation evals: 2x50K parallel, 3x parallel, 1x sequential
- Added snapshot to configs/history with gitignore !configs/history/
