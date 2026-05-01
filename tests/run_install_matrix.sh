#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
LOGDIR="${ROOT_DIR}/build_test_outputs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOGDIR"

run_case() {
  name="$1"
  shift
  printf "=== CASE: %s ===\n" "$name" | tee -a "$LOGDIR/summary.log"
  echo "=== CASE: $name ===" > "$LOGDIR/$name.log"
  if "$@" >> "$LOGDIR/$name.log" 2>&1; then
    echo "[OK] $name" | tee -a "$LOGDIR/summary.log"
    echo "0" > "$LOGDIR/$name.exit"
  else
    ec=$?
    echo "[FAIL] $name (exit $ec)" | tee -a "$LOGDIR/summary.log"
    echo "$ec" > "$LOGDIR/$name.exit"
  fi
}

# Prepare environment
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install pytest
# Don't install package via pip (pyproject requires >=3.12). Run modules from source using PYTHONPATH.
if [ -f requirements.txt ]; then
  pip install -r requirements.txt || true
fi

# Run unit tests
PYTHONPATH="${ROOT_DIR}" run_case "pytest_all" python -m pytest -q || true

mkdir -p /workspace/test_models

BASE_CMD=(python -m llamacpp_stack.install --mode user --models-dir /workspace/test_models --idle-ttl 300)

# 1. Test llama-cpp-mode variations (dry-run)
for mode in native prebuilt source; do
  run_case "install_mode_${mode}_dryrun" "${BASE_CMD[@]}" --llama-cpp-mode "$mode" --dry-run || true
done

# 2. Toggle boolean options (dry-run)
run_case "no_prefer_source_cuda" "${BASE_CMD[@]}" --llama-cpp-mode source --no-prefer-source-cuda --dry-run || true
run_case "no_prefer_binary" "${BASE_CMD[@]}" --llama-cpp-mode source --no-prefer-binary --dry-run || true
run_case "no_install_services" "${BASE_CMD[@]}" --llama-cpp-mode source --no-install-services --dry-run || true
run_case "update_binaries_true" "${BASE_CMD[@]}" --llama-cpp-mode source --update-binaries --dry-run || true
run_case "update_binaries_false" "${BASE_CMD[@]}" --llama-cpp-mode source --no-update-binaries --dry-run || true
run_case "migrate_model_ids_true" "${BASE_CMD[@]}" --llama-cpp-mode source --migrate-model-ids --dry-run || true
run_case "migrate_model_ids_false" "${BASE_CMD[@]}" --llama-cpp-mode source --no-migrate-model-ids --dry-run || true
run_case "enable_tls" "${BASE_CMD[@]}" --llama-cpp-mode source --enable-tls --dry-run || true
run_case "public_host_0.0.0.0" "${BASE_CMD[@]}" --llama-cpp-mode source --public-host 0.0.0.0 --public-port 11500 --dry-run || true
run_case "mode_system_dryrun" python -m llamacpp_stack.install --mode system --llama-cpp-mode source --models-dir /workspace/test_models --idle-ttl 300 --dry-run || true

# 3. Speculative behavior: when no server config and with existing server config
STATE_DIR="$HOME/.local/state/llamacpp-superserver"
CONFIG_DIR="$HOME/.config/llamacpp-superserver"
rm -rf "$STATE_DIR" || true
mkdir -p "$STATE_DIR"
cat > "$STATE_DIR/catalog.json" <<'JSON'
[
  {"model_id":"m1","local_path":"/workspace/test_models/m1.gguf"}
]
JSON
run_case "speculative_injection_when_no_server_config" "${BASE_CMD[@]}" --llama-cpp-mode source --dry-run || true

# Now create an existing server config to ensure speculative defaults are NOT injected
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/conf.json" <<'JSON'
{"_meta":{},"models":{},"llama_server_defaults":{"ctx_size":1234}}
JSON
run_case "speculative_no_injection_when_existing_defaults" "${BASE_CMD[@]}" --llama-cpp-mode source --dry-run || true

# 4. One real build-from-source using bundle wrapper (may take long)
# This will run the bundle bootstrap which installs uv and necessary bootstrap deps.
# Inside containers without a user systemd bus, installing/enabling services will fail.
# Pass --no-install-services to avoid running `systemctl --user` in the build.
run_case "build_from_source_real" ./llamacpp_stack/bundle/install_llamacpp_stack.sh --mode user --llama-cpp-mode source --models-dir /workspace/test_models --no-install-services || true

echo "Logs saved to: $LOGDIR"

echo "Completed. Summary:"
cat "$LOGDIR/summary.log" || true

exit 0
