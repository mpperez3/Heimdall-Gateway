#!/bin/bash
# Sync local changes to a configured LLM Server host and restart its services.
REMOTE_HOST="${LLM_SERVER_REMOTE_HOST:-${HEIMDALL_GATEWAY_REMOTE_HOST:?Set LLM_SERVER_REMOTE_HOST (or legacy HEIMDALL_GATEWAY_REMOTE_HOST) before using this script}}"
REMOTE_USER="${LLM_SERVER_REMOTE_USER:-${HEIMDALL_GATEWAY_REMOTE_USER:-${USER:-gateway}}}"
REMOTE_PATH="${LLM_SERVER_REMOTE_PATH:-${HEIMDALL_GATEWAY_REMOTE_PATH:?Set LLM_SERVER_REMOTE_PATH (or legacy HEIMDALL_GATEWAY_REMOTE_PATH) before using this script}}"

echo "🚀 Syncing code to $REMOTE_USER@$REMOTE_HOST..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude 'node_modules' --exclude '*.log' ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Sync complete. Restarting remote LLM Server services..."
    ssh "$REMOTE_USER@$REMOTE_HOST" "systemctl --user restart llm-server-manager llm-server-router"
    echo "✨ Done! Remote is now running the latest code."
else
    echo "❌ Sync failed. Check your connection or SSH keys."
    exit 1
fi
