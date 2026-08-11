#!/bin/bash
# Sync local changes to a configured Heimdall Gateway host and restart its services.
REMOTE_HOST="${HEIMDALL_GATEWAY_REMOTE_HOST:?Set HEIMDALL_GATEWAY_REMOTE_HOST before using this script}"
REMOTE_USER="${HEIMDALL_GATEWAY_REMOTE_USER:-${USER:-gateway}}"
REMOTE_PATH="${HEIMDALL_GATEWAY_REMOTE_PATH:?Set HEIMDALL_GATEWAY_REMOTE_PATH before using this script}"

echo "🚀 Syncing code to $REMOTE_USER@$REMOTE_HOST..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude 'node_modules' --exclude '*.log' ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Sync complete. Restarting remote Heimdall Gateway services..."
    ssh "$REMOTE_USER@$REMOTE_HOST" "systemctl --user restart heimdall-gateway-manager heimdall-gateway-router"
    echo "✨ Done! Remote is now running the latest code."
else
    echo "❌ Sync failed. Check your connection or SSH keys."
    exit 1
fi
