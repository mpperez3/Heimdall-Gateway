#!/bin/bash
# Sync local changes to the remote server and restart the service
REMOTE_IP="192.168.110.50"
REMOTE_USER="martin" # Assuming same user as local, adjust if needed
REMOTE_PATH="/home/martin/Developments/PycharmProjects/OpenCodeAutoModelDiscover/projects/llamacpp-stack"

echo "🚀 Syncing code to $REMOTE_IP..."
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude 'node_modules' --exclude '*.log' ./ $REMOTE_USER@$REMOTE_IP:$REMOTE_PATH

if [ $? -eq 0 ]; then
    echo "✅ Sync complete. Restarting remote manager..."
    ssh $REMOTE_USER@$REMOTE_IP "systemctl --user restart llamacpp-superserver-manager"
    echo "✨ Done! Remote is now running the latest code."
else
    echo "❌ Sync failed. Check your connection or SSH keys."
    exit 1
fi
