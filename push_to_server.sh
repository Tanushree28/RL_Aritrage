#!/bin/bash
# Push current code to DigitalOcean server and restart everything.

# Auto-detected server details
SERVER_IP="68.183.174.225"
SERVER_USER="root"
REMOTE_DIR="kalshi-arb" # Exact path: ~/kalshi-arb

echo "🚀 Deploying to $SERVER_USER@$SERVER_IP:~/$REMOTE_DIR..."

# Sync Code

echo "🚀 Deploying to $SERVER_USER@$SERVER_IP..."

# 1. Sync Code
# Excludes data/logs/target to save bandwidth and protect production data
echo "📦 Syncing files..."
rsync -avz --exclude 'target' \
           --exclude 'data' \
           --exclude 'logs' \
           --exclude '.git' \
           --exclude '.DS_Store' \
           --exclude '__pycache__' \
           --exclude '.env' \
           ./ "$SERVER_USER@$SERVER_IP:$REMOTE_DIR/"

# 2. Build & Restart on Server
echo "🔨 Building and Restarting on Remote..."
ssh "$SERVER_USER@$SERVER_IP" << EOF
    # Ensure cargo is in path
    [ -f ~/.cargo/env ] && source ~/.cargo/env

    cd $REMOTE_DIR
    
    # Update permissions just in case
    chmod +x *.sh

    echo "Building release..."
    cargo build --release

    # Ensure logs directory exists (critical!)
    mkdir -p logs

    echo "Restarting bots..."
    # start_everything.sh handles stopping old bots and starting dashboard
    ./start_everything.sh
EOF

echo ""
echo "✅ Deployment Complete!"
echo "Dashboard should be available at: http://$SERVER_IP:8501"
echo "Monitor logs with: ssh $SERVER_USER@$SERVER_IP 'tail -f ~/kalshi-arb/logs/core-*.log'"
