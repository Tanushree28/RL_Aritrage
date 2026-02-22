#!/bin/bash
# Start ALL Bots (Core ML + Bucket 9) for all 4 assets (8 total processes)

# Load environment variables safely
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export STARTING_CAPITAL=10000
export BET_SIZE=100

echo "=== Stopping ANY existing bots ==="
pkill -f "kalshi-arb" || true
sleep 2

# === Start Dashboard ===
if ! lsof -i :8501 > /dev/null; then
    echo "Starting Dashboard..."
    nohup streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 &
fi

echo "=== Launching 4 CORE ML Bots ==="
./start_all_assets.sh

echo "=== Launching 4 BUCKET 9 Bots ==="
./start_bucket9_all.sh

echo ""
echo "✅ All 8 Bots Launched!"
echo "---------------------------------------------------"
echo "Monitor Core Logs:    tail -f logs/core-*.log"
echo "Monitor Bucket9 Logs: tail -f logs/bucket9-*.log"
echo "Monitor EVERYTHING:   tail -f logs/*.log"
echo "---------------------------------------------------"
