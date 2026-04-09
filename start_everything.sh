#!/bin/bash
# Start ALL Bots (Core ML + Bucket 9 + Oracle + Ensemble + Optimal) for all 4 assets
# Plus 4 Continuous Learners and both dashboards

# Load environment variables safely
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export STARTING_CAPITAL=10000
export BET_SIZE=100

echo "=== Stopping ANY existing bots ==="
# Use specific path to avoid killing 'cargo build' processes
pkill -f "target/release/kalshi-arb" || true
pkill -f "continuous_learner.py" || true
pkill -f "python3 -m http.server" || true
sleep 2

# === Start Streamlit Dashboard ===
if ! lsof -i :8501 > /dev/null 2>&1; then
    echo "Starting Streamlit Dashboard on :8501..."
    nohup streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 &
fi

# === Start RL Dashboard ===
if ! lsof -i :8080 > /dev/null 2>&1; then
    echo "Starting RL Dashboard on :8080..."
    nohup python3 -m http.server 8080 > logs/rl-dashboard.log 2>&1 &
fi

echo ""
echo "=== Launching 4 CORE ML Bots ==="
for asset in btc eth sol xrp; do
    KALSHI_ASSET=$asset nohup ./target/release/kalshi-arb-$asset > logs/core-$asset.log 2>&1 &
    echo "  Core $asset started (PID: $!)"
done

echo "=== Launching 4 BUCKET 9 Bots ==="
for asset in btc eth sol xrp; do
    KALSHI_ASSET=$asset nohup ./target/release/kalshi-arb-bucket9-$asset > logs/bucket9-$asset.log 2>&1 &
    echo "  Bucket9 $asset started (PID: $!)"
done

echo "=== Launching 4 ORACLE Bots ==="
for asset in btc eth sol xrp; do
    KALSHI_ASSET=$asset nohup ./target/release/kalshi-arb-oracle > logs/oracle-$asset.log 2>&1 &
    echo "  Oracle $asset started (PID: $!)"
done

echo "=== Launching 4 ENSEMBLE Supervisor Bots ==="
for asset in btc eth sol xrp; do
    KALSHI_ASSET=ensemble-$asset nohup ./target/release/kalshi-arb-ensemble > logs/ensemble-$asset.log 2>&1 &
    echo "  Ensemble $asset started (PID: $!)"
done

echo "=== Launching Optimal Policy Bot (Paper — BTC only) ==="
KALSHI_ASSET=optimal-btc nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
echo "  Optimal BTC started (PID: $!)"

echo ""
echo "=== Launching 4 Continuous Learners ==="
for asset in btc eth sol xrp; do
    PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset $asset --loop --interval 120 > logs/continuous_learner-$asset.log 2>&1 &
    echo "  Learner $asset started (PID: $!)"
done

echo ""
echo "✅ All 17 Bots + 4 Learners + Dashboards Launched!"
echo "---------------------------------------------------"
echo "Monitor Core Logs:        tail -f logs/core-*.log"
echo "Monitor Bucket9 Logs:     tail -f logs/bucket9-*.log"
echo "Monitor Oracle Logs:      tail -f logs/oracle-*.log"
echo "Monitor Ensemble Logs:    tail -f logs/ensemble-*.log"
echo "Monitor Optimal Bot:      tail -f logs/optimal-btc.log"
echo "Monitor Learner Logs:     tail -f logs/continuous_learner-*.log"
echo "Monitor EVERYTHING:       tail -f logs/*.log"
echo "---------------------------------------------------"
echo "Streamlit Dashboard:      http://68.183.174.225:8501"
echo "RL State Space Explorer:  http://68.183.174.225:8080/rl_dashboard.html"
echo "---------------------------------------------------"
