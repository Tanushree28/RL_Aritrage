#!/bin/bash
# Start Core ML bots for BTC, ETH, SOL, XRP simultaneously

# Load environment variables from .env
if [ -f .env ]; then
  export $(cat .env | xargs)
fi
export STARTING_CAPITAL=10000
export BET_SIZE=100

# Kill old instances
pkill -f "kalshi-arb" || true

# Start Dashboard if not running
if ! lsof -i :8501 > /dev/null; then
    echo "Starting Dashboard..."
    nohup streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 &
fi

echo "Starting BTC Bot..."
export KALSHI_ASSET=btc
nohup ./target/release/kalshi-arb-btc > logs/core-btc.log 2>&1 &
echo "Started BTC (PID $!)"

echo "Starting ETH Bot..."
export KALSHI_ASSET=eth
nohup ./target/release/kalshi-arb-eth > logs/core-eth.log 2>&1 &
echo "Started ETH (PID $!)"

echo "Starting SOL Bot..."
export KALSHI_ASSET=sol
nohup ./target/release/kalshi-arb-sol > logs/core-sol.log 2>&1 &
echo "Started SOL (PID $!)"

echo "Starting XRP Bot..."
export KALSHI_ASSET=xrp
nohup ./target/release/kalshi-arb-xrp > logs/core-xrp.log 2>&1 &
echo "Started XRP (PID $!)"

echo "All bots started in background. Monitor logs with:"
echo "tail -f logs/core-btc.log"
echo "tail -f logs/core-eth.log"
