#!/bin/bash
# Start Bucket 9 bots for BTC, ETH, SOL, XRP simultaneously

# Load environment variables from .env
if [ -f .env ]; then
  export $(cat .env | xargs)
fi
# Bucket 9 uses its own capital settings but often same env vars
export STARTING_CAPITAL=10000
export BET_SIZE=100

echo "Starting BTC Bucket 9 Bot..."
export KALSHI_ASSET=btc
nohup ./target/release/kalshi-arb-bucket9-btc > logs/bucket9-btc.log 2>&1 &
echo "Started BTC Bucket 9 (PID $!)"

echo "Starting ETH Bucket 9 Bot..."
export KALSHI_ASSET=eth
nohup ./target/release/kalshi-arb-bucket9-eth > logs/bucket9-eth.log 2>&1 &
echo "Started ETH Bucket 9 (PID $!)"

echo "Starting SOL Bucket 9 Bot..."
export KALSHI_ASSET=sol
nohup ./target/release/kalshi-arb-bucket9-sol > logs/bucket9-sol.log 2>&1 &
echo "Started SOL Bucket 9 (PID $!)"

echo "Starting XRP Bucket 9 Bot..."
export KALSHI_ASSET=xrp
nohup ./target/release/kalshi-arb-bucket9-xrp > logs/bucket9-xrp.log 2>&1 &
echo "Started XRP Bucket 9 (PID $!)"

echo "All Bucket 9 bots started in background. Monitor logs with:"
echo "tail -f logs/bucket9-*.log"
