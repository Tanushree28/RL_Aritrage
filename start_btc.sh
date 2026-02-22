#!/bin/bash
# BTC Trading Bot Startup Script

cd "$(dirname "$0")"

# Use .env.btc as the env file (dotenvy in Rust will load it)
cp .env.btc .env

# Start the BTC bot in the background
nohup ./target/release/kalshi-arb-btc > /tmp/bot_btc.log 2>&1 &

echo "BTC bot started with PID: $!"
echo "Logs: tail -f /tmp/bot_btc.log"
