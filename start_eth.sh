#!/bin/bash
# ETH Trading Bot Startup Script

cd "$(dirname "$0")"

# Use .env.eth as the env file (dotenvy in Rust will load it)
cp .env.eth .env

# Start the ETH bot in the background
nohup ./target/release/kalshi-arb-eth > /tmp/bot_eth.log 2>&1 &

echo "ETH bot started with PID: $!"
echo "Logs: tail -f /tmp/bot_eth.log"
