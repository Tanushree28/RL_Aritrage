#!/bin/bash
# SOL Trading Bot Startup Script

cd "$(dirname "$0")"

# Use .env.sol as the env file (dotenvy in Rust will load it)
cp .env.sol .env

# Start the SOL bot in the background
nohup ./target/release/kalshi-arb-sol > /tmp/bot_sol.log 2>&1 &

echo "SOL bot started with PID: $!"
echo "Logs: tail -f /tmp/bot_sol.log"
