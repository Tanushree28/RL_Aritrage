#!/bin/bash
# Bucket 9 BTC Bot (Demo Mode) Startup Script

cd "$(dirname "$0")"

# Load the demo configuration
# Ensure you have filled out .env.demo with your keys!
cp .env.demo .env

# Set asset explicitly to be safe (though default is btc)
export KALSHI_ASSET=btc

# Start the bot in the background
# binary name defined in Cargo.toml
nohup ./target/release/kalshi-arb-bucket9-btc > logs/bucket9-btc-demo.log 2>&1 &

echo "BTC Bucket 9 (Demo) bot started with PID: $!"
echo "Logs: tail -f logs/bucket9-btc-demo.log"
