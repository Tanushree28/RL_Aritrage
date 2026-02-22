# Load environment variables from .env to ensure we use Production keys
if [ -f .env ]; then
  export $(cat .env | xargs)
fi
export STARTING_CAPITAL=10000
export BET_SIZE=100

# Kill old instances
pkill -f "kalshi-arb" || true

# Check dashboard
if lsof -i :8501 > /dev/null; then
    echo "Dashboard running."
else
    streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 &
fi


echo "Starting Core ML Bot (BTC) with exported keys..."
./target/release/kalshi-arb-btc


