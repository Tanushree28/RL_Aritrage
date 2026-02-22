# Kalshi Bot Command Reference

## 🚀 1. Start & Stop

### Start Everything (Recommended)

Cleans up old processes and launches all 8 bots (4 Core + 4 Bucket9) in the background.

```bash
./start_everything.sh
```

### Emergency Stop

Kills all running instances of the bot immediately.

```bash
pkill -f "kalshi-arb"
```

---

## 📜 2. Monitoring Logs

### Monitor EVERYTHING (The "Matrix" View)

Watch logs for ALL 8 bots simultaneously.

```bash
tail -f logs/*.log
```

### Monitor by Group

**All Core Bots (ML Strategy):**

```bash
tail -f logs/core-*.log
```

**All Bucket 9 Bots (Deep OTM Strategy):**

```bash
tail -f logs/bucket9-*.log
```

### Monitor by Asset (Core + Bucket 9 combined)

**Bitcoin (BTC):**

```bash
tail -f logs/*btc*.log
```

**Ethereum (ETH):**

```bash
tail -f logs/*eth*.log
```

**Solana (SOL):**

```bash
tail -f logs/*sol*.log
```

**Ripple (XRP):**

```bash
tail -f logs/*xrp*.log
```

### Monitor Specific Instances

```bash
tail -f logs/core-btc.log       # Core BTC only
tail -f logs/bucket9-btc.log    # Bucket 9 BTC only
```

---

## 📊 3. Verifying Data & Prices

### Verify Bitstamp Connection

Check if the Bitstamp WebSocket is connected and receiving data for a specific asset.

```bash
grep "Bitstamp" logs/core-btc.log | tail -n 10
```

### Check CSV Recordings (Price Feeds)

Verify that CSVs are being written and contain non-zero prices for Bitstamp (`bitstamp_price`).

**General Command (Last 5 rows of latest file):**

```bash
ls -t data/btc/recordings/$(date +%Y-%m-%d)/*.csv | head -n 1 | xargs tail -n 5
```

**Check other assets:**
Replace `btc` with `eth`, `sol`, or `xrp` in the path:

```bash
ls -t data/eth/recordings/$(date +%Y-%m-%d)/*.csv | head -n 1 | xargs tail -n 5
```

### Monitor Active CSV Updates (Real-time)

See which recording files are being updated right now (timestamps changing).

```bash
while true; do clear; ls -lt data/*/recordings/$(date +%Y-%m-%d)/*.csv | head -n 10; sleep 1; done
```

### Check Executed Trades (SQLite)

See the 5 most recent trades placed by the bots.

**Core BTC Trades:**

```bash
sqlite3 data/btc/trades.db "SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;"
```

**Bucket 9 BTC Trades:**

```bash
sqlite3 data/bucket9-btc/trades.db "SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;"
```

---

## 🛠 4. Maintenance & Debugging

### Run Interactive Mode (Single Bot Debug)

Run a single bot instance in the foreground to see immediate stdout/stderr.

```bash
./run_interactive.sh
```

### Force Rebuild & Restart

If you changed code and need to ensure the binary is updated:

```bash
# 1. Clean old binaries
cargo clean

# 2. Build release version (optimized)
cargo build --release

# 3. Kill old bots
pkill -f "kalshi-arb"

# 4. Start fresh
./start_everything.sh
```

### Check Process Status

Verify which bots are actually running:

```bash
ps aux | grep kalshi-arb | grep -v grep
```
