# Kalshi Bot Command Reference
> 🖥️ = Run inside SSH on server | 💻 = Run on your local Mac

---

## 🚀 1. Start & Stop

### Start Everything (17 Bots + 4 Learners + Dashboards)
```bash
# 🖥️
cd /root/kalshi-arb && ./start_everything.sh
```

### Emergency Stop (All Bots + Learners)
```bash
# 🖥️
pkill -f "target/release/kalshi-arb" || true
pkill -f "continuous_learner.py" || true
```

> ⚠️ Never use `pkill -f "kalshi-arb"` — it also kills `cargo build`

### Stop Only Continuous Learners
```bash
# 🖥️
pkill -f "continuous_learner.py"
```

### Restart Continuous Learners (All 4 Assets)
```bash
# 🖥️
pkill -f "continuous_learner.py" || true
sleep 2
cd /root/kalshi-arb
for asset in btc eth sol xrp; do
    PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset $asset --loop --interval 120 > logs/continuous_learner-$asset.log 2>&1 &
    echo "$asset learner PID: $!"
done
```

### Start Optimal Policy Bot (Paper Mode, BTC)
```bash
# 🖥️
cd /root/kalshi-arb && KALSHI_ASSET=optimal-btc nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
echo "Optimal BTC PID: $!"
```

### Start With a Specific Frozen Policy Version
```bash
# 🖥️
KALSHI_ASSET=optimal-btc \
OPTIMAL_POLICY_PATH=model_registry/btc/versions/v20260409_013544/rl_policy.json \
nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
```

### Stop Optimal Bot
```bash
# 🖥️
pkill -f "kalshi-arb-optimal"
```

### Restart Dashboard Only
```bash
# 🖥️
cd /root/kalshi-arb
fuser -k 8501/tcp 2>/dev/null || true
sleep 2
nohup streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 &
```

---

## 🔨 2. Building

> ⚠️ Only needed when `.rs` Rust files change. Python/shell/JSON changes need NO rebuild.

### Build Everything (All Binaries) — ~10 min
```bash
# 🖥️ Kill bots first, then build
pkill -f "target/release/kalshi-arb" || true
pkill -f "continuous_learner.py" || true
sleep 2
cd /root/kalshi-arb && cargo build --release 2>&1 | tail -5
```

### Build Single Binary (Fast — ~1-2 min)
```bash
# 🖥️
cd /root/kalshi-arb && cargo build --release --bin kalshi-arb-optimal 2>&1 | tail -5
cd /root/kalshi-arb && cargo build --release --bin kalshi-arb-ensemble 2>&1 | tail -5
```

### Upload Changed Rust Files from Mac then Rebuild
```bash
# 💻 Upload files
scp ~/Desktop/kalshi-arb/src/optimal_executor.rs \
    ~/Desktop/kalshi-arb/src/ensemble_executor.rs \
    ~/Desktop/kalshi-arb/src/spread_compression.rs \
    root@68.183.174.225:/root/kalshi-arb/src/

# 🖥️ Build only what changed
cd /root/kalshi-arb && cargo build --release --bin kalshi-arb-optimal 2>&1 | tail -5
```

---

## 📜 3. Monitoring Logs

### All Bots at Once
```bash
# 🖥️
tail -f logs/*.log
```

### By Bot Type
```bash
# 🖥️
tail -f logs/core-*.log          # Core ML bots
tail -f logs/bucket9-*.log       # Bucket 9 bots
tail -f logs/ensemble-*.log      # Ensemble supervisor bots
tail -f logs/oracle-*.log        # Oracle Black-Scholes bots
tail -f logs/optimal-btc.log     # Optimal policy bot
tail -f logs/continuous_learner-*.log  # All 4 learners
```

### Watch Optimal Bot Decisions
```bash
# 🖥️
tail -f logs/optimal-btc.log | grep -i "action\|trade\|entry\|hold\|buy"
```

### Watch Ensemble Voting
```bash
# 🖥️
tail -f logs/ensemble-btc.log
```

### Watch Continuous Learner Training
```bash
# 🖥️
tail -f logs/continuous_learner-btc.log
tail -f logs/continuous_learner-btc.log | grep -i "sharpe\|promot\|beat\|score"
```

---

## 📊 4. Dashboards & URLs

| Dashboard | URL |
|-----------|-----|
| **Main Monitoring Dashboard** | http://68.183.174.225:8501 |
| **RL State Space Explorer** | http://68.183.174.225:8080/rl_dashboard.html |

---

## 🧠 5. RL Model Management

### Check All Model Performance
```bash
# 🖥️
cd /root/kalshi-arb && python3 check_models.py
```

### Check Which Policy Optimal Bot is Using + When It Was Trained
```bash
# 🖥️
python3 -c "
import json, datetime
with open('/root/kalshi-arb/model/btc/rl_policy.json') as f:
    d = json.load(f)
utc = datetime.datetime.fromtimestamp(d['timestamp'], tz=datetime.timezone.utc)
print('Policy trained (UTC):', utc.strftime('%b %d %I:%M %p UTC'))
print('States in Q-table:', len(d['q_table']))
print('Epsilon:', d.get('epsilon', 'unknown'))
"
```

### Run Manual Training Cycle (BTC)
```bash
# 🖥️
cd /root/kalshi-arb && python3 continuous_learner.py --asset btc 2>&1
```

### Run Fresh Training (Wipes Q-table, starts from zero)
```bash
# 🖥️
cd /root/kalshi-arb && python3 continuous_learner.py --asset btc --fresh 2>&1
```

### Fix: Model Not Promoting (Sharpe threshold too high)
```bash
# 🖥️ Lower MIN_SHARPE from 1.0 to 0.3
sed -i 's/MIN_SHARPE = 1.0/MIN_SHARPE = 0.3/' /root/kalshi-arb/continuous_learner.py
grep "MIN_SHARPE" /root/kalshi-arb/continuous_learner.py
# Then restart learners
pkill -f "continuous_learner.py" && sleep 2
for asset in btc eth sol xrp; do
    PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset $asset --loop --interval 120 > logs/continuous_learner-$asset.log 2>&1 &
done
```

### Manually Promote a Model to Production
```bash
# 🖥️ Find latest version
find model_registry/btc/versions -name "rl_policy.json" | sort -r | head -3

# Copy to production
cp model_registry/btc/versions/<VERSION>/rl_policy.json model/btc/rl_policy.json

# Update timestamp so dashboard shows new time
python3 -c "
import json, time
with open('model/btc/rl_policy.json') as f: d = json.load(f)
d['timestamp'] = time.time()
with open('model/btc/rl_policy.json', 'w') as f: json.dump(d, f)
print('Done:', time.strftime('%I:%M %p'))
"
```

---

## 📈 6. Trade Data

### Check Recent Trades by Bot
```bash
# 🖥️
sqlite3 data/btc/trades.db "SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;"
sqlite3 data/bucket9-btc/trades.db "SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;"
sqlite3 data/ensemble-btc/trades.db "SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;"
sqlite3 data/optimal-btc/trades.db "SELECT * FROM trades ORDER BY rowid DESC LIMIT 5;"
sqlite3 data/oracle-btc/trades.db "SELECT * FROM trades ORDER BY rowid DESC LIMIT 5;"
```

### Compare Ensemble vs Optimal
```bash
# 🖥️
echo "=== Ensemble ===" && sqlite3 data/ensemble-btc/trades.db "SELECT COUNT(*), AVG(pnl), SUM(pnl) FROM trades;"
echo "=== Optimal ===" && sqlite3 data/optimal-btc/trades.db "SELECT COUNT(*), AVG(pnl), SUM(pnl) FROM trades;"
```

---

## 🛠 7. Maintenance & Debugging

### Check All Running Processes
```bash
# 🖥️
ps aux | grep "target/release/kalshi-arb" | grep -v grep
ps aux | grep continuous_learner | grep -v grep
```

### Check Memory Usage
```bash
# 🖥️
free -h
ps aux --sort=-%mem | head -15
```

### Fix: Too Many Duplicate Learners (memory leak)
```bash
# 🖥️ Kill all, restart exactly 4
pkill -f "continuous_learner.py" || true
sleep 3
ps aux | grep continuous_learner | grep -v grep   # should show nothing
cd /root/kalshi-arb
for asset in btc eth sol xrp; do
    PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset $asset --loop --interval 120 > logs/continuous_learner-$asset.log 2>&1 &
done
free -h   # should show ~1GB free
```

### Fix: Dashboard Not Loading (port conflict)
```bash
# 🖥️
fuser -k 8501/tcp 2>/dev/null || true
sleep 2
cd /root/kalshi-arb && nohup streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 &
```

### Check Which Port Each Dashboard Is On
```bash
# 🖥️
ss -tlnp | grep -E ":8080|:8501"
```

### Upload Python/Shell Files (no rebuild needed)
```bash
# 💻
scp ~/Desktop/kalshi-arb/continuous_learner.py root@68.183.174.225:/root/kalshi-arb/continuous_learner.py
scp ~/Desktop/kalshi-arb/start_everything.sh root@68.183.174.225:/root/kalshi-arb/start_everything.sh
scp ~/Desktop/kalshi-arb/check_models.py root@68.183.174.225:/root/kalshi-arb/check_models.py
```
