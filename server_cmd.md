# 🚀 Kalshi Bot — Server Command Reference
> All commands below run **inside SSH** on the server unless marked 💻 (local Mac)

**Server IP:** `68.183.174.225`
**User:** `root`
**Path:** `/root/kalshi-arb/`

---

## 🌐 Dashboard URLs

| Dashboard | URL |
|-----------|-----|
| **RL State Space Explorer** | http://68.183.174.225:8080/rl_dashboard.html |
| **Main Monitoring Dashboard** | http://68.183.174.225:8501 |

---

## 1. Monitor Logs (Live) 🟢

**All 16 bots at once:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/*.log"
```

**Core ML Bots:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/core-*.log"
```

**Bucket 9 Bots:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/bucket9-*.log"
```

**Ensemble Supervisor Bots:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/ensemble-*.log"
```

**Continuous Learners:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/continuous_learner-*.log"
```

**Main Dashboard:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/dashboard.log"
```

---

## 2. Start / Stop Bots 🤖

**Start everything (all 16 bots + 4 learners):**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && ./start_everything.sh"
```

**Emergency stop ALL bots:**
```bash
ssh root@68.183.174.225 "pkill -f 'kalshi-arb'"
```

**Restart ensemble bots only:**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && pkill -f 'kalshi-arb-ensemble' && for asset in btc eth sol xrp; do KALSHI_ASSET=ensemble-\$asset nohup ./target/release/kalshi-arb-ensemble > logs/ensemble-\$asset.log 2>&1 & done"
```

**Restart continuous learners:**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && pkill -f continuous_learner && for asset in btc eth sol xrp; do PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset \$asset --loop --interval 120 > logs/continuous_learner-\$asset.log 2>&1 & done"
```

---

## 3. RL Model Management 🧠

**Check best models per asset:**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && python3 check_models.py"
```

**Run one manual training cycle (BTC):**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && python3 continuous_learner.py --asset btc 2>&1"
```

**Run fresh training (wipes Q-table):**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && python3 continuous_learner.py --asset btc --fresh 2>&1"
```

**Manually update dashboard with latest model:**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && cp model_registry/btc/versions/\$(ls model_registry/btc/versions | sort -r | head -1)/rl_policy.json model/btc/rl_policy.json && python3 -c \"import json,time; d=json.load(open('model/btc/rl_policy.json')); d['timestamp']=time.time(); json.dump(d,open('model/btc/rl_policy.json','w')); print('Done')\""
```

---

## 4. Sync Data (Server → Mac) 📥

**Download everything (data + logs):**
```bash
mkdir -p server_backup
rsync -avz --progress root@68.183.174.225:~/kalshi-arb/data/ ./server_backup/data/
rsync -avz --progress root@68.183.174.225:~/kalshi-arb/logs/ ./server_backup/logs/
```

**Download logs only:**
```bash
rsync -avz --progress root@68.183.174.225:~/kalshi-arb/logs/ ./server_logs/
```

**Back up to OneDrive:**
```bash
ssh root@68.183.174.225 "~/kalshi-arb/backup_to_onedrive.sh"
```

---

## 5. Deploy Code Updates (Mac → Server) 💻

**Push all code changes:**
```bash
./push_to_server.sh
```

**Push a specific file:**
```bash
scp /Users/tanushreenepal/Desktop/kalshi-arb/src/ensemble_executor.rs root@68.183.174.225:/root/kalshi-arb/src/ensemble_executor.rs
scp /Users/tanushreenepal/Desktop/kalshi-arb/continuous_learner.py root@68.183.174.225:/root/kalshi-arb/continuous_learner.py
scp /Users/tanushreenepal/Desktop/kalshi-arb/check_models.py root@68.183.174.225:/root/kalshi-arb/check_models.py
```

**Rebuild after code change:**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && cargo build --release --bin kalshi-arb-ensemble 2>&1 | tail -5"
```

---

## 6. Check Trade Data 📈

**Recent trades by bot:**
```bash
ssh root@68.183.174.225 "sqlite3 ~/kalshi-arb/data/btc/trades.db 'SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;'"
ssh root@68.183.174.225 "sqlite3 ~/kalshi-arb/data/bucket9-btc/trades.db 'SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;'"
ssh root@68.183.174.225 "sqlite3 ~/kalshi-arb/data/ensemble-btc/trades.db 'SELECT * FROM trades ORDER BY opened_at DESC LIMIT 5;'"
```

**Check ensemble paper trades:**
```bash
ssh root@68.183.174.225 "sqlite3 ~/kalshi-arb/data/ensemble-btc/trades.db 'SELECT * FROM trades ORDER BY id DESC LIMIT 10;'"
```

---

## 7. System Health 🛠️

**Check all running bots:**
```bash
ssh root@68.183.174.225 "ps aux | grep -E 'kalshi-arb|continuous_learner' | grep -v grep"
```

**Check CPU/RAM:**
```bash
ssh root@68.183.174.225 "htop"
```

**Check dashboard ports:**
```bash
ssh root@68.183.174.225 "ss -tlnp | grep -E ':80|:8050|:8080|:8501'"
```

**Restart RL State Space Explorer (port 8080):**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && pkill -f 'http.server' ; nohup python3 -m http.server 8080 > /dev/null 2>&1 &"
```

**Restart Main Dashboard (port 8501):**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && pkill -f streamlit; nohup streamlit run monitoring/dashboard.py --server.address 0.0.0.0 --server.port 8501 > logs/dashboard.log 2>&1 &"
```

---

## 8. Oracle Bot (Black-Scholes) ⚡

**Start all oracle bots:**
```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && for asset in btc eth sol xrp; do KALSHI_ASSET=\$asset PAPER_MODE=1 nohup ./target/release/kalshi-arb-oracle > logs/oracle-\$asset.log 2>&1 & done"
```

**Stop oracle bots:**
```bash
ssh root@68.183.174.225 "pkill -f kalshi-arb-oracle"
```

**Watch oracle logs:**
```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/oracle-*.log"
```

---

## 9. Bucket 9 Data Analysis 📊

**Watch live Bucket 9 data:**
```bash
ssh root@68.183.174.225 "ls -t ~/kalshi-arb/data/bucket9-btc/recordings/\$(date +%Y-%m-%d)/*.csv | head -n 1 | xargs tail -f"
```

**Analyze Bucket 9 opportunities:**
```bash
ssh root@68.183.174.225 "python3 ~/kalshi-arb/analyze_bucket9.py"
```
