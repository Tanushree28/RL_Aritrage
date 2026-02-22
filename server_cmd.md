# 🚀 Kalshi Bot Server Commands

**Server IP:** `68.183.174.225`
**User:** `root`
**Remote Path:** `~/kalshi-arb`

---

## 1. Monitor Logs (Live) 🟢

Check if the bots are running and what they are doing right now.

**Watch Core ML Bots (BTC, ETH, SOL, XRP):**

```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/core-*.log"
```

**Watch Bucket 9 Bots:**

```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/bucket9-*.log"
```

**Watch Dashboard Logs:**

```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/dashboard.log"
```

---

## 2. Sync Data (Server ➔ Mac) 📥

Download the recorded data and logs from the server to your local machine for analysis.

**Download EVERYTHING (Data + Logs):**

```bash
# Creates a 'server_backup' folder on your Mac
mkdir -p server_backup
rsync -avz --progress root@68.183.174.225:~/kalshi-arb/data/ ./server_backup/data/
rsync -avz --progress root@68.183.174.225:~/kalshi-arb/logs/ ./server_backup/logs/
```

**Download ONLY Logic/Trade Logs:**

```bash
rsync -avz --progress root@68.183.174.225:~/kalshi-arb/logs/ ./server_logs/
```

**Back up to OneDrive (Cloud) ☁️:**
Push data to OneDrive manually:

```bash
ssh root@68.183.174.225 "~/kalshi-arb/backup_to_onedrive.sh"
```

---

## 3. Dashboard Access 📊

View the live performance monitoring.

**URL:** [http://68.183.174.225:8501](http://68.183.174.225:8501)

**If Dashboard is Down/Stuck:**
Restart it remotely with this valid `nohup` command:

```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && pkill -f streamlit; nohup streamlit run monitoring/dashboard.py > logs/dashboard.log 2>&1 & disown"
```

---

## 4. Server Management 🛠️

**Restart ALL Bots (Clean Reset):**
This stops everything and restarts fresh (good if things look weird).

```bash
ssh root@68.183.174.225 "cd ~/kalshi-arb && ./start_everything.sh"
```

**Deploy Code Updates (Mac ➔ Server):**
If you changed code on your Mac, push it to the server:

```bash
./push_to_server.sh
```

**Check System Usage (CPU/RAM):**

```bash
ssh root@68.183.174.225 "htop"
```

_(Press `q` to exit htop)_

---

## 5. View Bucket 9 Data (Live) 📈

See the exact prices the Bucket 9 bot is recording right now.

**Check Latest BTC Data:**

```bash
ssh root@68.183.174.225 "tail -f ~/kalshi-arb/logs/bucket9-*.log"
```

### 11. Find the Exact File Currently Being Updated

Run this to see the name of the active `.csv` file in the bucket 9 data folder:

```bash
ssh root@68.183.174.225 "ls -t ~/kalshi-arb/data/bucket9-btc/recordings/\$(date +%Y-%m-%d)/*.csv | head -n 1"
```

This auto-updates your terminal as new prices arrive for Bucket9 BTC!

```bash
ssh root@68.183.174.225 "ls -t ~/kalshi-arb/data/bucket9-btc/recordings/\$(date +%Y-%m-%d)/*.csv | head -n 1 | xargs tail -f"
```

### 12. Analyze Bucket9 Opportunities Today

```bash
ssh root@68.183.174.225 "python3 ~/kalshi-arb/analyze_bucket9.py"
```

sh root@68.183.174.225 "ls -lt ~/kalshi-arb/data/bucket9-btc/recordings/2026-02-17/"

ssh root@68.183.174.225 "tail -n 5 ~/kalshi-arb/data/bucket9-btc/recordings/2026-02-17/KXBTC15M-26FEB171715-15.csv"

ssh root@68.183.174.225 'ls -t ~/kalshi-arb/data/bucket9-btc/recordings/$(date +%Y-%m-%d)/\*.csv | head -n 1 | xargs head -n 10'

026-02-19
No logs found for today.

Total 'Perfect' Bucket 9 Opportunities Today: 0
(base) tanushreenepal@Tanushrees-MacBook-Air kalshi-arb % mkdir -p data/bucket9-btc/recordings/$(date +%Y-%m-%d) && scp root@68.183.174.225:~/kalshi-arb/data/bucket9-btc/recordings/$(date +%Y-%m-%d)/\*.csv data/bucket9-btc/recordings/$(date +%Y-%m-%d)/ && python analyze_bucket9.py

root@68.183.174.225's password:
KXBTC15M-26FEB191515-15.csv 100% 85KB 449.9KB/s 00:00  
Scanning logs in: /Users/tanushreenepal/Desktop/kalshi-arb/data/bucket9-btc/recordings/2026-02-19
Found 109 matches in KXBTC15M-26FEB191515-15.csv!
...

Total 'Perfect' Bucket 9 Opportunities Today: 109
(base) tanushreenepal@Tanushrees-MacBook-Air kalshi-arb %
