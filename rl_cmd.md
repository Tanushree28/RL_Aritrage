# RL / Q-Learning / Continuous Learning Command Reference
> 🖥️ = Run inside SSH on server | 💻 = Run on your local Mac

---

## 🧠 How It All Works (Quick Summary)

```
Live Bots trade → experiences saved to experiences.db
        ↓ every 2 hours
Continuous Learner reads experiences → trains Q-table
        ↓ if Sharpe > MIN_SHARPE (0.3)
New model promoted → model/{asset}/rl_policy.json updated
        ↓
Ensemble bots + Optimal bot use this policy to trade smarter
```

**Key files:**
- `rl_strategy.py` — Q-learning logic, state space, reward clipping
- `continuous_learner.py` — training loop, promotion gate
- `model/{asset}/rl_policy.json` — current production policy (what bots use)
- `model_registry/{asset}/versions/` — all trained versions ever

---

## 📊 1. Check Model Status

### See All Current Production Models + Top Versions
```bash
# 🖥️
cd /root/kalshi-arb && python3 check_models.py
```
Shows: states, last updated, Sharpe, PnL, epsilon for each asset.

### Check Which Policy Optimal Bot is Using
```bash
# 🖥️
python3 -c "
import json, datetime
with open('/root/kalshi-arb/model/btc/rl_policy.json') as f:
    d = json.load(f)
utc = datetime.datetime.fromtimestamp(d['timestamp'], tz=datetime.timezone.utc)
print('Policy trained (UTC):', utc.strftime('%b %d %I:%M %p UTC'))
print('States in Q-table:', len(d['q_table']))
print('Epsilon at training:', d.get('epsilon', 'unknown'))
"
```

### Check Q-table Key Format (should be 5D)
```bash
# 🖥️
python3 -c "
import json
with open('/root/kalshi-arb/model/btc/rl_policy.json') as f:
    d = json.load(f)
keys = list(d['q_table'].keys())[:3]
print('Sample keys:', keys)
print('Dimensions:', keys[0].count(',') + 1, '(should be 5)')
"
```

### See All Trained Versions for an Asset
```bash
# 🖥️
find /root/kalshi-arb/model_registry/btc/versions -name "metadata.json" | sort -r | head -5 | \
xargs -I{} python3 -c "
import json, sys
with open('{}') as f: d = json.load(f)
e = d.get('evaluation', {})
print(d.get('version','?'), 'Sharpe:', round(e.get('sharpe_ratio',0),2), 'PnL:', e.get('total_pnl',0), 'WR:', e.get('win_rate',0))
"
```

---

## 🔄 2. Continuous Learner

### Start All 4 Learners
```bash
# 🖥️
cd /root/kalshi-arb
pkill -f "continuous_learner.py" || true
sleep 2
for asset in btc eth sol xrp; do
    PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset $asset --loop --interval 120 > logs/continuous_learner-$asset.log 2>&1 &
    echo "$asset learner PID: $!"
done
```

### Watch a Learner Train in Real Time
```bash
# 🖥️
tail -f logs/continuous_learner-btc.log
tail -f logs/continuous_learner-eth.log
tail -f logs/continuous_learner-sol.log
tail -f logs/continuous_learner-xrp.log
```

### Watch Just Promotion Decisions
```bash
# 🖥️
tail -f logs/continuous_learner-btc.log | grep -i "sharpe\|promot\|not promot\|beat"
```

### Run One Manual Training Cycle (Foreground — see full output)
```bash
# 🖥️
cd /root/kalshi-arb && python3 continuous_learner.py --asset btc 2>&1
```

### Run Fresh Training (Wipes Q-table, Starts from Zero)
```bash
# 🖥️
cd /root/kalshi-arb && python3 continuous_learner.py --asset btc --fresh 2>&1
```
> ⚠️ Only use fresh if the model is badly corrupted. Normally use warm-start (default).

### Check How Many Learners Are Running (should be exactly 4)
```bash
# 🖥️
ps aux | grep continuous_learner | grep -v grep | wc -l
ps aux | grep continuous_learner | grep -v grep
```

### Fix: Too Many Duplicate Learners Running (memory issue)
```bash
# 🖥️
pkill -f "continuous_learner.py" || true
sleep 3
ps aux | grep continuous_learner | grep -v grep   # should be empty
# Then restart 4 clean ones ↑
```

---

## 🎛️ 3. Tuning Parameters

### Current Key Settings (rl_strategy.py)
| Parameter | Value | What It Does |
|-----------|-------|--------------|
| `REWARD_CLIP` | 50.0 | Caps reward at ±$50 — prevents Q-value explosion |
| `EPSILON_START` | 1.0 | Start fully random (new model) |
| `EPSILON_MIN` | 0.01 | Never go below 1% random |
| `EPSILON_WARMSTART` | 0.15 | When loading existing model, start at 15% random |
| `EPSILON_DECAY` | 0.99 | Reduce randomness by 1% each episode |
| `MIN_SHARPE` | 0.3 | Min Sharpe to promote new model to production |

### Fix: Model Not Getting Promoted (Sharpe too low)
> Full step-by-step — do these IN ORDER

```bash
# 🖥️ STEP 1 — Lower the threshold
grep "MIN_SHARPE" /root/kalshi-arb/continuous_learner.py      # check current value
sed -i 's/MIN_SHARPE = 1.0/MIN_SHARPE = 0.3/' /root/kalshi-arb/continuous_learner.py
grep "MIN_SHARPE" /root/kalshi-arb/continuous_learner.py      # verify → should show 0.3

# 🖥️ STEP 2 — Restart learners to pick up the change
pkill -f "continuous_learner.py" && sleep 2
cd /root/kalshi-arb
for asset in btc eth sol xrp; do
    PYTHONUNBUFFERED=1 nohup python3 continuous_learner.py --asset $asset --loop --interval 120 > logs/continuous_learner-$asset.log 2>&1 &
    echo "$asset learner PID: $!"
done

# 🖥️ STEP 3 — Watch for promotion (happens in ~2 minutes)
tail -f logs/continuous_learner-btc.log | grep -i "sharpe\|promot"
# Wait until you see: ✅ MODEL PROMOTED vDATE_TIME TO PRODUCTION

# 🖥️ STEP 4 — Restart optimal bot to load the new policy
pkill -f "kalshi-arb-optimal" && sleep 1
KALSHI_ASSET=optimal-btc nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
echo "Optimal BTC PID: $!"

# 🖥️ STEP 5 — Verify new policy loaded
head -3 logs/optimal-btc.log
# Should show new timestamp in: [kalshi-arb-optimal] Policy: model/btc/rl_policy.json
```

### Fix: Q-Value Explosion (Max Q-Value too high on dashboard)
```bash
# 🖥️
# Verify reward clipping is active
grep "REWARD_CLIP\|clipped_reward" /root/kalshi-arb/rl_strategy.py
# Should show: REWARD_CLIP = 50.0 and clipped_reward = max(-REWARD_CLIP, ...)
```

### Check Current Epsilon in Production Model
```bash
# 🖥️
python3 -c "
import json
with open('/root/kalshi-arb/model/btc/rl_policy.json') as f: d = json.load(f)
print('Epsilon:', d.get('epsilon', 'not saved'))
"
```

### Change Live Epsilon (in .env — takes effect on next bot restart)
```bash
# 🖥️
grep "EPSILON" /root/kalshi-arb/.env        # check current
sed -i 's/EPSILON=.*/EPSILON=0.05/' /root/kalshi-arb/.env
grep "EPSILON" /root/kalshi-arb/.env        # verify
```

---

## 🏆 4. Promoting Models

### Auto-Promotion (happens every training cycle if Sharpe > MIN_SHARPE)
The learner promotes automatically. Check the log to see if it happened:
```bash
# 🖥️
grep "PROMOTED\|NOT PROMOT" logs/continuous_learner-btc.log | tail -5
```

### Manually Force-Promote a Specific Version
```bash
# 🖥️
# 1. Find available versions
find /root/kalshi-arb/model_registry/btc/versions -name "rl_policy.json" | sort -r | head -5

# 2. Copy chosen version to production
cp model_registry/btc/versions/<VERSION>/rl_policy.json model/btc/rl_policy.json

# 3. Update timestamp so dashboard shows correct time
python3 -c "
import json, time
with open('model/btc/rl_policy.json') as f: d = json.load(f)
d['timestamp'] = time.time()
with open('model/btc/rl_policy.json', 'w') as f: json.dump(d, f)
print('Promoted at:', time.strftime('%I:%M %p UTC'))
"

# 4. Restart optimal bot to load new policy
pkill -f "kalshi-arb-optimal" && sleep 1
KALSHI_ASSET=optimal-btc nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
```

---

## ⭐ 5. Optimal Policy Bot

### What it is
Paper-only bot that runs a **frozen snapshot** of the best Q-table.
- Loads policy once at startup — does NOT update while running
- To use latest model: restart the bot
- To use specific version: set `OPTIMAL_POLICY_PATH`

### Start with Latest Production Model
```bash
# 🖥️
cd /root/kalshi-arb
pkill -f "kalshi-arb-optimal" || true && sleep 1
KALSHI_ASSET=optimal-btc nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
echo "Optimal BTC PID: $!"
```

### Start with Specific Frozen Version
```bash
# 🖥️
KALSHI_ASSET=optimal-btc \
OPTIMAL_POLICY_PATH=/root/kalshi-arb/model_registry/btc/versions/v20260409_013544/rl_policy.json \
nohup ./target/release/kalshi-arb-optimal > logs/optimal-btc.log 2>&1 &
```

### Watch Optimal Bot Trade Decisions
```bash
# 🖥️
tail -f logs/optimal-btc.log
tail -f logs/optimal-btc.log | grep -i "action\|entry\|settle\|buy\|hold"
```

### Check Optimal Bot Trade Results
```bash
# 🖥️
sqlite3 /root/kalshi-arb/data/optimal-btc/trades.db \
  "SELECT opened_at, side, entry_price, exit_price, pnl, settlement FROM trades ORDER BY rowid DESC LIMIT 10;"
```

---

## 🔍 6. Debugging the Q-Table

### Why is Optimal Bot Always HOLDing?
```bash
# 🖥️
# Check key format matches Python (must be 5D: "d,t,p,dir,mom")
python3 -c "
import json
with open('/root/kalshi-arb/model/btc/rl_policy.json') as f: d = json.load(f)
k = list(d['q_table'].keys())[0]
print('Key format:', k, '— dims:', k.count(',')+1, '(need 5)')
"
```

### Check Q-Value Distribution (is it exploded?)
```bash
# 🖥️
python3 -c "
import json
with open('/root/kalshi-arb/model/btc/rl_policy.json') as f: d = json.load(f)
all_q = [v for vals in d['q_table'].values() for v in vals]
print('Max Q:', round(max(all_q),2))
print('Min Q:', round(min(all_q),2))
print('Avg Q:', round(sum(all_q)/len(all_q),2))
print('Healthy range: -50 to +50')
"
```

### How Many States Has the Bot Explored?
```bash
# 🖥️
python3 -c "
import json
for asset in ['btc','eth','sol','xrp']:
    try:
        with open(f'/root/kalshi-arb/model/{asset}/rl_policy.json') as f: d = json.load(f)
        print(f'{asset.upper()}: {len(d[\"q_table\"])} states')
    except: print(f'{asset.upper()}: no model')
"
```

---

## 📋 7. State Space Reference

The Q-table key is **5 dimensions**: `"dist_bucket, time_bucket, price_bucket, direction, momentum"`

| Dimension | Values | Meaning |
|-----------|--------|---------|
| dist_bucket | 0-5 | How far spot price is from strike (0=very close, 5=far) |
| time_bucket | 0-10 | Time left before expiry (0=expired, 10=>300s) |
| price_bucket | 0-9 | YES contract price ($0.00-$1.00 in 10 bands) |
| direction | 0 or 1 | 0=below strike, 1=above strike |
| momentum | 0,1,2 | 0=falling, 1=neutral, 2=rising |

**Actions:** 0=HOLD, 1=BUY YES, 2=BUY NO

**Total possible states:** ~5,400 | **Currently explored:** ~778 (19.6%)
