# Kalshi Arbitrage Bot — Full System Explained
> Plain English + technical detail. Read top to bottom for the full picture.

---

## 🎯 What Are We Actually Doing?

Kalshi is a prediction market. You can bet YES or NO on questions like:
> "Will BTC be above $70,000 at 11:45 PM?"

If you bet YES and BTC is above $70k at expiry → you win $1 per contract.
If BTC is below → you get $0.

**Our bots watch these markets and try to find mispriced contracts — buy cheap, sell/settle expensive.**

There are 17 bots running simultaneously, each with a different strategy. They all watch the same markets but decide differently.

---

## 📚 Part 1: The Q-Table & Reinforcement Learning

### The Simple Version (Layman)

Imagine teaching a child to play chess. You don't give them a rulebook. Instead:
- They make a move
- They win or lose
- Over thousands of games, they learn: "When the board looks like THIS, move THAT way"

That's exactly what Q-learning does. The bot plays the market over and over, and slowly learns a table of:
> "When the situation looks like X → do action Y"

This table is the **Q-table**.

### The Technical Version

Q-learning is a type of **Reinforcement Learning (RL)**. The bot learns a function:

```
Q(state, action) = expected future reward if I take this action in this state
```

Every time a trade completes:
```python
new_q = old_q + learning_rate × (actual_reward - old_q)
```

Over thousands of trades, Q-values converge to reflect which actions are actually profitable.

### What is a "State"?

A state is a snapshot of the market at a moment in time. Our bot describes every market moment using **5 dimensions**:

| Dimension | What It Measures | Example Values |
|-----------|-----------------|----------------|
| **dist_bucket** | How far is BTC price from the strike? | 0=very close, 5=far away |
| **time_bucket** | How many seconds until expiry? | 0=expired, 10=>300s left |
| **price_bucket** | How much does the YES contract cost? | 0=$0-0.10, 9=$0.90-1.00 |
| **direction** | Is BTC above or below the strike? | 0=below, 1=above |
| **momentum** | Is BTC price rising or falling? | 0=falling, 1=neutral, 2=rising |

**Combined into a key:** `"2,4,6,1,1"` ← this is what the bot actually stores

**Total possible states:** ~5,400
**Currently explored:** ~778 (19.6%) — the bot hasn't seen every possible situation yet

### What is an "Action"?

For each state, the bot can take one of 3 actions:
- **0 = HOLD** — do nothing, wait
- **1 = BUY YES** — bet the price will be above strike at expiry
- **2 = BUY NO** — bet the price will be below strike at expiry

### The Q-Table (Visual)

Think of it like a giant spreadsheet:

```
State Key    | HOLD  | BUY YES | BUY NO  | → Best Action
-------------|-------|---------|---------|
"2,4,6,1,1" |  2.1  |  14.7   |  3.1    | → BUY YES ✅
"0,2,3,0,0" |  1.5  |  -2.3   |  8.9    | → BUY NO ✅
"3,8,5,1,2" |  0.0  |  0.0    |  0.0    | → unexplored
```

The bot always picks the action with the highest Q-value for the current state.

---

## 🔄 Part 2: Continuous Learning

### The Simple Version

The bot gets smarter every 2 hours. Like a student who reviews their mistakes after every exam.

### The Technical Flow

```
Every second:
  Live bots trade → result saved to experiences.db
                           ↓
Every 2 hours:
  continuous_learner.py reads all past trades
  Replays them through the Q-learning update formula
  Computes new Q-values for each state
                           ↓
  Evaluates new model: calculates Sharpe ratio
                           ↓
  If Sharpe > 0.3 (MIN_SHARPE threshold):
    → Promote: copy to model/btc/rl_policy.json
    → Dashboard updates
    → Next time optimal bot restarts, it uses new model
  Else:
    → Keep current model, discard new one
```

### What Does "Explore vs Exploit" Mean?

**Epsilon (ε)** controls this balance:
- **ε = 1.0** → 100% random trades (pure exploration — trying new things)
- **ε = 0.0** → 100% uses Q-table (pure exploitation — only does what it knows)
- **ε = 0.05** → 5% random, 95% Q-table ← what we run

Why not 0%? The bot might never have seen a new market situation. Keeping 5% random helps it discover new profitable states it hasn't tried.

### Why Did Q-Values Explode to 18,787?

This was a bug we fixed. Without reward clipping:
- Bot wins a big $989 trade
- That trade gets replayed 100× in training
- Each replay adds to Q(BUY YES) for that state
- After 100 replays: Q(BUY YES) = 18,787

The problem: **all Q-values inflate**, not just the good states. The bot can no longer tell which states are actually better — all Q-values are noise.

**Fix: Reward Clipping** (`REWARD_CLIP = 50.0`)
```python
clipped_reward = max(-50.0, min(50.0, actual_reward))
# A $989 win = treated as $50 win for learning purposes
# A $200 loss = treated as $50 loss for learning purposes
```

Q-values now stay in the ±50 range, which is meaningful and comparable.

---

## 📊 Part 3: The Sharpe Ratio

### The Simple Version (Layman)

Imagine two traders:
- **Trader A**: Makes $100 profit every single day. Never loses.
- **Trader B**: Averages $100/day but sometimes makes $500, sometimes loses $300.

Both average $100/day. But Trader A is **much better** because they're consistent.

The Sharpe Ratio measures exactly this: **profit relative to how unpredictable that profit is.**

```
Sharpe = Average PnL per trade ÷ Standard Deviation of PnL per trade
```

| Sharpe Value | Meaning |
|-------------|---------|
| < 0 | Losing money on average |
| 0 to 1 | Making money but very inconsistent |
| 1 to 3 | Good — solid risk-adjusted returns |
| > 3 | Excellent (or suspicious — might be overfitting) |
| 100 | **Fake** — Q-value explosion inflated the numbers |

### Why Did We See Sharpe = 100?

The old models (before reward clipping) had inflated Q-values. When we evaluated them, the simulation replayed their best trades over and over, making PnL look amazing. The Sharpe was capped at 100 in our code because the real values were in the thousands — completely meaningless.

After the fix: honest Sharpe values of 0.3 to 5.5 — real numbers we can actually compare.

### MIN_SHARPE Threshold

We only promote a new model if `new_sharpe > MIN_SHARPE (0.3)`. This prevents a randomly bad training cycle from replacing a good model.

---

## 🤖 Part 4: Bot Types — What Each One Does

### 4A. Core ML Bots (4 bots: BTC, ETH, SOL, XRP)

The original bots. Use the Q-table directly:
1. Every second: observe current market state
2. Convert to 5D state key: `"2,4,6,1,1"`
3. Look up Q-table → pick best action
4. If BUY YES/NO: place real order on Kalshi
5. Track result → save to experiences.db for future training

**These place REAL orders** (live money or paper depending on `PAPER_MODE`).

### 4B. Bucket 9 Bots (4 bots: BTC, ETH, SOL, XRP)

Same as Core ML but only trade when YES price is in the `$0.80-$0.90` range ("bucket 9"). Higher-confidence trades, fewer but more selective.

### 4C. Oracle Bots (4 bots: BTC, ETH, SOL, XRP)

Use Black-Scholes options pricing formula instead of Q-learning. Pure math — no machine learning. Calculates theoretical fair value and trades when market price deviates significantly.

### 4D. Ensemble Bots (4 bots: BTC, ETH, SOL, XRP)

**The most conservative bot.** Uses multiple "voters" — they all must agree before placing a trade.

```
Voter 1: RL Policy (Q-table) → says BUY YES
Voter 2: Actual Trades History → says BUY YES
Both agree → TRADE ✅

Voter 1: BUY YES
Voter 2: HOLD
Disagree → NO TRADE ❌
```

Also runs the **Spread Compression** strategy (see below) as a secondary strategy.

**These place REAL orders** when voters agree.

### 4E. Optimal Policy Bot (1 bot: BTC only)

**Paper trading only — never places real orders.**

The idea: take the best Q-table ever trained, freeze it, and run it in simulation to see how it performs with no further learning.

```
Best Q-table ever trained
        ↓
Frozen (locked — won't update)
        ↓
Optimal bot loads it once at startup
        ↓
Trades paper (simulated) using that frozen policy
        ↓
Results tracked in data/optimal-btc/trades.db
```

**Why is this useful?** It's a benchmark. If the frozen "best ever" policy loses money in paper trading, it tells us the Q-table quality is still poor and needs more training.

---

## 📋 Part 5: The Spread Compression Strategy

This runs inside the Ensemble bots as a secondary strategy, independent of Q-learning.

### The Logic

Kalshi contracts have a bid/ask spread. When the spread is unusually wide (>25%), it means the market is uncertain/illiquid.

Strategy:
1. Find a contract where YES is deep ITM (BTC clearly above strike) but spread is wide (>25%)
2. Buy at mid-price (between bid and ask)
3. Wait for spread to compress below 10% OR for contract to expire
4. Exit at profit as market becomes more efficient

**Example:**
```
BTC at $71,000 | Strike $70,000 (BTC clearly above)
YES bid: $0.55 | YES ask: $0.85 | Spread: 30% ← wide!
Mid-price = $0.70 → BUY at $0.70

2 hours later:
YES bid: $0.78 | YES ask: $0.84 | Spread: 6% ← compressed!
Exit at $0.81 → Profit: $0.11 per contract ✅
```

---

## 🗂️ Part 6: How the Policy File Works

### The rl_policy.json File

This is the "brain" of the RL bots. It's a JSON file stored at `model/btc/rl_policy.json`:

```json
{
  "timestamp": 1744174671.2,       ← when this model was trained
  "epsilon": 0.033,                 ← exploration rate at time of training
  "buckets": {
    "distance": [0.1, 0.2, 0.3, 0.4, 0.5],   ← thresholds for dist_bucket
    "time": [30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    "price": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "q_table": {
    "0,4,9,1,1": [2.1, 14.7, 3.1],   ← [HOLD, BUY_YES, BUY_NO] Q-values
    "0,3,3,0,2": [1.5, -2.3, 8.9],
    ...778 states total...
  }
}
```

### How a Bot Uses This File

```
Step 1: Load rl_policy.json at startup → store in memory

Step 2: Every second, observe market:
  - BTC spot = $70,915
  - Strike = $70,000
  - Distance = (70915 - 70000) / 70000 = 1.3% above
  - Seconds to expiry = 87s
  - YES ask = $0.66
  - Momentum = neutral (price barely moving)

Step 3: Convert to buckets:
  - dist_bucket = 2 (1.3% → between 0.5% and 2.5% threshold)
  - time_bucket = 4 (87s → between 60s and 90s bucket)
  - price_bucket = 6 ($0.66 → between $0.60-$0.70)
  - direction = 1 (above strike)
  - momentum = 1 (neutral)

Step 4: Build key: "2,4,6,1,1"

Step 5: Look up q_table["2,4,6,1,1"] → [2.1, 14.7, 3.1]

Step 6: Max is 14.7 at index 1 → BUY YES ✅

Step 7: Place order for YES @ $0.66
```

### Why the Key Must Be 5D (Critical Bug We Fixed)

The Python learner saves keys as `"2,4,6,1,1"` (5 numbers).

Previously, the Rust bots were looking up keys as `"2,4,6,1,0,1,2"` (7 numbers — included spread_bucket and time_of_day_bucket that Python never saved).

Result: **Every single Q-table lookup returned nothing.** Bot defaulted to HOLD on every trade. This is why:
- Optimal bot had **0 trades** for days
- Ensemble RL voter was **never triggering**

Fix: Changed Rust key format to match Python's 5D format.

---

## 🏗️ Part 7: Model Registry vs Production Model

```
model_registry/btc/versions/
├── v20260211_062522/    ← Feb 11 model (old, inflated Sharpe=100)
│   ├── rl_policy.json
│   └── metadata.json
├── v20260409_013544/    ← Apr 9 model (Sharpe=5.52, honest)
│   ├── rl_policy.json
│   └── metadata.json
└── v20260409_053751/    ← Apr 9 latest (Sharpe=100 capped, current production)
    ├── rl_policy.json
    └── metadata.json

model/btc/
└── rl_policy.json       ← CURRENT PRODUCTION (bots read this)
```

**Registry** = archive of every version ever trained (never deleted)
**model/btc/rl_policy.json** = the one the bots actually use right now

When a new model is promoted, it gets:
1. Saved to `model_registry/btc/versions/vDATE_TIME/`
2. Copied to `model/btc/rl_policy.json`

---

## ❓ Part 8: Common Questions Answered

### "Why is the Q-table only 19.6% explored?"
The bot has only seen 778 of ~5,400 possible market situations. Many combinations (e.g., very high YES price + very close to expiry + falling momentum) simply haven't occurred yet in live trading. The remaining 80% of cells are blank — bot defaults to HOLD for those.

### "Why does the RL State Space Explorer show different cells than Top Confident States?"
The heatmap is a **2D slice** — you're filtering by Direction + Price + Momentum, but there are 5 dimensions. Top Confident States searches all 5D globally. The "BUY YES $18,787" states might be in a different spread_bucket or time-of-day slice that isn't visible in the current heatmap filter.

### "Why does the dashboard show 12:37 AM instead of current time?"
The timestamp shows **when the model was last trained and promoted** — not the current time. It's displayed in your browser's local timezone. The model hasn't been updated since 5:37 AM UTC because new models weren't beating the Sharpe threshold.

### "Why is Win Rate 72% but Total PnL is -$17,449?"
Wins and losses are not equal size. Classic asymmetry problem:
- 72% of trades win small amounts ($5-$20)
- 28% of trades lose large amounts ($50-$200)

Net result: lots of small wins, few large losses = negative overall. This is a bet-sizing and model quality problem, not a win rate problem.

### "Does the Optimal bot update its policy automatically?"
No. It loads the policy **once at startup** and freezes it. To use a newer policy, restart the bot. This is intentional — it acts as a stable benchmark.

### "What is Epsilon in the model JSON?"
The epsilon saved in the JSON is the exploration rate **at the time training stopped**. A low epsilon (e.g., 0.033) means the model had mostly converged — it was making confident decisions by the end of training, not random ones. Good sign.

### "Why do all models show WR=100%?"
This is **overfitting** — the model is evaluated on the same data it trained on. It memorized the past. Real live win rate is lower (around 40-60%). This is a known limitation of tabular Q-learning on small datasets.

---

## 📈 Part 9: What "Good" Looks Like

After 48-72 hours of honest training (with reward clipping), you should see:

| Metric | Currently | Target |
|--------|-----------|--------|
| Max Q-Value | 18,787 | < 50 |
| Sharpe Ratio | 0.45 | > 1.0 |
| Explored States | 778 (19.6%) | 1,500+ (30%+) |
| Optimal Bot Avg PnL/trade | -$10.79 | > $0 |
| Live Win Rate | ~40% | > 55% |

The system is currently in the **"cleaning up inflated history"** phase. The reward clipping fix is applied, the key format bug is fixed — now it just needs time to retrain with good data.

---

## 🗺️ Part 10: Full System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MARKET DATA FEEDS                     │
│         Coinbase + Binance (spot) + Kalshi (contracts)  │
└────────────────────────┬────────────────────────────────┘
                         │ price every second
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌─────────────┐
   │ Core ML  │   │ Bucket 9 │   │   Oracle    │
   │ (4 bots) │   │ (4 bots) │   │  (4 bots)  │
   │ Q-table  │   │ Q-table  │   │Black-Scholes│
   │ based    │   │ ITM only │   │    math     │
   └────┬─────┘   └────┬─────┘   └──────┬──────┘
        │               │                │
        ▼               ▼                ▼
   ┌──────────────────────────────────────────┐
   │          experiences.db / trades.db      │
   │     (every trade saved for learning)     │
   └───────────────────┬──────────────────────┘
                       │ every 2 hours
                       ▼
   ┌──────────────────────────────────────────┐
   │         continuous_learner.py            │
   │  reads experiences → trains Q-table      │
   │  if Sharpe > 0.3 → promote to production │
   └───────────────────┬──────────────────────┘
                       │ promotes
                       ▼
   ┌──────────────────────────────────────────┐
   │      model/btc/rl_policy.json            │
   │         (current best policy)            │
   └────────────┬──────────────┬──────────────┘
                │              │
                ▼              ▼
   ┌──────────────────┐  ┌──────────────────┐
   │  Ensemble Bots   │  │   Optimal Bot    │
   │    (4 bots)      │  │   (1 bot, BTC)   │
   │  2 voters must   │  │  Frozen policy   │
   │  agree to trade  │  │  Paper only      │
   └──────────────────┘  └──────────────────┘
```
