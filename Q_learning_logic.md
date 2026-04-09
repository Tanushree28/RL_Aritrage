# Q-Learning Logic — Kalshi 15-Minute Binary Options

## Overview

The bot learns which trades are profitable by recording every decision and its outcome, then training a Q-table that maps market conditions to the best action.

```
LIVE TRADING (Rust)              OFFLINE TRAINING (Python)
────────────────────             ────────────────────────
See market → discretize state    Load experiences from SQLite
Lookup Q-table → pick action     Q(s,a) += α * [reward - Q(s,a)]
Trade settles → record reward    Export new rl_policy.json
Store (state, action, reward)    Bot reloads on next restart
in SQLite
```

---

## 1. State Space (5D → 3,960 states)

Every market tick is compressed into 5 discrete features:

| # | Feature | Buckets | Values | Meaning |
|---|---------|---------|--------|---------|
| 1 | **Distance to Strike** | `[0.1, 0.2, 0.3, 0.4, 0.5]%` | 6 (0-5) | How far BTC price is from the strike price |
| 2 | **Time Remaining** | `[30, 60, 90, ..., 300]s` | 11 (0-10) | Seconds until market settlement |
| 3 | **Yes Ask Price** | `[$0.10, $0.20, ..., $0.90]` | 10 (0-9) | Current price to buy YES |
| 4 | **Direction** | Above / Below strike | 2 (0-1) | Is BTC above or below the strike? |
| 5 | **Momentum** | Down / Neutral / Up | 3 (0-2) | Price trend over last 30 ticks |

**Total: 6 × 11 × 10 × 2 × 3 = 3,960 possible states**

### Discretization Logic

```
distance_bucket = count(thresholds where dist_pct >= threshold)
time_bucket     = count(thresholds where secs_left <= threshold)
price_bucket    = count(thresholds where yes_ask >= threshold)
direction       = 1 if price > strike else 0
momentum        = 0 if change < -0.1% | 1 if neutral | 2 if change > +0.1%
```

**Example:** BTC at $87,500, strike $87,400, 45 seconds left, YES asking $0.72, price rising
→ State: `(1, 9, 7, 1, 2)` = 0.1% away, ~45s left, $0.70-0.80, above, upward

Python: `rl_strategy.py:102-131` | Rust: `trade_executor.rs:108-127`

---

## 2. Action Space

| Action | ID | Meaning |
|--------|-----|---------|
| HOLD | 0 | Don't trade (wait) |
| BUY YES | 1 | Buy YES contracts (bet price goes above strike) |
| BUY NO | 2 | Buy NO contracts (bet price goes below strike) |

---

## 3. Q-Table

A lookup table storing the expected reward for each (state, action) pair:

```
Q-table: { state_tuple → [Q_HOLD, Q_YES, Q_NO] }

Example:
  (1, 9, 7, 1, 2) → [0.0, +45.3, -12.1]
                       ↑      ↑       ↑
                     HOLD  BUY_YES  BUY_NO

  Best action: BUY_YES (highest Q-value = +45.3)
```

**Storage format** (`rl_policy.json`):
```json
{
  "buckets": {
    "distance": [0.1, 0.2, 0.3, 0.4, 0.5],
    "time": [30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
    "price": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "q_table": {
    "1,9,7,1,2": [0.0, 45.3, -12.1],
    "0,3,2,0,1": [0.0, -5.2, 18.7],
    ...
  },
  "timestamp": 1711500000
}
```

Key = comma-separated 5D state, Value = array of 3 Q-values.

Python: `rl_strategy.py:177-204` | Rust: `trade_executor.rs:84-89`

---

## 4. Experience Recording (Live Trading → SQLite)

### 4a. At Trade Entry

When the bot decides to trade, it captures the current state:

```
1. Compute momentum from last 30 price ticks
2. Discretize (dist_pct, secs_left, yes_ask, is_above, momentum) → 5D state
3. Lookup Q-table → epsilon-greedy action selection
4. If action = BUY_YES or BUY_NO → execute trade, save entry state
5. If action = HOLD → save state for counterfactual generation later
```

Rust: `trade_executor.rs:882-896, 998-1020`

### 4b. At Market Settlement (15 min later)

```
1. Query Kalshi API for settlement result (YES won or NO won)
2. Calculate reward = payout - cost
   - Won:  reward = (1.0 - entry_price) × contracts  (positive)
   - Lost: reward = -entry_price × contracts          (negative)
3. Create experience tuple: (state, action, reward, next_state)
4. Push to experience_queue
```

Rust: `trade_executor.rs:1355-1425`

### 4c. Counterfactual Experiences (HER)

For every actual trade, the bot also generates **what-if** experiences for the actions NOT taken:

```
If we bought YES and won (+$50):
  → Generate: "If we'd bought NO, we would have lost (-$30)"
  → Generate: "If we'd held, reward = 0"

This 3x multiplies training data and teaches the agent about ALL actions in each state.
```

Rust: `trade_executor.rs:1427-1500`

### 4d. SQLite Persistence

Every 5 seconds, experiences drain from memory to `data/{asset}/experiences.db`:

```sql
CREATE TABLE experiences (
    id INTEGER PRIMARY KEY,
    timestamp REAL,
    ticker TEXT,
    state_json TEXT,      -- "[1, 9, 7, 1, 2]" (6D in storage, spread dropped in training)
    action INTEGER,       -- 0=HOLD, 1=YES, 2=NO
    reward REAL,          -- PnL from settlement
    next_state_json TEXT, -- same as state (terminal, γ=0)
    priority REAL         -- abs(reward) + 0.01 (for prioritized replay)
)
```

**Note:** Rust stores 6D (includes `spread_bucket`). Python training drops spread to get 5D.

Rust: `experience_recorder.rs:14-156`

---

## 5. Training (Q-Learning Update)

### Core Equation

```
Q(s, a) ← Q(s, a) + α × [reward + γ × max Q(s') − Q(s, a)]
```

With **γ = 0** (episodic, each 15-min market settles independently):

```
Q(s, a) ← Q(s, a) + α × [reward − Q(s, a)]
```

This converges Q(s,a) toward the **average reward** observed when taking action `a` in state `s`.

### Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α (learning rate) | 0.1 → 0.01 | Step size, decays over training |
| γ (discount factor) | **0.0** | No future discounting — each market is independent |
| ε (exploration) | 1.0 → 0.01 | Start fully random, decay to mostly greedy |
| ε decay | 0.995/episode | Slow exploration decay |
| Episodes | 200 | Full passes through sampled data |
| Samples/episode | 1000 (per-asset), 2000 (pooled) | Random batch per episode |

### Training Process

```python
# train_optimal_policy.py
for episode in range(200):
    # Sample random batch from experience database
    batch = random_sample(train_data, size=1000)

    for (state, action, reward, next_state) in batch:
        # Convert 6D/7D state → 5D (drop spread, time_of_day)
        state_5d = migrate_state_to_5d(state)

        # Q-learning update
        current_q = Q[state_5d][action]
        Q[state_5d][action] += α × (reward - current_q)

    # Decay exploration and learning rate
    ε *= 0.995
    α *= 0.999
```

Python: `train_optimal_policy.py:116-179`

### State Migration (6D/7D → 5D)

Historical data has different dimensions. Training normalizes all to 5D:

```
7D (Rust full):  [dist, time, price, dir, spread, momentum, time_of_day]
                                           ↓ drop    ↓ keep    ↓ drop
6D (SQLite):     [dist, time, price, dir, spread, momentum]
                                           ↓ drop    ↓ keep
5D (training):   [dist, time, price, dir, momentum]  ← Q-table key format
```

Python: `train_optimal_policy.py:79-89`

---

## 6. Training Modes

### Per-Asset Training
Separate Q-table for each asset (BTC, ETH, SOL, XRP). Captures asset-specific behavior.

```bash
python3 train_optimal_policy.py --asset btc --episodes 200
```
Output: `model/btc/rl_policy.json`

### Pooled Universal Training
All 4 assets combined (~800k experiences). Better coverage of rare states.

```bash
python3 train_optimal_policy.py --all --episodes 200
```
Output: `model/universal/rl_policy_universal.json` + per-asset copies

### Why Both?
- Per-asset: BTC and XRP behave differently. Per-asset captures this.
- Pooled: Some states are rarely seen for one asset but common in another. Pooled fills gaps.
- Deployment: Use per-asset primary, pooled as fallback for unexplored states.

---

## 7. Evaluation

### Train/Test Split
- Sort all experiences by timestamp
- Oldest 80% → training
- Newest 20% → evaluation (out-of-sample)

### Metrics
- **Win Rate**: % of trades with positive PnL
- **Avg PnL**: Average dollar profit per trade
- **Sharpe Ratio**: avg_pnl / std_pnl (risk-adjusted return)
- **State Coverage**: How many of 3,960 states have been explored

### Baselines
- **Random**: Pick HOLD/YES/NO randomly (should be ~50% win rate, ~0 PnL)
- Trained policy should beat random on both win rate AND PnL

Python: `train_optimal_policy.py:182-331`

---

## 8. Production Usage (Rust)

### Startup
1. Load `model/{asset}/rl_policy.json`
2. Parse bucket thresholds + Q-table into memory
3. Read `EPSILON` env var (default 0.5 for exploration, 0.0 for pure exploitation)

### Every Market Tick
```
1. Compute: dist_pct, secs_left, yes_ask, is_above_strike, momentum
2. Discretize → 5D state tuple
3. Build key: "d,t,p,dir,mom"
4. Lookup Q-table:
   - Found: argmax([Q_HOLD, Q_YES, Q_NO]) → best action
   - Not found: random action (unexplored state)
5. Apply epsilon-greedy: with probability ε, override with random action
6. Execute trade if action ≠ HOLD
```

Rust: `trade_executor.rs:129-170`

### Key Safety Checks (before RL action is used)
- `MINIMUM_EDGE ≥ 0.05` (5% minimum profitable edge)
- `MAX_EDGE ≤ 0.30` (filter miscalibration)
- `MIN_ENTRY_PRICE ≥ 0.10` (avoid penny options)
- `MAX_PROB_MISMATCH ≤ 0.20` (model-market divergence limit)

---

## 9. Complete File Map

| File | Role |
|------|------|
| `src/trade_executor.rs` | State discretization, Q-table lookup, action selection, experience creation |
| `src/experience_recorder.rs` | Drain experience queue → SQLite every 5s |
| `src/model.rs` | RlState struct (7D), Experience struct, AppState |
| `data/{asset}/experiences.db` | SQLite database of all (state, action, reward) tuples |
| `rl_strategy.py` | QLearningAgent class, discretization buckets, save/load JSON |
| `train_optimal_policy.py` | Training script: load SQLite → Q-learn → export JSON |
| `continuous_learner.py` | Continuous training loop (runs alongside bots) |
| `model/{asset}/rl_policy.json` | Trained Q-table + bucket definitions (consumed by Rust) |
| `model/universal/rl_policy_universal.json` | Pooled cross-asset policy |
| `rl_dashboard.html` | Interactive Q-table visualization |

---

## 10. Key Insight: Why γ = 0?

Each Kalshi market is a **15-minute binary option** that settles independently. There's no "next state" that matters — you buy, wait, and it settles YES or NO. This makes it episodic:

- Traditional RL: `Q(s,a) = reward + γ × future_value` (chain of decisions)
- Our case: `Q(s,a) = reward` (one-shot decision, no future to discount)

With γ=0, Q-values converge to the **average PnL** for each (state, action) pair. The policy simply picks the action with the highest average historical profit in each market condition.
