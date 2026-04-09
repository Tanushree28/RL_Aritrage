# Reinforcement Learning State Space Design Plan

## Kalshi 15-Minute Crypto Market Trading Agent

## 1. Problem Statement

We are building a Reinforcement Learning (RL) agent that trades on Kalshi's 15-minute crypto binary option markets. The agent must decide, at each decision point, whether to **BUY YES**, **BUY NO**, or **HOLD** (take no action). The goal is to maximize cumulative profit (PnL) across many 15-minute market windows.

The key challenge is designing a **state representation** that:

1. Captures the relevant market information for decision-making
2. Remains small enough for feasible exploration and training
3. Generalizes well to unseen market conditions

---

## 2. Market Structure (Context)

Each Kalshi 15-minute crypto market has:

- A **strike price** (e.g., BTC > $97,250)
- A **settlement rule**: YES pays $1 if BTC's 60-second rolling average exceeds the strike at expiry; NO pays $1 otherwise
- An **order book** with Yes/No bid-ask prices (ranging $0.01 – $0.99)
- A **15-minute countdown** (900 seconds → 0)

Our trading bot operates on the **last 5 minutes** (300 seconds) of each market, where price dynamics become most predictive.

---

## 3. Action Space

| Action ID | Action      | Description                                                          |
| --------- | ----------- | -------------------------------------------------------------------- |
| 0         | **HOLD**    | Do not trade. Q-value represents opportunity cost.                   |
| 1         | **BUY YES** | Buy Yes contracts (betting price will be above strike at settlement) |
| 2         | **BUY NO**  | Buy No contracts (betting price will be below strike at settlement)  |

- HOLD has no direct cost but has opportunity cost (missed profitable trades)
- Each BUY action has an entry cost (the ask price) and binary settlement ($1 win or $0 loss)
- The agent receives reward = `PnL from settlement` after the market closes

---

## 4. State Space Design

### 4.1 Current Implementation (6 Dimensions)

| #   | Feature                | Discretization                                    | Bucket Count | Rationale                                                                                                         |
| --- | ---------------------- | ------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------- |
| 1   | Distance to Strike (%) | `[0.1, 0.2, 0.3, 0.4, 0.5]`                       | 6 (0–5)      | How far the current BTC price is from the strike as a percentage. Most important predictor of settlement outcome. |
| 2   | Time Remaining (sec)   | `[30, 60, 90, 120, 150, 180, 210, 240, 270, 300]` | 11 (0–10)    | Seconds left in the market. Determines how much price can still move.                                             |
| 3   | Yes Ask Price ($)      | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`   | 10 (0–9)     | Entry cost for a Yes contract. Directly affects risk/reward.                                                      |
| 4   | Direction              | Above / Below                                     | 2 (0–1)      | Whether BTC is currently above or below the strike price.                                                         |
| 5   | Bid-Ask Spread         | Tight (<$0.05) / Wide                             | 2 (0–1)      | Market liquidity indicator. Wide spreads increase cost.                                                           |
| 6   | Momentum               | Down / Neutral / Up                               | 3 (0–2)      | Rolling 30-point price change direction. Captures short-term trend.                                               |

**Total Theoretical State Space:**

```
6 × 11 × 10 × 2 × 2 × 3 = 7,920 unique states
```

**Q-Table Size:** 7,920 states × 3 actions = **23,760 Q-values**

### 4.2 State Discretization Logic

Each continuous observation is mapped to a discrete bucket:

```
Distance Bucket: count of thresholds exceeded
  - 0: price within 0.1% of strike (very close)
  - 5: price more than 0.5% away (far from strike)

Time Bucket: count of time thresholds remaining ≤ bucket value
  - 0: more than 300 seconds left (early, outside trading window)
  - 10: less than 30 seconds left (very close to settlement)

Price Bucket: count of price thresholds exceeded
  - 0: Yes ask < $0.10 (very cheap, market says NO is likely)
  - 9: Yes ask > $0.90 (very expensive, market says YES is likely)

Direction: binary
  - 0: BTC price < strike (below)
  - 1: BTC price > strike (above)

Spread: binary
  - 0: bid-ask spread ≤ $0.05 (liquid, tight)
  - 1: bid-ask spread > $0.05 (illiquid, wide)

Momentum: ternary
  - 0: 30-point rolling price change < -0.1% (downward trend)
  - 1: between -0.1% and +0.1% (neutral)
  - 2: > +0.1% (upward trend)
```

### 4.3. Dimension Correlation Analysis

Not all dimension combinations occur naturally. Key correlations:

| Correlation          | Implication                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Distance ↔ Price     | When distance is small (near strike), yes price ≈ $0.50. When far above strike, yes price → $0.90+. These are correlated. |
| Time ↔ Price         | Early in window, prices are moderate ($0.40–$0.60). Extreme prices ($0.05 or $0.95) only appear close to settlement.      |
| Spread ↔ Time        | Spreads tend to tighten as settlement approaches (more activity).                                                         |
| Direction ↔ Distance | Not correlated — price can be above/below at any distance.                                                                |
| Momentum ↔ Direction | Weakly correlated — upward momentum slightly more common when above.                                                      |

**Implication:** The effective reachable state space is estimated at **~2,000–3,000 states** (not the full 7,920), because many dimension combinations don't naturally co-occur.

### 4.4 Proposed Reduced State Space (~4,000 States)

Based on what was discussed, target a state space of approximately **4,000 states** to improve exploration coverage while retaining the most informative features. The recommended approach is to **drop the Spread dimension**:

**Justification for removing Spread:**

- In practice, the Kalshi crypto markets are liquid and the bid-ask spread is **tight (bucket 0) in >95% of observations**
- The Spread dimension splits every state in two, but only one half (tight) receives meaningful training data
- Removing it halves the state space with minimal information loss

**Proposed 5-Dimension State Space:**

| #   | Feature                | Discretization                                    | Bucket Count | Rationale                               |
| --- | ---------------------- | ------------------------------------------------- | ------------ | --------------------------------------- |
| 1   | Distance to Strike (%) | `[0.1, 0.2, 0.3, 0.4, 0.5]`                       | 6 (0–5)      | Primary predictor of settlement outcome |
| 2   | Time Remaining (sec)   | `[30, 60, 90, 120, 150, 180, 210, 240, 270, 300]` | 11 (0–10)    | How much time remains for price to move |
| 3   | Yes Ask Price ($)      | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`   | 10 (0–9)     | Entry cost / risk-reward ratio          |
| 4   | Direction              | Above / Below                                     | 2 (0–1)      | Price position relative to strike       |
| 5   | Momentum               | Down / Neutral / Up                               | 3 (0–2)      | Short-term price trend direction        |

```
6 × 11 × 10 × 2 × 3 = 3,960 unique states ≈ 4,000
```

**Q-Table Size:** 3,960 states × 3 actions = **11,880 Q-values**

**Expected benefits of the reduced state space:**

- Coverage improves from ~13% → ~26% with existing training data (same ~1,025 explored states now cover a larger fraction)
- Effective reachable coverage improves from ~35–50% → ~60–80%
- Faster convergence during training — fewer states to explore
- More reliable Q-values — each state receives more training visits on average

---

## 5. Exploration Coverage Analysis

### 5.1 Current Coverage

Based on the trained Q-table (`rl_policy.json` for BTC):

| Metric                        | Current (6-dim) | Proposed (5-dim)   |
| ----------------------------- | --------------- | ------------------ |
| Total possible states         | 7,920           | 3,960              |
| States with learned Q-values  | ~1,025          | ~1,025 (same data) |
| Theoretical coverage          | **~13%**        | **~26%**           |
| Estimated reachable coverage  | **~35–50%**     | **~60–80%**        |
| States with non-zero Q-values | ~800            | ~800               |
| Training episodes completed   | ~50–100         | ~50–100            |

### 5.2 Exploration Strategy

The agent uses **ε-greedy exploration**:

| Parameter   | Value | Purpose                         |
| ----------- | ----- | ------------------------------- |
| ε_start     | 1.0   | 100% random initially           |
| ε_min       | 0.01  | 1% random at convergence        |
| ε_decay     | 0.99  | Decay rate per episode          |
| ε_warmstart | 0.15  | When fine-tuning existing model |

**Concern:** With 7,920 states and only ~50–100 episodes × ~200 markets per episode, some states are visited zero or very few times. The Q-values for rarely-visited states are unreliable.

### 5.3 Coverage Distribution (Expected)

```
Frequently Visited (>50 visits):   ~200 states  — Reliable Q-values
Moderately Visited (10-50):        ~400 states  — Somewhat reliable
Rarely Visited (1-10):             ~425 states  — Unreliable Q-values
Never Visited:                   ~6,895 states  — Default to [0,0,0]
```

---

## 6. Design Considerations & Open Questions

### 6.1 Time Granularity

**Current:** 30-second intervals (11 buckets over 300 seconds).

| Option                 | Time Buckets | Total States | Pros                              | Cons                               |
| ---------------------- | ------------ | ------------ | --------------------------------- | ---------------------------------- |
| 30 sec (current)       | 11           | 7,920        | Manageable size, decent coverage  | May miss rapid changes in last 60s |
| 15 sec                 | 21           | 15,120       | Better resolution near settlement | Doubles state space, needs 2× data |
| Non-uniform (proposed) | ~14          | 10,080       | Fine-grained when it matters      | More complex discretization        |

**Proposed Non-Uniform Time Buckets:**

```
[15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 600, 900]
  ^--- 15-sec resolution in last minute ---^  ^--- 30-sec mid ---^  ^- coarse early -^
```

This gives finer resolution in the **critical last 60 seconds** while keeping the total state space manageable.

### 6.2 State Space Reduction Options

| Approach                                        | New State Count | Trade-off                                                |
| ----------------------------------------------- | --------------- | -------------------------------------------------------- |
| **Drop Spread (recommended)** ✅                | **3,960**       | **Minimal info loss, doubles coverage, hits ~4K target** |
| Drop Momentum (weakly predictive)               | 2,640           | Lose trend info but triple coverage                      |
| Drop both Spread + Momentum                     | 1,320           | Very compact, fast convergence, but lose information     |
| Merge Distance + Direction into signed distance | 5,940           | More natural representation                              |

### 6.3 Tabular Q-Learning vs Deep Q-Network (DQN)

We currently maintain **both** approaches:

| Aspect             | Tabular Q-Learning                 | DQN (Neural Network)              |
| ------------------ | ---------------------------------- | --------------------------------- |
| File               | `rl_strategy.py`                   | `dqn_agent.py`                    |
| State handling     | Discrete buckets                   | Continuous (6-dim vector)         |
| Generalization     | None — unseen state = no knowledge | Interpolates between states       |
| Exploration needed | Must visit every state             | Can generalize from nearby states |
| Interpretability   | ✅ Full — can inspect each Q-value | ❌ Black box                      |
| Visualization      | ✅ Easy — heatmaps, tables         | ❌ Hard — must sample             |
| Training speed     | Fast (table lookup)                | Slower (backprop)                 |
| Deployed in Rust   | ✅ Via JSON Q-table lookup         | ✅ Via ONNX inference             |

**Recommendation:** Use tabular Q-learning for **interpretability and visualization**, DQN for **production trading**. The dashboard should visualize the Q-table to build intuition, while the DQN handles unseen states gracefully.

---

## 7. Dashboard Visualization Plan

### 7.1 Purpose

Build an interactive dashboard that visualizes the agent's learned state space to:

1. **Understand** what the agent has learned (which states → which actions)
2. **Identify gaps** where the agent lacks training data
3. **Validate** that the policy makes intuitive sense (e.g., BUY YES when close to strike + above + near settlement)
4. **Diagnose** unexpected behavior (why did the agent HOLD when it should have traded?)

### 7.2 Dashboard Components

#### Component 1: Policy Heatmap (Primary View)

- **Axes:** Distance to Strike (Y-axis) × Time Remaining (X-axis)
- **Cell Color:** Best action (Green = BUY YES, Red = BUY NO, Gray = HOLD, Dark = Unexplored)
- **Cell Text:** Best Q-value magnitude
- **Interactive:** Click cell to inspect Q-values

#### Component 2: Filter Controls

- **Direction toggle:** Above Strike / Below Strike / All
- **Spread toggle:** Tight / Wide / All
- **Momentum toggle:** Down / Neutral / Up / All
- These filters slice the 6D state space into viewable 2D cross-sections

#### Component 3: Q-Value Inspector Panel

- **Trigger:** Click any heatmap cell
- **Shows:** Bar chart of all 3 Q-values (HOLD, BUY YES, BUY NO)
- **Shows:** List of all individual states within that (distance, time) combination
- **Shows:** Breakdown by price bucket, direction, spread, momentum

#### Component 4: Exploration Coverage View

- **Toggle view:** Switch heatmap coloring to show visit counts instead of actions
- **Color scale:** Red (never visited) → Yellow (few visits) → Green (well-explored)
- **Purpose:** Identify training data gaps

#### Component 5: Summary Statistics

- Total states explored vs theoretical maximum
- Action distribution (% BUY YES / BUY NO / HOLD)
- Maximum Q-value and its corresponding state
- Coverage percentage

#### Component 6: Top Confident States Table

- Sorted by highest Q-value (most confident decisions)
- Shows state parameters, best action, and Q-value magnitude
- Helps identify the agent's strongest convictions

### 7.3 Technical Implementation

| Aspect      | Choice                                               | Rationale                                              |
| ----------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Technology  | Standalone HTML + JavaScript                         | No dependencies, shareable, works offline              |
| Data Source | `rl_policy.json` file (drag & drop or auto-load)     | Already exported by training pipeline                  |
| Hosting     | Served via `python -m http.server` or opened locally | Simple, no server needed                               |
| Styling     | Dark theme, glassmorphism                            | Consistent with existing Streamlit dashboard aesthetic |

---

## 8. Reward Structure

| Outcome            | Reward                             | Description                 |
| ------------------ | ---------------------------------- | --------------------------- |
| BUY YES → YES wins | `+(1.0 - entry_price) × contracts` | Profit from correct YES bet |
| BUY YES → NO wins  | `-(entry_price) × contracts`       | Loss from incorrect YES bet |
| BUY NO → NO wins   | `+(1.0 - entry_price) × contracts` | Profit from correct NO bet  |
| BUY NO → YES wins  | `-(entry_price) × contracts`       | Loss from incorrect NO bet  |
| HOLD (any outcome) | `0.0`                              | No risk, no reward          |

- **Bet size:** Fixed at $100 per decision
- **Contracts:** `floor(100 / entry_price)`
- **Settlement:** Binary — each contract pays $1 or $0

---

## 9. Next Steps

### Phase 1: Dashboard Implementation

- [ ] Build interactive state space visualization dashboard
- [ ] Load and parse existing Q-table data
- [ ] Implement heatmap, filters, and inspector

### Phase 2: State Space Analysis (Using Dashboard)

- [ ] Identify which states have strong learned policies
- [ ] Identify exploration gaps and unreliable states
- [ ] Validate policy makes intuitive sense

### Phase 3: State Space Refinement (Based on Analysis)

- [ ] Decide on time granularity (keep 30s vs non-uniform)
- [ ] Evaluate dropping low-information dimensions (spread, momentum)
- [ ] Consider signed distance (merge direction + distance)
- [ ] Retrain with refined state space

### Phase 4: Compare Tabular vs DQN

- [ ] Add DQN policy visualization to dashboard
- [ ] Compare action recommendations side-by-side
- [ ] Measure performance difference on held-out data

---

## 10. References

- `rl_strategy.py` — Tabular Q-learning implementation and state discretization
- `dqn_agent.py` — Deep Q-Network implementation
- `model/btc/rl_policy.json` — Trained Q-table for BTC (exported for Rust consumption)
- `src/trade_executor.rs` — Rust-side state discretization (must match Python)
