# Kalshi Arbitration Bot - Logic & Architecture

## 1. Core Logic Flow (How it Works)

The bot operates in a continuous loop designed to be faster than human traders but safer than high-frequency trading.

### Step 1: Data Collection (The Eyes)

- **Kalshi Feed**: Connects to Kalshi WebSocket to receive live updates on prediction markets (e.g., "Will BTC be > $70,000?").
- **Reference Feed**: Connects to Kraken WebSocket to get the _real_ price of the asset (BTC/USD).
- **Time Synchronization**: Matches timestamps to ensure we are comparing apples to apples.

### Step 2: Analysis (The Brain)

Every time a new price comes in (millisecond level):

1.  **Calculate Probability**: Based on Kraken price and time remaining, what is the _true_ probability this event happens? (Using Black-Scholes or simple volatility models).
2.  **Compare to Kalshi**:
    - _Example_: Kalshi says "Yes" costs $0.40 (40% probability).
    - _Model_: Calculated probability is 60%.
    - _Opportunity_: The market is underpricing the event by 20%.

### Step 3: Execution (The Hands)

1.  **Check Constraints**: Do we have capital? Is the spread wide enough?
2.  **Place Order**: Send a limit order to Kalshi via REST API.
3.  **Record**: Log the trade to `trades.db` and send a Telegram notification.

---

## 2. Difference Between Bots

You are running two distinct strategies simultaneously. They look at the same data but behave differently.

### A. Core ML Bots (`kalshi-arb-btc`, etc.)

**"The Smart Trader"**

- **Strategy**: Uses Machine Learning (Reinforcement Learning) or simple spread logic.
- **Target**: Finds _any_ mispricing where the spread is > 25% (configurable).
  - _Example_: Buying at $0.40 when it should be $0.55.
- **Frequency**: Trades more often (multiple times per day/hour).
- **Risk**: Moderate. It takes bets on uncertain outcomes (40-60% probability) if the math looks good.
- **Code**: `src/main.rs` + `src/trade_executor.rs`

### B. Bucket 9 Bots (`kalshi-arb-bucket9-btc`, etc.)

**"The Sniper"**

- **Strategy**: Extremely conservative arbitrage.
- **Target**: Specifically looks for "Free Money" at the edges of the probability curve ($0.90 - $0.99).
  - _Example_: Kalshi price is $0.92 (92% chance), but real price is already WAY past the strike price (99.9% chance).
  - It buys the "Yes" at $0.92 to sell/settle at $1.00.
- **Frequency**: Very low (maybe once a day or less).
- **Risk**: Extremely Low. It theoretically only takes "sure things".
- **Code**: `src/main_bucket9.rs` + `src/trade_executor_bucket9.rs`

---

## 3. Data Structure

### Where is it saved?

Each bot has its own folder to avoid collision:

- **Core BTC**: `data/btc/`
- **Bucket 9 BTC**: `data/bucket9-btc/`

Inside each folder:

- `recordings/`: Raw CSV files of every single price tick (Gigabytes of data).
- `trades.db`: SQLite database of executed trades (The PnL history).
- `experiences.db`: For ML bots, this stores "what happened" to train the AI.

## 4. Why 8 Bots?

We run **1 process per asset per strategy** for stability.

- 4 Assets (BTC, ETH, SOL, XRP)
- x 2 Strategies (Core, Bucket 9)
- = **8 Processes**

If the "BTC Core" bot crashes, the "ETH Bucket 9" bot keeps running perfectly.

### 5. Telegram Notifications

The bot only sends Telegram messages when:

It starts up ("Bot Started").
It executes a trade ("Long BTC...").
It closes a trade ("Trade Closed: Profit...").
Hourly PnL summary.

### Oracle Bot Logic (Simple)

What it trades: Kalshi 15-minute crypto binary options — contracts that pay $1 if price is above a strike at expiry, $0 if below.

How it decides to trade:

Get fair value — Uses Black-Scholes formula to compute the true probability that price will be above the strike at expiry. Inputs: current Coinbase spot price, strike price, time remaining, and EWMA volatility.

4 gates must pass before placing an order:

Coinbase L2 data is fresh (< 0.5s old)
Time remaining is between 60–240 seconds (not too early, not too late)
Fair value ≥ 0.85 (only trade high-probability contracts)
Edge ≥ 0.05 (fair value minus Kalshi ask price — we need at least 5¢ of edge)
Order placement — Places a passive limit order (post-only, maker) at fair_value - 0.03 (3¢ below fair value). This avoids paying taker fees.

Dynamic amendment — Every loop tick (~1s), it recalculates fair value. If the optimal price has moved by ≥ 2¢, it cancels and replaces the resting order.

Win/Loss:

Buys YES when fair value is high (price is well above strike)
Wins ($1 payout) if price stays above strike at expiry → profit = $1 - entry_price
Loses ($0 payout) if price drops below strike → loss = entry_price
Example: Strike = $100k, BTC at $101k with 2 min left. B-S says 92% probability → fair value = $0.92. Kalshi ask = $0.86. Edge = $0.06 (≥ 5¢ threshold). Bot places limit buy YES at $0.89 ($0.92 - $0.03). If filled and BTC stays above $100k → wins $0.11. If BTC drops below $100k → loses $0.89.

Key insight: It only trades contracts where the math strongly favors the outcome (≥85% probability) AND the market is underpricing them by ≥5¢.
