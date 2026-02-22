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
