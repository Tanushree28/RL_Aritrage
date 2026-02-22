# Project Specification: Kalshi 15-Min Crypto Arbitrage Bot (Rust Edition)

## 1. Project Goal
To build a high-performance, low-latency execution bot for **Kalshi 15-minute Crypto Markets** (BTC, ETH, SOL). The bot is written in **Rust** to maximize concurrency and safety. It uses an LSTM model (via ONNX runtime) to identify arbitrage opportunities between the "True" synthetic index price and Kalshi's implied probability.

## 2. Critical Architecture: The "Source of Truth"
**Constraint:** We strictly **DO NOT** use aggregators like CoinGecko. We construct a **Synthetic Index** locally to match the CME CF Real-Time Index used for settlement.

### A. The Synthetic Index Engine
* **Primary Data Source:** `tokio-tungstenite` WebSockets connected to **Coinbase** and **Kraken**.
* **Data Handling:** Compute a weighted average of these feeds in a dedicated thread to approximate the official settlement price in real-time.

### B. Settlement Logic (The 60-Second Rule)
* **Rule:** Settlement is the **simple average** of the index values over the **last 60 seconds** before expiration.
* **Implementation:** The bot maintains a `VecDeque<f64>` rolling window of 60 seconds. At any given tick, the "Projected Settlement" is the mean of this buffer, not the current spot price.

## 3. Data Ingestion & State Management (Rust)

### Input 1: Kalshi Market Data (WebSocket)
* **Source:** Kalshi WebSocket API.
* **Struct:** `MarketState { strike: f64, yes_ask: f64, no_ask: f64, ... }`
* **Update Frequency:** Event-driven (Real-time).

### Input 2: Live Asset Data (WebSocket)
* **Source:** Coinbase & Kraken Public Feeds.
* **Struct:** `IndexState { current_price: f64, rolling_avg_60s: f64 }`

## 4. Machine Learning Strategy (Hybrid Architecture)

**Training (Python):** Model is trained in PyTorch and exported to **.onnx**.
**Inference (Rust):** The bot loads the `.onnx` model using the `ort` crate for sub-millisecond inference.

* **Input Tensor:** `[Distance_to_Strike, Volatility_Index, Time_Remaining_Secs, Order_Book_Imbalance]`
* **Output:** Probability `P_model` (0.0 to 1.0).

## 5. Execution Logic & Fee Structure

### A. The Official Fee Formula
$$\text{Fee} = \lceil 0.07 \times C \times P \times (1 - P) \rceil$$
* **Implementation:** Rust `f64` calculation with `ceil()` rounding.

### B. Trade Execution
1.  **Trigger:** `(Model_Prob - Market_Prob) - (Fee_Impact) > Minimum_Edge (0.02)`
2.  **Action:** Send HTTP POST to Kalshi `create_order` endpoint.
3.  **Post-Action:** Immediately trigger Telegram "Trade Entry" notification.

## 6. Telegram Reporting Module

The bot must maintain a local SQLite database (`rusqlite`) or JSON state file to track open positions and calculate session stats.

### A. Trade Entry Notification
**Trigger:** On HTTP 201 (Order Created).
**Message Format:**
> 🚀 **OPEN POSITION: BTC > $96,000**
> 🟢 **Side:** YES
> 💰 **Price:** $0.45 (Implied: 45%)
> 🤖 **Model:** 72% Conf (Edge: +27%)
> 📅 **Expires:** 15:45

### B. Trade Settlement Notification
**Trigger:** Market status changes to `finalized` or balance update detected.
**Message Format:**
> ✅ **WIN** (or ❌ **LOSS**)
> 💵 **Profit:** +$55.00
> 📈 **ROI:** +40%
> ---
> 📊 **Session Stats**
> **Win Rate:** 62% (8/13)
> **Total PnL:** +$124.50

## 7. Data Recorder & Backtesting Module (New)

The bot must act as a "Black Box Flight Recorder," saving tick-level data to a CSV/Parquet file for future model retraining.

**File Structure:** `data/recordings/{date}/{ticker}.csv`
**Schema:**
1.  `timestamp_unix_ms`: (u64)
2.  `seconds_to_expiry`: (i32)
3.  `strike_price`: (f64)
4.  `kalshi_yes_ask`: (f64)
5.  `kalshi_no_ask`: (f64)
6.  `synth_index_spot`: (f64) - Current weighted avg.
7.  `synth_index_60s`: (f64) - **Critical:** The rolling settlement value.
8.  `coinbase_price`: (f64) - Raw feed A.
9.  `kraken_price`: (f64) - Raw feed B.
10. `model_probability`: (f64) - What the bot predicted at this moment.

## 8. Technical Stack (Rust)
* **Core:** `tokio` (Async Runtime), `serde` (JSON).
* **Networking:** `reqwest` (HTTP Client), `tokio-tungstenite` (WebSockets).
* **ML Inference:** `ort` (ONNX Runtime bindings).
* **Notifications:** `teloxide` or raw Telegram API.
* **Data Storage:** `csv` crate (for recording) + `rusqlite` (for PnL tracking).

## 9. Immediate Task for Code Generation
**Goal:** Initialize the Rust project structure.
**Task:** Generate the `Cargo.toml` file with all necessary dependencies and a `main.rs` skeleton that spawns four async tasks:
1.  `coinbase_listener`
2.  `kalshi_listener`
3.  `telegram_reporter`
4.  `data_recorder`