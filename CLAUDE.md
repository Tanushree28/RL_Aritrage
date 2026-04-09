# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kalshi crypto arbitrage bot targeting 15-minute binary options markets (BTC, ETH, SOL, XRP). Rust handles low-latency execution; Python handles ML training and monitoring.

## Build & Run Commands

### Rust

```bash
cargo build --release                    # Build all binaries (optimized)
cargo build --release --bin kalshi-arb-btc  # Build single asset binary
cargo clean && cargo build --release     # Full rebuild
```

**Binary targets:** `kalshi-arb-{btc,eth,sol,xrp}`, `kalshi-arb-rulebased`, `kalshi-arb-bucket9-{btc,eth,sol,xrp}`

### Running Bots

```bash
./start_everything.sh      # All 8 bots + dashboard
./start_all_assets.sh      # 4 core ML bots only
./start_bucket9_all.sh     # 4 bucket9 bots only
pkill -f "kalshi-arb"      # Emergency stop all bots
```

### Monitoring Dashboard

```bash
streamlit run monitoring/dashboard.py    # Local dashboard on :8501
```

### Deployment

```bash
./push_to_server.sh        # Rsync to DigitalOcean (68.183.174.225)
```

### Logs

```bash
tail -f logs/core-*.log      # Core ML bot logs
tail -f logs/bucket9-*.log   # Bucket9 bot logs
```

## Architecture

### Rust Core (`src/`)

Each bot binary spawns 9 concurrent Tokio tasks sharing a single `Arc<Mutex<AppState>>`:
1. **Coinbase listener** — WebSocket price feed
2. **Coinbase price aggregator** — 60-second rolling average (settlement reference)
3. **Binance listener** — Secondary price source
4. **Kalshi listener** — Market orderbook updates via WebSocket
5. **Telegram reporter** — Trade notifications
6. **Data recorder** — CSV tick recording
7. **Trade executor** — Core trading logic with RL policy
8. **Spread compression** — Spread analysis metrics
9. **Experience recorder** — RL training data to SQLite

**Three execution strategies** with separate main/executor files:
- **Core ML** (`main.rs` + `trade_executor.rs`) — RL-based, loads ONNX models via `ort`
- **Bucket9** (`main_bucket9.rs` + `trade_executor_bucket9.rs`) — Conservative arb at extreme probabilities (0.90-0.99)
- **Rule-based** (`main_rulebased.rs` + `trade_executor_rulebased.rs`) — Simple heuristic baseline

**Key data types** in `model.rs`: `AppState`, `AssetConfig`, `MarketState`, `RlState` (6D discretized), `Experience`, `OpenPosition`

### Python ML Pipeline

- `dqn_agent.py` — Double DQN with target networks (PyTorch). 5D input → 3 actions (HOLD, BUY_YES, BUY_NO)
- `rl_strategy.py` — Q-Learning with tabular Q-table, 6D state space (~7,920 states)
- `continuous_learner.py` — Training loop: prioritized replay from SQLite → train → promote model
- `experience_buffer.py` — SQLite-backed replay buffer with prioritized sampling and auto-pruning
- `model_versioning.py` — Atomic model promotion/rollback via versioned registry
- `train_model.py` — ONNX model export (18-feature feedforward network)

### Monitoring (`monitoring/`)

- `dashboard.py` — Streamlit dashboard: live trades, PnL charts, win rates, model versions
- `backtest_engine.py` — Three backtesting strategies: spread compression, momentum, rolling average

## Data Layout

```
data/{asset}/recordings/{YYYY-MM-DD}/*.csv   # Tick-level market data
data/{asset}/trades.db                        # Executed trades (SQLite)
data/{asset}/experiences.db                   # RL experience tuples (SQLite)
```

Asset dirs: `btc`, `eth`, `sol`, `xrp`, `btc-rulebased`, `bucket9-{btc,eth,sol,xrp}`

## Environment

Copy `.env.example` to `.env` and fill in credentials. Asset-specific overrides: `.env.btc`, `.env.eth`, `.env.sol`. Key vars:
- `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY` — RSA auth for Kalshi API
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — Trade notifications
- `PAPER_MODE=1` — Paper trading (no real orders); `0` for live

## Key Constants (trade_executor.rs)

- `MINIMUM_EDGE: 0.05` (5% min edge to trade)
- `MAX_EDGE: 0.30` (30% cap, filter miscalibration)
- `MIN_ENTRY_PRICE: 0.10` (avoid penny options)
- `MAX_PROB_MISMATCH: 0.20` (model-market divergence limit)
