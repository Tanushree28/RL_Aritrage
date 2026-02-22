# RL Arbitrage (Kalshi Crypto Arbitrage Bot)

**🟢 Live Server Dashboard:** [http://68.183.174.225:8501](http://68.183.174.225:8501)

## Overview

This repository contains a high-performance execution bot developed in Rust targeting Kalshi's 15-minute crypto markets. It heavily relies on a custom-built synthetic index engine and is transitioning towards utilizing Deep Q-Network (DQN) Reinforcement Learning algorithms for intelligent trade execution across major crypto assets (BTC, ETH, SOL, XRP).

## Key Features

- **Synthetic Index Engine:** A fast engine written in Rust focusing on accurate 60-second rolling settlement rules.
- **Multi-Exchange Websockets:** Real-time, low-latency data streaming and book tracking from Coinbase and Kraken APIs.
- **Advanced Execution Strategy:** Operates off complex execution logic evaluating model probabilities, market implied probabilities, and transaction fee impacts.
- **Machine Learning Pipeline:** Dedicated Python implementations for Reinforcement Learning (RL) to analyze and generate trading policies.
- **Live Monitoring Dashboard:** A Streamlit-based operations dashboard giving insights into real-time metrics, logs, and bot states.

## Getting Started

### Prerequisites

- **Rust Toolchain:** (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Python 3.9+:** (Required for Machine Learning model training and Streamlit Monitoring)
- API Keys for Kalshi, Coinbase, and Kraken.

### Local Initialization

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Tanushree28/RL_Aritrage.git
   cd RL_Aritrage
   ```

2. **Environment Configuration:**
   Copy the example environment securely, then add your credentials.

   ```bash
   cp .env.example .env
   ```

3. **Running the Bot locally:**
   Start the core services via the provided bash scripts spanning multiple assets:

   ```bash
   chmod +x start_everything.sh
   ./start_everything.sh
   ```

4. **Running the Dashboard:**
   To visualize market monitoring and performance metrics:
   ```bash
   streamlit run monitoring/dashboard.py
   ```

## Deployment

Automated scripts are set up to sync and deploy seamlessly to a DigitalOcean Droplet:

```bash
./push_to_server.sh
```

_Note: Deployments exclude `target/` and `/logs/` to safeguard production bandwidth._

## Documentation

For in-depth analysis on the reinforcement strategies, architectural breakdown, and daily trading metrics overviews, refer to the included markdown strategies and logs in the repository.
