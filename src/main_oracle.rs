//! Oracle Bot — Main Entry Point
//!
//! Coinbase-Oracle Sniper Bot: Uses Binary Black-Scholes fair value pricing
//! with Coinbase Advanced Trade WebSocket L2 data to trade Kalshi 15-minute
//! crypto futures. Places passive maker orders at a 3-cent discount to fair
//! value and dynamically amends them as market conditions change.
//!
//! Run with:
//!   KALSHI_ASSET=btc ./target/release/kalshi-arb-oracle > logs/oracle-btc.log 2>&1 &

mod black_scholes;
mod coinbase_advanced_listener;
mod coinbase_listener;
mod coinbase_price;
mod data_recorder;
mod experience_recorder;
mod kalshi_listener;
mod model;
mod order_manager;
mod telegram_reporter;
mod trade_executor;
mod trade_executor_oracle;

use std::sync::{Arc, Mutex};

use model::{AppState, AssetConfig};

/// Get asset configuration based on KALSHI_ASSET env var
fn get_asset_config() -> AssetConfig {
    let asset = std::env::var("KALSHI_ASSET")
        .unwrap_or_else(|_| "btc".to_string())
        .to_lowercase();

    let is_demo = std::env::var("KALSHI_ENV").unwrap_or_default().to_lowercase() == "demo";
    let api_url = if is_demo {
        "https://demo-api.kalshi.co".to_string()
    } else {
        "https://api.elections.kalshi.com".to_string()
    };

    match asset.as_str() {
        "btc" => {
            std::env::set_var("KALSHI_ASSET", "oracle-btc");
            AssetConfig {
                name: "oracle-btc".to_string(),
                kalshi_series: "KXBTC15M".to_string(),
                kraken_symbol: "BTC/USD".to_string(),
                coinbase_pair: "BTC-USD".to_string(),
                volatility_baseline: 0.02,
                api_base_url: api_url,
                min_fair_value: 0.85,
            }
        }
        "eth" => {
            std::env::set_var("KALSHI_ASSET", "oracle-eth");
            AssetConfig {
                name: "oracle-eth".to_string(),
                kalshi_series: "KXETH15M".to_string(),
                kraken_symbol: "ETH/USD".to_string(),
                coinbase_pair: "ETH-USD".to_string(),
                volatility_baseline: 0.03,
                api_base_url: api_url,
                min_fair_value: 0.85,
            }
        }
        "sol" => {
            std::env::set_var("KALSHI_ASSET", "oracle-sol");
            AssetConfig {
                name: "oracle-sol".to_string(),
                kalshi_series: "KXSOL15M".to_string(),
                kraken_symbol: "SOL/USD".to_string(),
                coinbase_pair: "SOL-USD".to_string(),
                volatility_baseline: 0.05,
                api_base_url: api_url,
                min_fair_value: 0.85,
            }
        }
        "xrp" => {
            std::env::set_var("KALSHI_ASSET", "oracle-xrp");
            AssetConfig {
                name: "oracle-xrp".to_string(),
                kalshi_series: "KXXRP15M".to_string(),
                kraken_symbol: "XRP/USD".to_string(),
                coinbase_pair: "XRP-USD".to_string(),
                volatility_baseline: 0.04,
                api_base_url: api_url,
                min_fair_value: 0.85,
            }
        }
        _ => {
            eprintln!("[Oracle] Unknown asset '{}', defaulting to BTC", asset);
            std::env::set_var("KALSHI_ASSET", "oracle-btc");
            AssetConfig {
                name: "oracle-btc".to_string(),
                kalshi_series: "KXBTC15M".to_string(),
                kraken_symbol: "BTC/USD".to_string(),
                coinbase_pair: "BTC-USD".to_string(),
                volatility_baseline: 0.02,
                api_base_url: api_url,
                min_fair_value: 0.85,
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let asset_config = Arc::new(get_asset_config());
    let asset_upper = asset_config.name.replace("oracle-", "").to_uppercase();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║         ORACLE SNIPER BOT ({})                          ║", asset_upper);
    eprintln!("║  Strategy: Binary Black-Scholes + Coinbase L2              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Validate required secrets
    let _kalshi_api_key_id = std::env::var("KALSHI_API_KEY_ID")
        .map_err(|_| anyhow::anyhow!("Required env var KALSHI_API_KEY_ID is not set"))?;
    let _kalshi_private_key = std::env::var("KALSHI_PRIVATE_KEY")
        .map_err(|_| anyhow::anyhow!("Required env var KALSHI_PRIVATE_KEY is not set"))?;
    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));

    if !paper_mode {
        std::env::var("TELEGRAM_BOT_TOKEN")
            .map_err(|_| anyhow::anyhow!("Required env var TELEGRAM_BOT_TOKEN is not set"))?;
        std::env::var("TELEGRAM_CHAT_ID")
            .map_err(|_| anyhow::anyhow!("Required env var TELEGRAM_CHAT_ID is not set"))?;
    }

    let coinbase_status = if std::env::var("COINBASE_API_KEY").is_ok() {
        "L2 WebSocket"
    } else {
        "REST Polling (no COINBASE_API_KEY)"
    };

    eprintln!("[Oracle] Mode: {} | Asset: {} | Series: {}",
        if paper_mode { "PAPER" } else { "LIVE" },
        asset_config.name, asset_config.kalshi_series);
    eprintln!("[Oracle] Coinbase feed: {}", coinbase_status);

    // Shared state
    let state = Arc::new(Mutex::new(AppState::new()));

    // Spawn core tasks
    let h_kraken          = tokio::spawn(coinbase_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_l2     = tokio::spawn(coinbase_advanced_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_price  = tokio::spawn(coinbase_price::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_kalshi          = tokio::spawn(kalshi_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_recorder        = tokio::spawn(data_recorder::run(Arc::clone(&state)));
    let h_executor        = tokio::spawn(trade_executor_oracle::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_telegram        = tokio::spawn(telegram_reporter::run(Arc::clone(&state), Arc::clone(&asset_config)));

    // Await all tasks
    let (r0, r1, r2, r3, r4, r5, r6) =
        tokio::join!(h_kraken, h_coinbase_l2, h_coinbase_price, h_kalshi, h_recorder, h_executor, h_telegram);
    r0??;
    r1??;
    r2??;
    r3??;
    r4??;
    r5??;
    r6??;

    Ok(())
}
