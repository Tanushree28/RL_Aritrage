//! Rule-based BTC trading bot entry point.
//!
//! This is a separate binary from the main RL-based bot for A/B testing.
//! Uses the OPTIMAL STRATEGY: 90-240s window, $0.15-$0.70 entry price.
//!
//! Run with:
//!   KALSHI_ASSET=btc cargo run --bin kalshi-arb-rulebased
//!
//! Or in paper mode:
//!   PAPER_MODE=1 KALSHI_ASSET=btc cargo run --bin kalshi-arb-rulebased

mod coinbase_listener;
mod coinbase_price;
mod data_recorder;
mod experience_recorder;
mod kalshi_listener;
mod model;
mod spread_compression;
mod telegram_reporter;
mod trade_executor;
mod trade_executor_rulebased;

use std::sync::{Arc, Mutex};

use model::{AppState, AssetConfig};

/// Get asset configuration (only BTC supported for rule-based bot)
/// Uses "btc-rulebased" as asset name to write to separate database
fn get_asset_config() -> AssetConfig {
    // Force set KALSHI_ASSET to btc-rulebased for data separation
    std::env::set_var("KALSHI_ASSET", "btc-rulebased");

    let is_demo = std::env::var("KALSHI_ENV").unwrap_or_default().to_lowercase() == "demo";
    let api_url = if is_demo {
        "https://demo-api.kalshi.co".to_string()
    } else {
        "https://api.elections.kalshi.com".to_string()
    };

    AssetConfig {
        name: "btc-rulebased".to_string(),  // Separate data directory
        kalshi_series: "KXBTC15M".to_string(),  // Same markets as BTC
        kraken_symbol: "BTC/USD".to_string(),
        coinbase_pair: "BTC-USD".to_string(),
        volatility_baseline: 0.02,
        api_base_url: api_url,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║          RULE-BASED BTC TRADING BOT                          ║");
    eprintln!("║  Strategy: 90-240s window, $0.15-$0.70 entry price           ║");
    eprintln!("║  Expected: ~30 trades/day, 67% win rate, ~$959/day PnL       ║");
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

    eprintln!("[RULEBASED] All env vars validated. Spawning tasks...");

    // Shared state
    let state = Arc::new(Mutex::new(AppState::new()));

    // Asset configuration (BTC only)
    let asset_config = Arc::new(get_asset_config());
    eprintln!("[RULEBASED] Asset: {} (series: {})", asset_config.name, asset_config.kalshi_series);

    // Spawn core tasks (using rule-based executor instead of RL executor)
    let h_coinbase       = tokio::spawn(coinbase_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_price = tokio::spawn(coinbase_price::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_kalshi         = tokio::spawn(kalshi_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_telegram       = tokio::spawn(telegram_reporter::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_recorder       = tokio::spawn(data_recorder::run(Arc::clone(&state)));
    let h_executor       = tokio::spawn(trade_executor_rulebased::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_spread         = tokio::spawn(spread_compression::run(Arc::clone(&state), Arc::clone(&asset_config)));
    // Note: experience_recorder not needed for rule-based bot (no RL training)

    // Await all tasks
    let (r1, r2, r3, r4, r5, r6, r7) =
        tokio::join!(h_coinbase, h_coinbase_price, h_kalshi, h_telegram, h_recorder, h_executor, h_spread);
    r1??;
    r2??;
    r3??;
    r4??;
    r5??;
    r6??;
    r7??;

    Ok(())
}
