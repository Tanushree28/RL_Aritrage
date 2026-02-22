//! Bucket 9 Maximizer Bot - BTC only
//!
//! This bot ONLY trades bucket 9 (price $0.90-$0.99) with NO position limits.
//! The hypothesis is that bucket 9 has 99% win rate, so taking unlimited trades
//! should capture more profit than the 2-trade limit in the main bot.
//!
//! Key differences from main bot:
//! - Only trades bucket 9 (entry price >= $0.90)
//! - No position limits (can take unlimited trades per market)
//! - Only blocks opposite-direction trades in same 15-minute market
//!
//! Run with:
//!   PAPER_MODE=1 ./target/release/kalshi-arb-bucket9 > logs/bucket9-btc.log 2>&1 &

mod binance_listener;
mod coinbase_listener;
mod coinbase_price;
mod data_recorder;
mod experience_recorder;
mod kalshi_listener;
mod model;
mod spread_compression;
mod telegram_reporter;
mod trade_executor;
mod trade_executor_bucket9;

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
            std::env::set_var("KALSHI_ASSET", "bucket9-btc");
            AssetConfig {
                name: "bucket9-btc".to_string(),
                kalshi_series: "KXBTC15M".to_string(),
                kraken_symbol: "BTC/USD".to_string(),
                coinbase_pair: "BTC-USD".to_string(),
                volatility_baseline: 0.02,
                api_base_url: api_url,
            }
        }
        "eth" => {
            std::env::set_var("KALSHI_ASSET", "bucket9-eth");
            AssetConfig {
                name: "bucket9-eth".to_string(),
                kalshi_series: "KXETH15M".to_string(),
                kraken_symbol: "ETH/USD".to_string(),
                coinbase_pair: "ETH-USD".to_string(),
                volatility_baseline: 0.03,
                api_base_url: api_url,
            }
        }
        "sol" => {
            std::env::set_var("KALSHI_ASSET", "bucket9-sol");
            AssetConfig {
                name: "bucket9-sol".to_string(),
                kalshi_series: "KXSOL15M".to_string(),
                kraken_symbol: "SOL/USD".to_string(),
                coinbase_pair: "SOL-USD".to_string(),
                volatility_baseline: 0.05,
                api_base_url: api_url,
            }
        }
        "xrp" => {
            std::env::set_var("KALSHI_ASSET", "bucket9-xrp");
            AssetConfig {
                name: "bucket9-xrp".to_string(),
                kalshi_series: "KXXRP15M".to_string(),
                kraken_symbol: "XRP/USD".to_string(),
                coinbase_pair: "XRP-USD".to_string(),
                volatility_baseline: 0.04,
                api_base_url: api_url,
            }
        }
        _ => {
            eprintln!("[BUCKET9] Unknown asset '{}', defaulting to BTC", asset);
            std::env::set_var("KALSHI_ASSET", "bucket9-btc");
            AssetConfig {
                name: "bucket9-btc".to_string(),
                kalshi_series: "KXBTC15M".to_string(),
                kraken_symbol: "BTC/USD".to_string(),
                coinbase_pair: "BTC-USD".to_string(),
                volatility_baseline: 0.02,
                api_base_url: api_url,
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Get asset config first to show in banner
    let asset_config = Arc::new(get_asset_config());
    let asset_upper = asset_config.name.replace("bucket9-", "").to_uppercase();

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║          BUCKET 9 MAXIMIZER BOT ({})                        ║", asset_upper);
    eprintln!("║  Strategy: ONLY bucket 9 ($0.90-$0.99), NO position limits   ║");
    eprintln!("║  Expected: 99% win rate, unlimited trades per market         ║");
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

    eprintln!("[BUCKET9] All env vars validated. Spawning tasks...");

    // Shared state
    let state = Arc::new(Mutex::new(AppState::new()));

    // Asset config already created above for banner
    eprintln!("[BUCKET9] Asset: {} (series: {})", asset_config.name, asset_config.kalshi_series);

    // Spawn core tasks
    let h_coinbase       = tokio::spawn(coinbase_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_price = tokio::spawn(coinbase_price::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_binance        = tokio::spawn(binance_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_kalshi         = tokio::spawn(kalshi_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_telegram       = tokio::spawn(telegram_reporter::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_recorder       = tokio::spawn(data_recorder::run(Arc::clone(&state)));
    let h_executor       = tokio::spawn(trade_executor_bucket9::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_spread         = tokio::spawn(spread_compression::run(Arc::clone(&state), Arc::clone(&asset_config)));

    // Await all nine tasks
    let (r1, r2, r3, r4, r5, r6, r7, r8) =
        tokio::join!(h_coinbase, h_coinbase_price, h_binance, h_kalshi, h_telegram, h_recorder, h_executor, h_spread);
    r1??;
    r2??;
    r3??;
    r4??;
    r5??;
    r6??;
    r7??;
    r8??;

    Ok(())
}
