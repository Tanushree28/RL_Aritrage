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

use std::sync::{Arc, Mutex};

use model::{AppState, AssetConfig};

/// Get asset configuration from environment variable
/// Get asset configuration from environment variable
fn get_asset_config() -> AssetConfig {
    let asset = std::env::var("KALSHI_ASSET")
        .expect("KALSHI_ASSET env var required (btc, eth, sol, or xrp)");

    let is_demo = std::env::var("KALSHI_ENV").unwrap_or_default().to_lowercase() == "demo";
    let api_url = if is_demo {
        "https://demo-api.kalshi.co".to_string()
    } else {
        "https://api.elections.kalshi.com".to_string()
    };

    match asset.as_str() {
        "btc" => AssetConfig {
            name: "btc".to_string(),
            kalshi_series: "KXBTC15M".to_string(),
            kraken_symbol: "BTC/USD".to_string(),
            coinbase_pair: "BTC-USD".to_string(),
            volatility_baseline: 0.02,
            api_base_url: api_url,
        },
        "eth" => AssetConfig {
            name: "eth".to_string(),
            kalshi_series: "KXETH15M".to_string(),
            kraken_symbol: "ETH/USD".to_string(),
            coinbase_pair: "ETH-USD".to_string(),
            volatility_baseline: 0.03,
            api_base_url: api_url,
        },
        "sol" => AssetConfig {
            name: "sol".to_string(),
            kalshi_series: "KXSOL15M".to_string(),
            kraken_symbol: "SOL/USD".to_string(),
            coinbase_pair: "SOL-USD".to_string(),
            volatility_baseline: 0.04,
            api_base_url: api_url,
        },
        "xrp" => AssetConfig {
            name: "xrp".to_string(),
            kalshi_series: "KXXRP15M".to_string(),
            kraken_symbol: "XRP/USD".to_string(),
            coinbase_pair: "XRP-USD".to_string(),
            volatility_baseline: 0.03,
            api_base_url: api_url,
        },
        _ => panic!("Invalid KALSHI_ASSET: {}", asset),
    }
}

/// Entry point. Loads env vars, constructs shared state, and spawns the four
/// core async tasks. Fails fast if any required env var is missing.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (non-fatal if missing — vars may already be
    // set in the shell environment).
    dotenvy::dotenv().ok();

    // ---------------------------------------------------------------------------
    // Validate required secrets upfront. Each line returns a clear error message
    // indicating exactly which variable is missing.
    // ---------------------------------------------------------------------------
    let _kalshi_api_key_id  = std::env::var("KALSHI_API_KEY_ID")
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

    eprintln!("[kalshi-arb] All env vars validated. Spawning tasks...");

    // ---------------------------------------------------------------------------
    // Shared state — single Arc<Mutex<>> passed to all tasks.
    // ---------------------------------------------------------------------------
    let state = Arc::new(Mutex::new(AppState::new()));

    // ---------------------------------------------------------------------------
    // Asset configuration — passed to tasks that need asset-specific behavior.
    // ---------------------------------------------------------------------------
    let asset_config = Arc::new(get_asset_config());
    eprintln!("[kalshi-arb] Asset: {} (series: {})", asset_config.name, asset_config.kalshi_series);

    // ---------------------------------------------------------------------------
    // Spawn the eight core tasks. Each receives a cheap Arc clone.
    // ---------------------------------------------------------------------------
    let h_coinbase       = tokio::spawn(coinbase_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_price = tokio::spawn(coinbase_price::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_binance        = tokio::spawn(binance_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_kalshi         = tokio::spawn(kalshi_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_telegram       = tokio::spawn(telegram_reporter::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_recorder       = tokio::spawn(data_recorder::run(Arc::clone(&state)));
    let h_executor       = tokio::spawn(trade_executor::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_spread         = tokio::spawn(spread_compression::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_exp_recorder   = tokio::spawn(experience_recorder::run(Arc::clone(&state)));

    // Await all nine.
    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) =
        tokio::join!(h_coinbase, h_coinbase_price, h_binance, h_kalshi, h_telegram, h_recorder, h_executor, h_spread, h_exp_recorder);
    r1??;
    r2??;
    r3??;
    r4??;
    r5??;
    r6??;
    r7??;
    r8??;
    r9??;

    Ok(())
}
