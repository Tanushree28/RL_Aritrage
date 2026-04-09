mod binance_listener;
mod black_scholes;
mod coinbase_listener;
mod coinbase_price;
mod data_recorder;
mod ensemble_executor;
mod experience_recorder;
mod kalshi_listener;
mod model;
mod spread_compression;
mod telegram_reporter;

use std::sync::{Arc, Mutex};

use model::{AppState, AssetConfig};

fn get_asset_config() -> AssetConfig {
    let asset = std::env::var("KALSHI_ASSET")
        .expect("KALSHI_ASSET env var required (btc, eth, sol, xrp, or ensemble-{btc,eth,sol,xrp})");

    let is_demo = std::env::var("KALSHI_ENV").unwrap_or_default().to_lowercase() == "demo";
    let api_url = if is_demo {
        "https://demo-api.kalshi.co".to_string()
    } else {
        "https://api.elections.kalshi.com".to_string()
    };

    // Strip "ensemble-" prefix to get underlying asset for market config,
    // but keep the full name so data_recorder writes to data/ensemble-{asset}/
    let underlying = asset.strip_prefix("ensemble-").unwrap_or(&asset);

    match underlying {
        "btc" => AssetConfig {
            name: asset.clone(),
            kalshi_series: "KXBTC15M".to_string(),
            kraken_symbol: "BTC/USD".to_string(),
            coinbase_pair: "BTC-USD".to_string(),
            volatility_baseline: 0.02,
            api_base_url: api_url,
            min_fair_value: 0.0,
        },
        "eth" => AssetConfig {
            name: asset.clone(),
            kalshi_series: "KXETH15M".to_string(),
            kraken_symbol: "ETH/USD".to_string(),
            coinbase_pair: "ETH-USD".to_string(),
            volatility_baseline: 0.03,
            api_base_url: api_url,
            min_fair_value: 0.0,
        },
        "sol" => AssetConfig {
            name: asset.clone(),
            kalshi_series: "KXSOL15M".to_string(),
            kraken_symbol: "SOL/USD".to_string(),
            coinbase_pair: "SOL-USD".to_string(),
            volatility_baseline: 0.04,
            api_base_url: api_url,
            min_fair_value: 0.0,
        },
        "xrp" => AssetConfig {
            name: asset.clone(),
            kalshi_series: "KXXRP15M".to_string(),
            kraken_symbol: "XRP/USD".to_string(),
            coinbase_pair: "XRP-USD".to_string(),
            volatility_baseline: 0.03,
            api_base_url: api_url,
            min_fair_value: 0.0,
        },
        _ => panic!("Invalid KALSHI_ASSET: {}", asset),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

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

    eprintln!("[kalshi-arb-ensemble] All env vars validated. Spawning tasks...");

    let state = Arc::new(Mutex::new(AppState::new()));
    let asset_config = Arc::new(get_asset_config());
    eprintln!("[kalshi-arb-ensemble] Asset: {} (series: {})", asset_config.name, asset_config.kalshi_series);

    let h_coinbase       = tokio::spawn(coinbase_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_price = tokio::spawn(coinbase_price::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_binance        = tokio::spawn(binance_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_kalshi         = tokio::spawn(kalshi_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_telegram       = tokio::spawn(telegram_reporter::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_recorder       = tokio::spawn(data_recorder::run(Arc::clone(&state)));
    let h_executor       = tokio::spawn(ensemble_executor::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_spread         = tokio::spawn(spread_compression::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_exp_recorder   = tokio::spawn(experience_recorder::run(Arc::clone(&state)));

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
