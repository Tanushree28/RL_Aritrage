mod binance_listener;
mod black_scholes;
mod coinbase_listener;
mod coinbase_price;
mod data_recorder;
mod experience_recorder;
mod kalshi_listener;
mod model;
mod optimal_executor;
mod spread_compression;
mod telegram_reporter;

use std::sync::{Arc, Mutex};
use model::{AppState, AssetConfig};

fn get_asset_config() -> AssetConfig {
    let asset = std::env::var("KALSHI_ASSET")
        .expect("KALSHI_ASSET env var required (e.g. optimal-btc)");

    let api_url = "https://api.elections.kalshi.com".to_string(); // always prod feed, paper trades only

    // Strip "optimal-" prefix to get underlying asset for market config
    let underlying = asset.strip_prefix("optimal-").unwrap_or(&asset);

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
        _ => panic!("Invalid KALSHI_ASSET '{}' — use optimal-btc, optimal-eth, optimal-sol, or optimal-xrp", asset),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // API keys needed for market data feed (not for order placement)
    std::env::var("KALSHI_API_KEY_ID")
        .map_err(|_| anyhow::anyhow!("Required env var KALSHI_API_KEY_ID is not set"))?;
    std::env::var("KALSHI_PRIVATE_KEY")
        .map_err(|_| anyhow::anyhow!("Required env var KALSHI_PRIVATE_KEY is not set"))?;

    // Force paper mode — optimal executor never places real orders
    std::env::set_var("PAPER_MODE", "1");

    let asset_config = Arc::new(get_asset_config());
    eprintln!("[kalshi-arb-optimal] Asset: {} (series: {})", asset_config.name, asset_config.kalshi_series);

    let policy_path = std::env::var("OPTIMAL_POLICY_PATH")
        .unwrap_or_else(|_| {
            let underlying = asset_config.name.strip_prefix("optimal-").unwrap_or(&asset_config.name);
            format!("model/{}/rl_policy.json", underlying)
        });
    eprintln!("[kalshi-arb-optimal] Policy: {}", policy_path);

    let state = Arc::new(Mutex::new(AppState::new()));

    let h_coinbase       = tokio::spawn(coinbase_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_coinbase_price = tokio::spawn(coinbase_price::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_binance        = tokio::spawn(binance_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_kalshi         = tokio::spawn(kalshi_listener::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_recorder       = tokio::spawn(data_recorder::run(Arc::clone(&state)));
    let h_executor       = tokio::spawn(optimal_executor::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_spread         = tokio::spawn(spread_compression::run(Arc::clone(&state), Arc::clone(&asset_config)));
    let h_exp_recorder   = tokio::spawn(experience_recorder::run(Arc::clone(&state)));
    let h_telegram       = tokio::spawn(telegram_reporter::run(Arc::clone(&state), Arc::clone(&asset_config)));

    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) =
        tokio::join!(h_coinbase, h_coinbase_price, h_binance, h_kalshi, h_recorder, h_executor, h_spread, h_exp_recorder, h_telegram);
    r1??; r2??; r3??; r4??; r5??; r6??; r7??; r8??; r9??;

    Ok(())
}
