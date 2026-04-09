use std::sync::{Arc, Mutex};
use tokio::time::Duration;

use crate::model::{AppState, AssetConfig};

/// Entry point. Polls Coinbase REST every 2 seconds and updates
/// `AppState::coinbase_price`.  The price is independent of the Kraken feed
/// so that the CSV can capture real divergence between the two exchanges.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[coinbase_feed] task started (Coinbase REST polling {})", config.coinbase_pair);

    let client = reqwest::Client::new();
    let url = format!("https://api.coinbase.com/v2/prices/{}/spot", config.coinbase_pair);

    loop {
        tokio::time::sleep(Duration::from_secs(2)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[coinbase_feed] shutdown requested, exiting");
            return Ok(());
        }

        match fetch_price(&client, &url).await {
            Ok(price) => {
                let now = chrono::Utc::now();
                let mut app = state.lock().unwrap();
                app.coinbase_price = Some(price);
                app.coinbase_last_update = Some(now);

                // Also feed into price_window so settlement avg + feature computation
                // stays alive even when Kraken is down
                app.price_window.push_back((now, price));
                let cutoff = now - chrono::TimeDelta::seconds(60);
                while let Some(&(ts, _)) = app.price_window.front() {
                    if ts < cutoff { app.price_window.pop_front(); } else { break; }
                }
            }
            Err(e) => {
                eprintln!("[coinbase_feed] fetch error: {e}");
            }
        }
    }
}

/// GETs the current spot price from Coinbase.
/// Response schema: `{ "data": { "amount": "123456.78", ... } }`
async fn fetch_price(client: &reqwest::Client, url: &str) -> anyhow::Result<f64> {
    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Coinbase HTTP {}", resp.status()));
    }
    let json: serde_json::Value = resp.json().await?;
    let amount_str = json
        .get("data")
        .and_then(|d| d.get("amount"))
        .and_then(|a| a.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing data.amount"))?;
    let price: f64 = amount_str.parse()?;
    Ok(price)
}
