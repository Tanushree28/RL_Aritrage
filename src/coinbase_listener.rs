use chrono::{TimeDelta, Utc};
use futures::{SinkExt, StreamExt};
use std::sync::{Arc, Mutex};
use tokio::time::Duration;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::model::{AppState, AssetConfig, IndexState};

// Coinbase's Advanced Trade WebSocket now requires authentication for all
// channels. Kraken v2 provides the same public ticker feed with no auth and
// is already listed as a primary source in the spec. When Coinbase API
// credentials become available a second listener can be added and the two
// feeds averaged together for the synthetic index.
const WS_URL: &str = "wss://ws.kraken.com/v2";
const WINDOW_SECS: i64 = 60;

/// Entry point. Connects to Kraken, processes ticker messages, and
/// reconnects with exponential backoff on any failure.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[price_feed] task started (Kraken {})", config.kraken_symbol);

    let mut backoff_secs: u64 = 1;

    loop {
        match connect_and_listen(&state, &config.kraken_symbol).await {
            Ok(()) => {
                eprintln!("[price_feed] stream ended, reconnecting...");
                backoff_secs = 1;
            }
            Err(e) => {
                eprintln!("[price_feed] error: {e} — retrying in {backoff_secs}s");
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(60);
            }
        }

        if state.lock().unwrap().shutdown {
            eprintln!("[price_feed] shutdown requested, exiting");
            return Ok(());
        }
    }
}

/// Opens the WebSocket, sends the subscribe frame, and processes incoming
/// ticker messages until the connection drops or an error occurs.
///
/// Kraken v2 sends several message types on the same connection (status,
/// subscribe ack, ticker snapshot, ticker updates). We parse every frame as a
/// generic JSON value and only act on `channel == "ticker"` frames, which
/// avoids needing separate structs for each message type.
async fn connect_and_listen(state: &Arc<Mutex<AppState>>, symbol: &str) -> anyhow::Result<()> {
    let (mut stream, _) = connect_async(WS_URL).await?;
    eprintln!("[price_feed] connected");

    // Ticker is a public channel — no authentication required.
    let subscribe = serde_json::json!({
        "method": "subscribe",
        "params": {
            "channel": "ticker",
            "symbol": [symbol]
        }
    });
    stream.send(Message::Text(subscribe.to_string())).await?;

    while let Some(msg) = stream.next().await {
        let msg = msg?;

        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Filter to ticker channel only (skips status, subscribe ack, etc.)
        if value.get("channel").and_then(|c| c.as_str()) != Some("ticker") {
            continue;
        }

        // data is an array of ticker objects; extract the one matching our symbol
        if let Some(data) = value.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if item.get("symbol").and_then(|s| s.as_str()) != Some(symbol) {
                    continue;
                }
                if let Some(last) = item.get("last").and_then(|l| l.as_f64()) {
                    update_state(state, last);
                }
            }
        }
    }

    Ok(())
}

fn update_state(state: &Arc<Mutex<AppState>>, price: f64) {
    let now = Utc::now();
    let mut app = state.lock().unwrap();

    app.price_window.push_back((now, price));

    // Evict entries outside the settlement window.
    let cutoff = now - TimeDelta::seconds(WINDOW_SECS);
    while let Some(&(ts, _)) = app.price_window.front() {
        if ts < cutoff {
            app.price_window.pop_front();
        } else {
            break;
        }
    }

    // Rolling average over the window.
    let rolling_avg = {
        let sum: f64 = app.price_window.iter().map(|(_, p)| p).sum();
        sum / app.price_window.len() as f64
    };

    app.coinbase = Some(IndexState {
        current_price:   price,
        rolling_avg_60s: rolling_avg,
        ts:              now,
    });
    app.kraken_last_update = Some(now);
}
