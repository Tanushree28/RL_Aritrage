use crate::model::{AppState, AssetConfig};
use anyhow::Result;
use futures::{SinkExt, StreamExt};
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use serde_json::{Value, json};

// Switch to Bitstamp WebSocket (Very simple & reliable)
const BITSTAMP_WS_URL: &str = "wss://ws.bitstamp.net";

pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> Result<()> {
    // Clean asset name (remove 'bucket9-' prefix if present)
    let clean_name = config.name.replace("bucket9-", "").to_lowercase();

    // Map asset to Bitstamp channel (e.g., "live_trades_btcusd")
    let channel = match clean_name.as_str() {
        "btc" => "live_trades_btcusd",
        "eth" => "live_trades_ethusd",
        "sol" => "live_trades_solusd",
        "xrp" => "live_trades_xrpusd",
        _ => "live_trades_btcusd", // default
    };
    
    eprintln!("[binance_listener] task started (actually Bitstamp for {})", channel);

    loop {
        eprintln!("[binance_listener] connecting to {}...", BITSTAMP_WS_URL);
        
        match connect_async(BITSTAMP_WS_URL).await {
            Ok((mut ws_stream, _)) => {
                eprintln!("[binance_listener] connected to Bitstamp");
                
                // Subscribe to trade updates
                let subscribe_msg = json!({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                });
                
                if let Err(e) = ws_stream.send(Message::Text(subscribe_msg.to_string())).await {
                    eprintln!("[binance_listener] failed to subscribe: {}", e);
                    sleep(Duration::from_secs(5)).await;
                    continue; 
                }

                let (_, mut read) = ws_stream.split();

                while let Some(msg) = read.next().await {
                   match msg {
                        Ok(Message::Text(text)) => {
                            // Parse JSON
                            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                                // Bitstamp format: {"event": "trade", "data": {"price": 68000.5, ...}}
                                if let Some(event) = v["event"].as_str() {
                                    if event == "trade" {
                                        if let Some(data) = v.get("data") {
                                            // Bitstamp sends price as float or string depending on version, usually float in V2
                                            if let Some(price_val) = data.get("price") {
                                                let price = if let Some(p) = price_val.as_f64() {
                                                    Some(p)
                                                } else if let Some(s) = price_val.as_str() {
                                                    s.parse::<f64>().ok()
                                                } else {
                                                    None
                                                };

                                                if let Some(p) = price {
                                                    if let Ok(mut s) = state.lock() {
                                                        s.binance_price = Some(p);
                                                        let now = chrono::Utc::now();
                                                        s.bitstamp_last_update = Some(now);

                                                        // Also feed into price_window
                                                        s.price_window.push_back((now, p));
                                                        let cutoff = now - chrono::TimeDelta::seconds(60);
                                                        while let Some(&(ts, _)) = s.price_window.front() {
                                                            if ts < cutoff { s.price_window.pop_front(); } else { break; }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else if event == "bts:subscription_succeeded" {
                                        eprintln!("[binance_listener] subscription succeeded for {}", channel);
                                    }
                                }
                            }
                        }
                        Ok(Message::Ping(_)) => {} // Auto-handled by tungstenite usually
                        Ok(Message::Close(_)) => {
                            eprintln!("[binance_listener] server closed connection");
                            break;
                        }
                        Err(e) => {
                            eprintln!("[binance_listener] WS error: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                eprintln!("[binance_listener] connection failed: {}", e);
            }
        }
        
        eprintln!("[binance_listener] disconnected, retrying in 5s...");
        sleep(Duration::from_secs(5)).await;
    }
}
