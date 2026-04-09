//! Coinbase Advanced Trade WebSocket L2 order book listener.
//!
//! Connects to the Coinbase Advanced Trade WebSocket using CDP API key
//! JWT authentication, subscribes to the level2 channel, and maintains
//! best bid/ask + mid-price in AppState::coinbase_l2.

use chrono::{TimeDelta, Utc};
use futures::{SinkExt, StreamExt};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokio::time::Duration;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use base64::Engine;
use rand::Rng;

use crate::model::{AppState, AssetConfig, CoinbaseL2State};

const WS_URL: &str = "wss://advanced-trade-ws.coinbase.com";
const STALE_THRESHOLD_SECS: i64 = 5;
const MAX_VOL_WINDOW: usize = 5000;

/// Load Coinbase API credentials from environment variables.
fn load_coinbase_credentials() -> Option<(String, String)> {
    if let (Ok(key), Ok(secret)) = (
        std::env::var("COINBASE_API_KEY").or_else(|_| std::env::var("COINBASE_API_KEY_NAME")),
        std::env::var("COINBASE_API_SECRET"),
    ) {
        Some((key, secret))
    } else {
        None
    }
}

/// Entry point. Connects to Coinbase Advanced Trade WebSocket, processes L2
/// updates, and reconnects with exponential backoff on any failure.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[coinbase_l2] task started (Coinbase Advanced Trade {})", config.coinbase_pair);

    let (api_key, api_secret) = match load_coinbase_credentials() {
        Some(creds) => creds,
        None => {
            eprintln!("[coinbase_l2] No Coinbase credentials found — running without L2 feed");
            eprintln!("[coinbase_l2] Set COINBASE_API_KEY and COINBASE_API_SECRET env vars");
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                if state.lock().unwrap().shutdown { return Ok(()); }
            }
        }
    };

    let mut backoff_secs: u64 = 1;

    loop {
        match connect_and_listen(&state, &config.coinbase_pair, &api_key, &api_secret, &config.name).await {
            Ok(()) => {
                eprintln!("[coinbase_l2] stream ended, reconnecting...");
                backoff_secs = 1;
            }
            Err(e) => {
                eprintln!("[coinbase_l2] error: {e} — retrying in {backoff_secs}s");
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(60);
            }
        }

        if state.lock().unwrap().shutdown {
            eprintln!("[coinbase_l2] shutdown requested, exiting");
            return Ok(());
        }
    }
}

/// Build a CDP JWT for Coinbase Advanced Trade WebSocket authentication.
/// Uses ES256 signing with the jsonwebtoken crate.
fn build_coinbase_jwt(api_key: &str, api_secret: &str) -> anyhow::Result<String> {
    // Decode the base64 private key
    let key_bytes = base64::engine::general_purpose::STANDARD
        .decode(api_secret)
        .map_err(|e| anyhow::anyhow!("Failed to decode COINBASE_API_SECRET: {e}"))?;

    if key_bytes.len() < 32 {
        return Err(anyhow::anyhow!("COINBASE_API_SECRET too short: {} bytes", key_bytes.len()));
    }

    // Extract the 32-byte EC private scalar
    let private_scalar = &key_bytes[..32];

    // Build PKCS#8 DER wrapper for P-256 private key
    let pkcs8_der = build_ec_p256_pkcs8(private_scalar);

    // Try ES256 JWT
    let encoding_key = jsonwebtoken::EncodingKey::from_ec_der(&pkcs8_der);

    let now = Utc::now().timestamp();
    let nonce: u128 = rand::rngs::OsRng.gen();

    let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::ES256);
    header.kid = Some(api_key.to_string());
    header.typ = Some("JWT".to_string());

    let claims = serde_json::json!({
        "sub": api_key,
        "iss": "coinbase-cloud",
        "nbf": now,
        "exp": now + 120,
        "aud": ["cdp_service"],
        "nonce": nonce.to_string()
    });

    let token = jsonwebtoken::encode(&header, &claims, &encoding_key)
        .map_err(|e| anyhow::anyhow!("JWT encode failed: {e}"))?;

    eprintln!("[coinbase_l2] CDP JWT created (ES256, kid={}...)", &api_key[..8]);
    Ok(token)
}

/// Construct a PKCS#8 v1 DER-encoded EC P-256 private key from a raw 32-byte scalar.
fn build_ec_p256_pkcs8(private_scalar: &[u8]) -> Vec<u8> {
    // PKCS#8 wrapping of SEC1 EC Private Key for P-256
    let mut der = Vec::with_capacity(67);
    // SEQUENCE (total_len = 65 = 0x41)
    der.extend_from_slice(&[0x30, 0x41]);
    // INTEGER 0 (version)
    der.extend_from_slice(&[0x02, 0x01, 0x00]);
    // SEQUENCE AlgorithmIdentifier (len=19 = 0x13)
    der.extend_from_slice(&[0x30, 0x13]);
    // OID 1.2.840.10045.2.1 (ecPublicKey)
    der.extend_from_slice(&[0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01]);
    // OID 1.2.840.10045.3.1.7 (secp256r1 / P-256)
    der.extend_from_slice(&[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07]);
    // OCTET STRING (len=39 = 0x27) wrapping SEC1 ECPrivateKey
    der.extend_from_slice(&[0x04, 0x27]);
    // SEQUENCE ECPrivateKey (len=37 = 0x25)
    der.extend_from_slice(&[0x30, 0x25]);
    // INTEGER 1 (version)
    der.extend_from_slice(&[0x02, 0x01, 0x01]);
    // OCTET STRING (len=32 = 0x20) private key
    der.extend_from_slice(&[0x04, 0x20]);
    der.extend_from_slice(private_scalar);
    der
}

/// Build an Ed25519 PKCS#8 DER key as fallback.
fn build_ed25519_pkcs8(seed: &[u8]) -> Vec<u8> {
    let mut der = Vec::with_capacity(48);
    // SEQUENCE (len=46 = 0x2e)
    der.extend_from_slice(&[0x30, 0x2e]);
    // INTEGER 0
    der.extend_from_slice(&[0x02, 0x01, 0x00]);
    // SEQUENCE AlgorithmIdentifier
    der.extend_from_slice(&[0x30, 0x05]);
    // OID 1.3.101.112 (Ed25519)
    der.extend_from_slice(&[0x06, 0x03, 0x2b, 0x65, 0x70]);
    // OCTET STRING wrapping OCTET STRING of private key seed
    der.extend_from_slice(&[0x04, 0x22, 0x04, 0x20]);
    der.extend_from_slice(seed);
    der
}

/// Try building JWT, falling back from ES256 → EdDSA → HMAC.
fn build_auth_subscribe(api_key: &str, api_secret: &str, product_id: &str) -> anyhow::Result<serde_json::Value> {
    // Try CDP JWT auth (ES256)
    match build_coinbase_jwt(api_key, api_secret) {
        Ok(jwt) => {
            return Ok(serde_json::json!({
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "level2",
                "jwt": jwt,
            }));
        }
        Err(e) => {
            eprintln!("[coinbase_l2] ES256 JWT failed: {e}, trying EdDSA...");
        }
    }

    // Fallback: try EdDSA (Ed25519) JWT
    match build_eddsa_jwt(api_key, api_secret) {
        Ok(jwt) => {
            return Ok(serde_json::json!({
                "type": "subscribe",
                "product_ids": [product_id],
                "channel": "level2",
                "jwt": jwt,
            }));
        }
        Err(e) => {
            eprintln!("[coinbase_l2] EdDSA JWT failed: {e}, trying HMAC...");
        }
    }

    // Final fallback: HMAC-SHA256 (legacy Cloud API keys)
    eprintln!("[coinbase_l2] Using HMAC-SHA256 auth (legacy)");
    let timestamp = Utc::now().timestamp().to_string();
    let message_to_sign = format!("{}level2{}", timestamp, product_id);

    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type HmacSha256 = Hmac<Sha256>;

    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
        .map_err(|e| anyhow::anyhow!("HMAC key error: {e}"))?;
    mac.update(message_to_sign.as_bytes());
    let signature = mac.finalize().into_bytes()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    Ok(serde_json::json!({
        "type": "subscribe",
        "product_ids": [product_id],
        "channel": "level2",
        "api_key": api_key,
        "timestamp": timestamp,
        "signature": signature,
    }))
}

/// Build EdDSA JWT as fallback for Ed25519 CDP keys.
fn build_eddsa_jwt(api_key: &str, api_secret: &str) -> anyhow::Result<String> {
    let key_bytes = base64::engine::general_purpose::STANDARD
        .decode(api_secret)
        .map_err(|e| anyhow::anyhow!("Failed to decode: {e}"))?;

    if key_bytes.len() < 32 {
        return Err(anyhow::anyhow!("Key too short"));
    }

    let pkcs8_der = build_ed25519_pkcs8(&key_bytes[..32]);
    let encoding_key = jsonwebtoken::EncodingKey::from_ed_der(&pkcs8_der);

    let now = Utc::now().timestamp();
    let nonce: u128 = rand::rngs::OsRng.gen();

    let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::EdDSA);
    header.kid = Some(api_key.to_string());
    header.typ = Some("JWT".to_string());

    let claims = serde_json::json!({
        "sub": api_key,
        "iss": "coinbase-cloud",
        "nbf": now,
        "exp": now + 120,
        "aud": ["cdp_service"],
        "nonce": nonce.to_string()
    });

    let token = jsonwebtoken::encode(&header, &claims, &encoding_key)
        .map_err(|e| anyhow::anyhow!("EdDSA JWT encode failed: {e}"))?;

    eprintln!("[coinbase_l2] CDP JWT created (EdDSA, kid={}...)", &api_key[..8]);
    Ok(token)
}

async fn connect_and_listen(
    state: &Arc<Mutex<AppState>>,
    product_id: &str,
    api_key: &str,
    api_secret: &str,
    asset_name: &str,
) -> anyhow::Result<()> {
    let (mut stream, _) = connect_async(WS_URL).await?;
    eprintln!("[coinbase_l2] connected to Coinbase Advanced Trade WebSocket");

    let subscribe = build_auth_subscribe(api_key, api_secret, product_id)?;
    stream.send(Message::Text(subscribe.to_string())).await?;
    eprintln!("[coinbase_l2] subscribe message sent for {product_id}");

    let mut bids: BTreeMap<OrderedFloat, f64> = BTreeMap::new();
    let mut asks: BTreeMap<OrderedFloat, f64> = BTreeMap::new();

    while let Some(msg) = stream.next().await {
        let msg = msg?;

        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            Message::Ping(data) => {
                let _ = stream.send(Message::Pong(data)).await;
                continue;
            }
            _ => continue,
        };

        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let channel = value.get("channel").and_then(|c| c.as_str()).unwrap_or("");

        // Check for error messages
        if let Some(msg_type) = value.get("type").and_then(|t| t.as_str()) {
            if msg_type == "error" {
                let error_msg = value.get("message").and_then(|m| m.as_str()).unwrap_or("unknown");
                eprintln!("[coinbase_l2] ❌ error from Coinbase: {error_msg}");
                return Err(anyhow::anyhow!("Coinbase error: {error_msg}"));
            }
        }

        match channel {
            "l2_data" => {
                if let Some(events) = value.get("events").and_then(|e| e.as_array()) {
                    for event in events {
                        let event_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");

                        if let Some(updates) = event.get("updates").and_then(|u| u.as_array()) {
                            if event_type == "snapshot" {
                                bids.clear();
                                asks.clear();
                                eprintln!("[coinbase_l2] 📸 L2 snapshot received");
                            }

                            for update in updates {
                                let side = update.get("side").and_then(|s| s.as_str()).unwrap_or("");
                                let price_str = update.get("price_level").and_then(|p| p.as_str()).unwrap_or("0");
                                let qty_str = update.get("new_quantity").and_then(|q| q.as_str()).unwrap_or("0");

                                let price: f64 = price_str.parse().unwrap_or(0.0);
                                let qty: f64 = qty_str.parse().unwrap_or(0.0);

                                if price <= 0.0 { continue; }

                                let key = OrderedFloat(price);
                                match side {
                                    "bid" => {
                                        if qty <= 0.0 { bids.remove(&key); }
                                        else { bids.insert(key, qty); }
                                    }
                                    "offer" => {
                                        if qty <= 0.0 { asks.remove(&key); }
                                        else { asks.insert(key, qty); }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    if let (Some((&best_bid_key, _)), Some((&best_ask_key, _))) =
                        (bids.iter().next_back(), asks.iter().next())
                    {
                        let best_bid = best_bid_key.0;
                        let best_ask = best_ask_key.0;
                        let mid_price = (best_bid + best_ask) / 2.0;
                        update_state(state, mid_price, best_bid, best_ask, asset_name);
                    }
                }
            }
            "subscriptions" => {
                eprintln!("[coinbase_l2] ✅ subscription confirmed");
            }
            "heartbeats" => {}
            _ => {
                // Log unknown channels for debugging
                if !channel.is_empty() {
                    eprintln!("[coinbase_l2] unknown channel: {channel}");
                }
            }
        }
    }

    Ok(())
}

fn update_state(state: &Arc<Mutex<AppState>>, mid_price: f64, best_bid: f64, best_ask: f64, asset_name: &str) {
    let now = Utc::now();
    let mut app = state.lock().unwrap();

    app.coinbase_l2 = Some(CoinbaseL2State {
        mid_price,
        best_bid,
        best_ask,
        ts: now,
    });

    // Update the stateful EWMA tracker
    let lambda = std::env::var("EWMA_LAMBDA").map_or(0.97, |v| v.parse().unwrap_or(0.97));
    let tracker = app.multi_volatility_trackers
        .entry(asset_name.to_string())
        .or_insert_with(|| crate::black_scholes::EwmaVolatilityTracker::new(lambda));

    tracker.update(mid_price, now);

    app.volatility_window.push_back((now, mid_price));
    while app.volatility_window.len() > MAX_VOL_WINDOW {
        app.volatility_window.pop_front();
    }

    let cutoff_5min = now - TimeDelta::seconds(300); // 5 minute window
    while let Some(&(ts, _)) = app.volatility_window.front() {
        if ts < cutoff_5min { app.volatility_window.pop_front(); }
        else { break; }
    }

    // Also update the 60-second price_window for spike detection
    app.price_window.push_back((now, mid_price));
    let cutoff_60s = now - TimeDelta::seconds(60);
    while let Some(&(ts, _)) = app.price_window.front() {
        if ts < cutoff_60s { app.price_window.pop_front(); }
        else { break; }
    }
}

pub fn is_l2_stale(state: &Arc<Mutex<AppState>>) -> bool {
    let app = state.lock().unwrap();
    match &app.coinbase_l2 {
        Some(l2) => (Utc::now() - l2.ts).num_seconds() > STALE_THRESHOLD_SECS,
        None => true,
    }
}

#[derive(Debug, Clone, Copy)]
struct OrderedFloat(f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}
impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}
