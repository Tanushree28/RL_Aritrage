use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use base64::Engine;
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use rand::rngs::OsRng;
use rand::Rng;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::RsaPrivateKey;
use sha2::Sha256;
use tokio::time::Duration;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use crate::model::{AppState, AssetConfig, MarketState};

// Production endpoint resolves in most environments; the bot falls back to
// the demo endpoint automatically if the production DNS lookup fails.
const WS_URL_PROD: &str = "wss://api.elections.kalshi.com/trade-api/ws/v2";
const WS_URL_DEMO: &str = "wss://demo-api.kalshi.co/trade-api/ws/v2";
const WS_PATH: &str = "/trade-api/ws/v2";

/// Entry point. Parses the RSA private key once, then enters a reconnect loop.
/// On first failure it tries production; if that fails with a DNS/connect error
/// it switches to the demo endpoint for the remainder of the session.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[kalshi_listener] task started (asset: {}, series: {})", config.name, config.kalshi_series);

    eprintln!("[kalshi_listener] loading vars...");
    let api_key_id = std::env::var("KALSHI_API_KEY_ID")?;
    let private_key_raw = std::env::var("KALSHI_PRIVATE_KEY")?;
    let private_key_pem = private_key_raw.replace("\\n", "\n");
    
    eprintln!("[kalshi_listener] parsing private key...");
    let private_key = match RsaPrivateKey::from_pkcs1_pem(&private_key_pem) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[kalshi_listener] FATAL: Failed to parse KALSHI_PRIVATE_KEY: {e}");
            eprintln!("[kalshi_listener] Key content (first 50 chars): {:.50}...", private_key_pem);
            std::process::exit(1);
        }
    };
    
    eprintln!("[kalshi_listener] creating signing key...");
    let signing_key = SigningKey::<Sha256>::new(private_key);

    eprintln!("[kalshi_listener] initialization complete, entering loop...");
    let mut backoff_secs: u64 = 1;
    let mut use_demo = config.api_base_url.contains("demo");

    loop {
        let url = if use_demo { WS_URL_DEMO } else { WS_URL_PROD };
        eprintln!("[kalshi_listener] connecting to {url}...");

        match connect_and_listen(&state, &api_key_id, &signing_key, url, &config.kalshi_series).await {
            Ok(()) => {
                eprintln!("[kalshi_listener] stream ended, reconnecting...");
                backoff_secs = 1;
            }
            Err(e) => {
                let err_str = e.to_string();

                // Auto-fallback: if production is unreachable, switch to demo.
                if !use_demo
                    && (err_str.contains("lookup") || err_str.contains("connect"))
                {
                    eprintln!("[kalshi_listener] production unreachable, falling back to demo");
                    use_demo = true;
                    continue;
                }

                eprintln!("[kalshi_listener] error: {e} — retrying in {backoff_secs}s");
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(60);
            }
        }

        if state.lock().unwrap().shutdown {
            eprintln!("[kalshi_listener] shutdown requested, exiting");
            return Ok(());
        }
    }
}

/// Signs the WebSocket upgrade request using RSA-PSS with SHA-256.
///
/// Kalshi's signature message is `"{timestamp_ms}GET{path}"`. The
/// `SigningKey` hashes internally, so we pass the raw UTF-8 bytes.
/// `Pss::new::<Sha256>()` (used by SigningKey) sets salt_len = 32 (SHA-256
/// digest size), which matches Python's `PSS.DIGEST_LENGTH`.
fn sign_handshake(signing_key: &SigningKey<Sha256>, timestamp_ms: u64) -> String {
    let message = format!("{}GET{}", timestamp_ms, WS_PATH);
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(signature.to_vec())
}

/// Opens the authenticated WebSocket, subscribes to the ticker channel, and
/// processes incoming messages until the connection drops.
async fn connect_and_listen(
    state: &Arc<Mutex<AppState>>,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    ws_url: &str,
    kalshi_series: &str,
) -> anyhow::Result<()> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_handshake(signing_key, timestamp_ms);

    // tokio-tungstenite passes http::Request through verbatim; we must include
    // the standard WebSocket upgrade headers ourselves.
    let mut ws_key_bytes = [0u8; 16];
    OsRng.fill(&mut ws_key_bytes);
    let ws_key = base64::engine::general_purpose::STANDARD.encode(&ws_key_bytes);

    let uri: http::Uri = ws_url.parse()?;
    let host = uri.authority().ok_or_else(|| anyhow::anyhow!("no host in URL"))?.as_str();
    let scheme = uri.scheme_str().unwrap_or("wss");
    let rest_scheme = if scheme == "wss" { "https" } else { "http" };
    let rest_url = format!("{}://{}", rest_scheme, host);

    eprintln!("[kalshi_listener] resolved REST URL for ticker polling: {rest_url}");

    let request = http::Request::builder()
        .uri(ws_url)
        .header("Host", host)
        .header("Connection", "Upgrade")
        .header("Upgrade", "websocket")
        .header("Sec-WebSocket-Version", "13")
        .header("Sec-WebSocket-Key", &ws_key)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &signature)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .body(())?;

    eprintln!("[kalshi_listener] attempting WS connection to {ws_url}...");
    let (mut stream, _) = connect_async(request).await?;
    eprintln!("[kalshi_listener] connected to {ws_url}");

    // Broad subscription — catches KXBTCD and any other tickers in the default broadcast.
    let subscribe = serde_json::json!({
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": ["ticker"]
        }
    });
    stream.send(Message::Text(subscribe.to_string())).await?;

    let rest_client = reqwest::Client::new();
    let mut sub_id: u64 = 2;

    // 15M markets are not included in the default broadcast; discover and
    // subscribe explicitly.  New 15-min windows open every 15 minutes, so we
    // refresh this subscription periodically inside the message loop.
    let market_tickers = fetch_active_tickers(&rest_client, api_key_id, signing_key, kalshi_series, &rest_url).await;
    if !market_tickers.is_empty() {
        eprintln!("[kalshi_listener] subscribing to {} {} markets", market_tickers.len(), kalshi_series);
        let sub = serde_json::json!({
            "id": sub_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": market_tickers,
                "send_initial_snapshot": true
            }
        });
        stream.send(Message::Text(sub.to_string())).await?;
        sub_id += 1;
    }

    // Cache: market_ticker → (floor_strike, close_time).
    // One REST call per market; all subsequent ticks reuse the cached values.
    let mut event_cache: HashMap<String, (f64, chrono::DateTime<chrono::Utc>)> = HashMap::new();
    let mut last_15m_refresh = tokio::time::Instant::now();

    while let Some(msg) = stream.next().await {
        let msg = msg?;

        // Refresh 15M subscription every 15 minutes to pick up new windows.
        let now = tokio::time::Instant::now();
        if now.duration_since(last_15m_refresh) >= Duration::from_secs(900) {
            let tickers = fetch_active_tickers(&rest_client, api_key_id, signing_key, kalshi_series, &rest_url).await;
            if !tickers.is_empty() {
                eprintln!("[kalshi_listener] refreshing {} subscription: {} markets", kalshi_series, tickers.len());
                let sub = serde_json::json!({
                    "id": sub_id,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["ticker"],
                        "market_tickers": tickers,
                        "send_initial_snapshot": true
                    }
                });
                stream.send(Message::Text(sub.to_string())).await?;
                sub_id += 1;
            }
            last_15m_refresh = now;
        }

        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Only act on ticker-type messages; ignore subscribe acks, errors, etc.
        let msg_type = value.get("type").and_then(|t| t.as_str()).unwrap_or("?");
        if msg_type != "ticker" {
            continue;
        }

        let msg_data = match value.get("msg") {
            Some(d) => d,
            None    => continue,
        };

        let market_ticker = match msg_data.get("market_ticker").and_then(|t| t.as_str()) {
            Some(t) => t,
            None    => continue,
        };

        // Only accept 15-minute contracts for the configured asset
        //   KXBTC15M-26FEB010915-15   (BTC 15-min)
        //   KXETH15M-26FEB010915-15   (ETH 15-min)
        let series_prefix = format!("{}-", kalshi_series);
        if !market_ticker.starts_with(&series_prefix) {
            continue;
        }

        // One REST round-trip per market to get floor_strike + close_time.
        // Cached by market_ticker because each KXBTCD market has its own strike.
        if !event_cache.contains_key(market_ticker) {
            match fetch_event_strike(&rest_client, api_key_id, signing_key, market_ticker, &rest_url).await {
                Some(info) => {
                    eprintln!(
                        "[kalshi_listener] market {} floor_strike={:.2} close={}",
                        market_ticker, info.0, info.1
                    );
                    event_cache.insert(market_ticker.to_string(), info);
                }
                None => continue,   // not ready or error already logged
            }
        }

        let &(strike, expiry) = event_cache.get(market_ticker).unwrap();
        process_ticker(state, msg_data, market_ticker, strike, expiry);
    }

    Ok(())
}

/// Fetches `floor_strike` and `close_time` for a market from the Kalshi REST
/// API.  Called once per event_ticker; the result is cached for the lifetime
/// of the WebSocket connection.
async fn fetch_event_strike(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    market_ticker: &str,
    base_url: &str,
) -> Option<(f64, chrono::DateTime<chrono::Utc>)> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let path = format!("/trade-api/v2/markets/{}", market_ticker);
    let message = format!("{}GET{}", timestamp_ms, path);
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_vec());

    let resp = match client
        .get(format!("{}{}", base_url, path))
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &sig_b64)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .send()
        .await
    {
        Ok(r)  => r,
        Err(e) => { eprintln!("[kalshi_listener] REST send error for {market_ticker}: {e}"); return None; }
    };

    let status = resp.status();
    let body = match resp.text().await {
        Ok(t)  => t,
        Err(e) => { eprintln!("[kalshi_listener] REST body read error for {market_ticker}: {e}"); return None; }
    };

    if !status.is_success() {
        eprintln!("[kalshi_listener] REST {status} for {market_ticker}: {body}");
        return None;
    }

    let data: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v)  => v,
        Err(e) => { eprintln!("[kalshi_listener] REST JSON parse error for {market_ticker}: {e}"); return None; }
    };
    let market = match data.get("market") {
        Some(m) => m,
        None    => { eprintln!("[kalshi_listener] REST no 'market' key for {market_ticker}: {body}"); return None; }
    };

    let floor_strike = match market.get("floor_strike").and_then(|v| v.as_f64()) {
        Some(fs) => fs,
        None     => return None,   // reference window not yet complete; retry next tick
    };
    if floor_strike <= 0.0 {
        eprintln!("[kalshi_listener] REST floor_strike={floor_strike} <= 0 for {market_ticker}");
        return None;
    }

    let close_time_str = match market.get("close_time").and_then(|v| v.as_str()) {
        Some(s) => s,
        None    => { eprintln!("[kalshi_listener] REST no close_time for {market_ticker}"); return None; }
    };
    let close_time = match chrono::DateTime::parse_from_rfc3339(close_time_str) {
        Ok(dt) => dt.with_timezone(&chrono::Utc),
        Err(e) => { eprintln!("[kalshi_listener] REST close_time parse error for {market_ticker}: {e}"); return None; }
    };

    Some((floor_strike, close_time))
}

/// Fetches the settlement result for a market from the Kalshi REST API.
/// Returns `Some((status, result))` where:
///   - `status`: "active", "closed", "finalized", "settled"
///   - `result`: Some("yes") or Some("no") if settled, None otherwise
/// Returns `None` on API error.
pub async fn fetch_market_result(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    market_ticker: &str,
    base_url: &str,
) -> Option<(String, Option<String>)> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let path = format!("/trade-api/v2/markets/{}", market_ticker);
    let message = format!("{}GET{}", timestamp_ms, path);
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_vec());

    let resp = match client
        .get(format!("{}{}", base_url, path))
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &sig_b64)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .send()
        .await
    {
        Ok(r)  => r,
        Err(e) => { eprintln!("[kalshi_listener] fetch_market_result send error for {market_ticker}: {e}"); return None; }
    };

    let http_status = resp.status();
    let body = match resp.text().await {
        Ok(t)  => t,
        Err(e) => { eprintln!("[kalshi_listener] fetch_market_result body error for {market_ticker}: {e}"); return None; }
    };

    if !http_status.is_success() {
        eprintln!("[kalshi_listener] fetch_market_result {http_status} for {market_ticker}: {body}");
        return None;
    }

    let data: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v)  => v,
        Err(e) => { eprintln!("[kalshi_listener] fetch_market_result JSON error for {market_ticker}: {e}"); return None; }
    };

    let market = data.get("market")?;

    // Extract status (e.g., "active", "closed", "finalized", "settled")
    let status = market.get("status").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();

    // Extract result if available (e.g., "yes", "no", "void")
    let result = market.get("result").and_then(|v| v.as_str()).map(|s| s.to_string());

    Some((status, result))
}

/// Writes the latest BTC 15-min tick into shared state.
/// Strike and expiry come from the REST-cached event data.
/// Prices from the WebSocket are in cents (0–100); normalised to 0.0–1.0.
fn process_ticker(
    state:          &Arc<Mutex<AppState>>,
    msg:            &serde_json::Value,
    market_ticker:  &str,
    strike:         f64,
    expiry:         chrono::DateTime<chrono::Utc>,
) {
    let yes_bid = msg.get("yes_bid").and_then(|v| v.as_f64()).unwrap_or(0.0) / 100.0;
    let yes_ask = msg.get("yes_ask").and_then(|v| v.as_f64()).unwrap_or(0.0) / 100.0;
    let no_bid = msg
        .get("no_bid")
        .and_then(|v| v.as_f64())
        .map(|v| v / 100.0)
        .unwrap_or(1.0 - yes_ask);
    let no_ask = msg
        .get("no_ask")
        .and_then(|v| v.as_f64())
        .map(|v| v / 100.0)
        .unwrap_or(1.0 - yes_bid);

    eprintln!(
        "[kalshi_listener] {} strike={strike:.2} yes_ask={yes_ask:.2} yes_bid={yes_bid:.2}",
        market_ticker
    );

    state.lock().unwrap().kalshi = Some(MarketState {
        ticker:    market_ticker.to_string(),
        strike,
        yes_ask,
        no_ask,
        yes_bid,
        no_bid,
        expiry_ts: Some(expiry),
        ts:        Utc::now(),
    });
}

/// Queries `GET /trade-api/v2/markets?series_ticker={series}&limit=1000` and
/// returns the market_ticker strings for markets whose close_time is in the
/// future.  On any error the function logs and returns an empty vec so the
/// listener keeps running.
async fn fetch_active_tickers(
    client:      &reqwest::Client,
    api_key_id:  &str,
    signing_key: &SigningKey<Sha256>,
    series:      &str,
    base_url:    &str,
) -> Vec<String> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    eprintln!("[kalshi_listener] fetching active tickers for series '{}' from {}", series, base_url);

    let path = format!("/trade-api/v2/markets?series_ticker={}&limit=1000", series);
    let message = format!("{}GET{}", timestamp_ms, path);
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_vec());

    let resp = match client
        .get(format!("{}{}", base_url, path))
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &sig_b64)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .timeout(Duration::from_secs(10))
        .send()
        .await
    {
        Ok(r)  => r,
        Err(e) => { eprintln!("[kalshi_listener] REST list {} error: {e}", series); return vec![]; }
    };

    let status = resp.status();
    let body = match resp.text().await {
        Ok(t)  => t,
        Err(e) => { eprintln!("[kalshi_listener] REST list {} body error: {e}", series); return vec![]; }
    };

    if !status.is_success() {
        eprintln!("[kalshi_listener] REST list {} {status}: {body}", series);
        return vec![];
    }

    let data: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v)  => v,
        Err(e) => { eprintln!("[kalshi_listener] REST list {} JSON error: {e}", series); return vec![]; }
    };

    let markets = match data.get("markets").and_then(|m| m.as_array()) {
        Some(arr) => arr,
        None      => { eprintln!("[kalshi_listener] REST list {}: no 'markets' array", series); return vec![]; }
    };

    let now = Utc::now();
    let mut tickers = Vec::new();
    let mut skipped_closed = 0usize;
    for m in markets {
        let ticker = match m.get("ticker").and_then(|t| t.as_str()) {
            Some(t) => t,
            None    => continue,
        };
        // Skip markets whose close_time is already past.
        if let Some(close_str) = m.get("close_time").and_then(|c| c.as_str()) {
            if let Ok(close_dt) = chrono::DateTime::parse_from_rfc3339(close_str) {
                if close_dt <= now {
                    skipped_closed += 1;
                    continue;
                }
            }
        }
        tickers.push(ticker.to_string());
    }

    eprintln!(
        "[kalshi_listener] {} market discovery: {} active, {} skipped (closed), {} total in response",
        series, tickers.len(), skipped_closed, markets.len()
    );
    tickers
}
