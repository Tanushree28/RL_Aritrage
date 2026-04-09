//! Order lifecycle management for the Oracle bot.
//!
//! Handles order cancellation, amendment (cancel + replace), and
//! emergency cancellation when fair value drops below limit price.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use chrono::{DateTime, Utc};
use rand::rngs::OsRng;
use rand::Rng;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use sha2::Sha256;

use crate::model::{ActiveOrder, AppState, MarketState};

const REST_PATH: &str = "/trade-api/v2/portfolio/orders";
const BASE_URL_PROD: &str = "https://api.elections.kalshi.com";

/// Minimum price change (in dollars) before triggering an amendment.
const AMEND_THRESHOLD: f64 = 0.01;

/// Minimum seconds between amendments for the same ticker.
const AMEND_COOLDOWN_SECS: i64 = 2;

/// Maximum number of post_only retry attempts.
const MAX_POST_ONLY_RETRIES: u32 = 3;

/// Post-only retry step: reduce limit by this much on each retry.
const POST_ONLY_RETRY_STEP: f64 = 0.01;

// ---------------------------------------------------------------------------
// Order Cancellation
// ---------------------------------------------------------------------------

/// Cancel an order on Kalshi via DELETE endpoint.
pub async fn cancel_order(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    order_id: &str,
) -> anyhow::Result<()> {
    let path = format!("{}/{}", REST_PATH, order_id);
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_request(signing_key, timestamp_ms, "DELETE", &path);
    let url = format!("{base_url}{path}");

    let resp = client
        .delete(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &signature)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .send()
        .await?;

    let status = resp.status();
    if status.is_success() || status == reqwest::StatusCode::NO_CONTENT {
        eprintln!("[order_mgr] ✅ cancelled order {order_id}");
        Ok(())
    } else {
        let text = resp.text().await?;
        Err(anyhow::anyhow!("Cancel failed {status}: {text}"))
    }
}

// ---------------------------------------------------------------------------
// Order Amendment (Cancel + Replace)
// ---------------------------------------------------------------------------

/// Amend an active order by cancelling and placing a new one.
/// Returns the new order_id if successful.
pub async fn amend_order(
    state: &Arc<Mutex<AppState>>,
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    ticker: &str,
    kalshi: &MarketState,
    new_limit_price: f64,
    side: &str,
    count: u64,
    paper_mode: bool,
) -> anyhow::Result<Option<String>> {
    // Get the current active order for this ticker
    let current_order = {
        let app = state.lock().unwrap();
        app.active_oracle_orders.get(ticker).cloned()
    };

    let current_order = match current_order {
        Some(o) => o,
        None => return Ok(None),
    };

    if paper_mode {
        // Paper mode: just update the limit price
        let now = Utc::now();
        let mut app = state.lock().unwrap();
        if let Some(order) = app.active_oracle_orders.get_mut(ticker) {
            let old_price = order.limit_price;
            order.limit_price = new_limit_price;
            order.last_amend = now;
            eprintln!(
                "[order_mgr] ♻️ PAPER amended {} ${:.2} → ${:.2}",
                ticker, old_price, new_limit_price
            );
        }
        return Ok(Some(current_order.order_id));
    }

    // Cancel the existing order
    cancel_order(client, api_key_id, signing_key, base_url, &current_order.order_id).await?;

    // Remove from active orders
    {
        let mut app = state.lock().unwrap();
        app.active_oracle_orders.remove(ticker);
    }

    // Place new order with updated limit
    let new_order_id = place_oracle_order(
        client, api_key_id, signing_key, base_url, kalshi, side, new_limit_price, count,
    ).await?;

    // Track the new order
    let now = Utc::now();
    {
        let mut app = state.lock().unwrap();
        app.active_oracle_orders.insert(ticker.to_string(), ActiveOrder {
            order_id: new_order_id.clone(),
            ticker: ticker.to_string(),
            side: side.to_string(),
            limit_price: new_limit_price,
            count,
            placed_at: now,
            last_amend: now,
        });
    }

    eprintln!(
        "[order_mgr] ♻️ amended {} → ${:.2} (new id: {})",
        ticker, new_limit_price, new_order_id
    );

    Ok(Some(new_order_id))
}

// ---------------------------------------------------------------------------
// Decision Functions
// ---------------------------------------------------------------------------

/// Returns true if the order should be amended (price moved enough, cooldown passed).
pub fn should_amend(current_limit: f64, new_limit: f64, last_amend: DateTime<Utc>) -> bool {
    // Only amend UPWARDS (never downwards, otherwise orders run away from fills)
    let delta = new_limit - current_limit;
    let elapsed = (Utc::now() - last_amend).num_seconds();

    delta >= AMEND_THRESHOLD && elapsed >= AMEND_COOLDOWN_SECS
}

/// Returns true if the order should be emergency-cancelled (fair value dropped below limit).
pub fn should_emergency_cancel(fair_value: f64, limit_price: f64) -> bool {
    fair_value < limit_price
}

// ---------------------------------------------------------------------------
// Order Placement (with post_only)
// ---------------------------------------------------------------------------

/// Place a limit order on Kalshi with post_only flag.
pub async fn place_oracle_order(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    kalshi: &MarketState,
    side: &str,
    limit_price: f64,
    count: u64,
) -> anyhow::Result<String> {
    let mut current_limit = limit_price;

    for attempt in 0..MAX_POST_ONLY_RETRIES {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let signature = sign_request(signing_key, timestamp_ms, "POST", REST_PATH);
        let price_cents = (current_limit * 100.0).round() as i64;

        let mut body = serde_json::json!({
            "ticker": kalshi.ticker,
            "action": "buy",
            "side": side,
            "count": count,
            "type": "limit",
            "post_only": true,
            "client_order_id": client_order_id(),
        });

        let price_key = if side == "yes" { "yes_price" } else { "no_price" };
        body[price_key] = serde_json::json!(price_cents);

        let url = format!("{base_url}{REST_PATH}");

        let resp = client
            .post(&url)
            .header("KALSHI-ACCESS-KEY", api_key_id)
            .header("KALSHI-ACCESS-SIGNATURE", &signature)
            .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if status == reqwest::StatusCode::CREATED {
            let order_json: serde_json::Value = resp.json().await?;
            let order_obj = order_json.get("order").unwrap_or(&order_json);
            let order_id = order_obj
                .get("order_id")
                .and_then(|id| id.as_str())
                .unwrap_or("unknown")
                .to_string();
            return Ok(order_id);
        } else {
            let text = resp.text().await?;
            // Check for post_only rejection (would have crossed the spread)
            if text.contains("post_only") || text.contains("would_cross") {
                eprintln!(
                    "[order_mgr] post_only rejected at ${:.2}, retrying ${:.2} (attempt {}/{})",
                    current_limit, current_limit - POST_ONLY_RETRY_STEP,
                    attempt + 1, MAX_POST_ONLY_RETRIES
                );
                current_limit -= POST_ONLY_RETRY_STEP;
                continue;
            }
            return Err(anyhow::anyhow!("Place order failed {status}: {text}"));
        }
    }

    Err(anyhow::anyhow!("post_only rejected after {} attempts", MAX_POST_ONLY_RETRIES))
}

/// Place a market order on Kalshi (fills immediately at best price).
pub async fn place_market_order(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    ticker: &str,
    side: &str,
    count: u64,
) -> anyhow::Result<String> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_request(signing_key, timestamp_ms, "POST", REST_PATH);

    let body = serde_json::json!({
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": count,
        "type": "market",
        "client_order_id": client_order_id(),
    });

    let url = format!("{base_url}{REST_PATH}");

    let resp = client
        .post(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &signature)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if status == reqwest::StatusCode::CREATED {
        let order_json: serde_json::Value = resp.json().await?;
        let order_obj = order_json.get("order").unwrap_or(&order_json);
        let order_id = order_obj
            .get("order_id")
            .and_then(|id| id.as_str())
            .unwrap_or("unknown")
            .to_string();
        Ok(order_id)
    } else {
        let text = resp.text().await?;
        Err(anyhow::anyhow!("Place market order failed {status}: {text}"))
    }
}

/// Place an "aggressive" limit order (priced at the ask) for immediate fill.
pub async fn place_aggressive_limit_order(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    ticker: &str,
    side: &str,
    limit_price: f64,
    count: u64,
) -> anyhow::Result<serde_json::Value> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_request(signing_key, timestamp_ms, "POST", REST_PATH);
    let price_cents = (limit_price * 100.0).round() as i64;

    let mut body = serde_json::json!({
        "ticker": ticker,
        "action": "buy",
        "side": side,
        "count": count,
        "type": "limit",
        "post_only": false, // Allow crossing the spread
        "client_order_id": client_order_id(),
    });

    let price_key = if side == "yes" { "yes_price" } else { "no_price" };
    body[price_key] = serde_json::json!(price_cents);

    let url = format!("{base_url}{REST_PATH}");

    let resp = client
        .post(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &signature)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if status == reqwest::StatusCode::CREATED {
        let order_json: serde_json::Value = resp.json().await?;
        let order_obj = order_json.get("order").unwrap_or(&order_json).clone();
        Ok(order_obj)
    } else {
        let text = resp.text().await?;
        Err(anyhow::anyhow!("Place aggressive limit order failed {status}: {text}"))
    }
}

/// Fetch full order information from Kalshi.
pub async fn fetch_order_status(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    order_id: &str,
) -> anyhow::Result<serde_json::Value> {
    let path = format!("{}/{}", REST_PATH, order_id);
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_request(signing_key, timestamp_ms, "GET", &path);
    let url = format!("{base_url}{path}");

    let resp = client
        .get(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &signature)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .send()
        .await?;

    if resp.status().is_success() {
        let order_json: serde_json::Value = resp.json().await?;
        let order_obj = order_json.get("order").unwrap_or(&order_json).clone();
        Ok(order_obj)
    } else {
        let status = resp.status();
        let text = resp.text().await?;
        Err(anyhow::anyhow!("Fetch order failed {status}: {text}"))
    }
}

fn sign_request(
    signing_key: &SigningKey<Sha256>,
    timestamp_ms: u64,
    method: &str,
    path: &str,
) -> String {
    let message = format!("{timestamp_ms}{method}{path}");
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(signature.to_vec())
}

fn client_order_id() -> String {
    let mut rng = OsRng;
    let a: u32 = rng.gen();
    let b: u16 = rng.gen();
    let c: u16 = (rng.gen::<u16>() & 0x0fff) | 0x4000;
    let d: u16 = (rng.gen::<u16>() & 0x3fff) | 0x8000;
    let e: u64 = rng.gen::<u64>() & 0x0000_ffff_ffff_ffff;
    format!("{a:08x}-{b:04x}-{c:04x}-{d:04x}-{e:012x}")
}
