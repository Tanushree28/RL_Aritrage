//! Bucket 9 Maximizer Trade Executor
//!
//! ONLY trades bucket 9 (price $0.90-$0.99) with NO position limits.
//! Tests the hypothesis that taking unlimited high-confidence trades
//! is more profitable than the 2-trade limit in the main bot.
//!
//! Key differences from main executor:
//! - Only trades bucket 9 (entry price >= $0.90 and < $1.00)
//! - NO position limits (can take unlimited same-side positions)
//! - NO MIN_PRICE_DELTA check (can take multiple at same price)
//! - KEEPS opposite-direction blocking (no YES+NO in same market)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use base64::Engine;
use chrono::Utc;
use rand::rngs::OsRng;
use rand::Rng;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use sha2::Sha256;
use tokio::time::Duration;

use crate::model::{AppState, AssetConfig, MarketState, MarketTracker, ObsSummary, OpenPosition, TradeRecord};
use crate::telegram_reporter;
use crate::kalshi_listener;

// ============================================================
// BUCKET 9 STRATEGY PARAMETERS
// ============================================================

/// Minimum entry price for bucket 9
const MIN_BUCKET9_PRICE: f64 = 0.90;

/// Maximum entry price for bucket 9 (exclusive)
const MAX_BUCKET9_PRICE: f64 = 1.00;

/// Minimum seconds to expiry for entry
const MIN_SECS_TO_EXPIRY: i64 = 30;

/// Maximum seconds to expiry for entry (5 minutes)
const MAX_SECS_TO_EXPIRY: i64 = 300;

/// Maximum contracts per position (safety cap)
const MAX_CONTRACTS_PER_POSITION: u64 = 1000;

// ============================================================
// KALSHI API
// ============================================================

const REST_PATH: &str = "/trade-api/v2/portfolio/orders";
const BASE_URL_PROD: &str = "https://api.elections.kalshi.com";
const BASE_URL_DEMO: &str = "https://demo-api.kalshi.co";

/// Kalshi taker fee on an order of `count` contracts at `price` (0–1).
fn order_fee(count: u64, price: f64) -> f64 {
    (0.07 * count as f64 * price * (1.0 - price) * 100.0).ceil() / 100.0
}

/// Check if price is in bucket 9 (asset-specific minimum to $0.99)
fn is_bucket_9(price: f64, asset: &str) -> bool {
    let min_price = get_min_price_for_asset(asset);
    price >= min_price && price < MAX_BUCKET9_PRICE
}

/// Get max seconds to expiry for asset-specific optimization
/// XRP: 180s (based on analysis showing 100% win rate in 120-180s window)
/// Others: 300s (default)
fn get_max_secs_for_asset(asset: &str) -> i64 {
    match asset.to_lowercase().as_str() {
        // ETH: $0.98 min price alone gives 100% win rate, can use 300s
        "eth" | "bucket9-eth" => 300,
        // BTC/SOL/XRP: need tighter 180s window
        _ => 180,
    }
}

/// Get minimum entry price for asset-specific optimization
/// Based on analysis: lower prices have higher loss rates
/// - BTC: $0.94+ = 100% win rate (losses at $0.90-0.94)
/// - ETH: $0.98+ = 100% win rate (losses at $0.90-0.98)
/// - SOL: $0.94+ = 100% win rate (losses at $0.90-0.94)
/// - XRP: $0.96+ = better performance
fn get_min_price_for_asset(asset: &str) -> f64 {
    match asset.to_lowercase().as_str() {
        "btc" | "bucket9-btc" => 0.94,  // BTC: Relaxed to $0.94 for more trades
        "eth" | "bucket9-eth" => 0.98,  // ETH: Keep strict at $0.98
        "sol" | "bucket9-sol" => 0.90,  // SOL: Relaxed to $0.90
        "xrp" | "bucket9-xrp" => 0.90,  // XRP: Relaxed to $0.90
        _ => MIN_BUCKET9_PRICE,         // Default: $0.90
    }
}

/// Entry point for the bucket 9 trade executor.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[BUCKET9] task started (asset: {})", config.name);
    let max_secs = get_max_secs_for_asset(&config.name);
    let min_price = get_min_price_for_asset(&config.name);
    eprintln!("[BUCKET9] Strategy: ONLY bucket 9 (${:.2}-${:.2}), NO position limits",
              min_price, MAX_BUCKET9_PRICE);
    eprintln!("[BUCKET9] Time window: {}s - {}s to expiry", MIN_SECS_TO_EXPIRY, max_secs);

    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);
    let bet_size: f64 = std::env::var("BET_SIZE")
        .unwrap_or_else(|_| "50".into())  // Smaller default since we take more trades
        .parse()
        .unwrap_or(50.0);

    if paper_mode {
        eprintln!("[BUCKET9] PAPER MODE — fills simulated, no orders sent");
    }

    let api_key_id = std::env::var("KALSHI_API_KEY_ID")?;
    let private_key_pem = std::env::var("KALSHI_PRIVATE_KEY")?;
    let private_key = RsaPrivateKey::from_pkcs1_pem(&private_key_pem)
        .context("Failed to parse KALSHI_PRIVATE_KEY as PKCS#1 PEM")?;
    let signing_key = SigningKey::<Sha256>::new(private_key);
    let client = reqwest::Client::new();

    let mut base_url: String = config.api_base_url.clone();

    // Initialize paper-mode capital
    if paper_mode {
        state.lock().unwrap().available_capital = starting_capital;
        eprintln!("[BUCKET9] capital=${starting_capital:.2}  bet_size=${bet_size:.2}");
    }

    let mut market_trackers: HashMap<String, MarketTracker> = HashMap::new();
    let mut last_tracker_cleanup: chrono::DateTime<chrono::Utc> = Utc::now();
    let mut last_kalshi_ts: Option<chrono::DateTime<chrono::Utc>> = None;

    // Track trades per ticker to avoid duplicate same-tick entries
    let mut last_trade_time: HashMap<String, chrono::DateTime<chrono::Utc>> = HashMap::new();

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[BUCKET9] shutdown requested, exiting");
            return Ok(());
        }

        // Snapshot shared state
        let (kalshi, coinbase, cb_price, binance_price) = {
            let app = state.lock().unwrap();
            (
                app.kalshi.clone(), 
                app.coinbase.clone(), 
                app.coinbase_price, 
                app.binance_price
            )
        };

        let (kalshi, coinbase) = match (kalshi, coinbase) {
            (Some(k), Some(c)) => (k, c),
            _ => continue, // wait for at least Kalshi + Kraken
        };

        // Hybrid Price: Average of all available sources
        let mut sum = 0.0;
        let mut count = 0.0;

        // Source 1: Kraken (primary WebSocket)
        if coinbase.current_price > 0.0 {
            sum += coinbase.current_price;
            count += 1.0;
        }
        // Source 2: Coinbase (REST)
        if let Some(p) = cb_price {
             if p > 0.0 { sum += p; count += 1.0; }
        }
        // Source 3: Binance/Bybit (WebSocket)
        if let Some(p) = binance_price {
             if p > 0.0 { sum += p; count += 1.0; }
        }

        let spot = if count > 0.0 {
            sum / count
        } else {
            coinbase.current_price // Fallback if others fail
        };

        // Settle expired paper positions
        if paper_mode {
            settle_expired(&state, &client, &api_key_id, &signing_key).await;
        }

        // Only trade 15-min markets for BTC
        let series_prefix = format!("{}-", config.kalshi_series);
        if !kalshi.ticker.starts_with(&series_prefix) {
            continue;
        }

        // Push hybrid price into all active trackers
        market_trackers.entry(kalshi.ticker.clone()).or_insert_with(MarketTracker::new);
        for tracker in market_trackers.values_mut() {
            tracker.price_history.push_back((Utc::now(), spot));
            if tracker.price_history.len() > 600 {
                tracker.price_history.pop_front();
            }
        }

        // Periodic cleanup of stale trackers
        if (Utc::now() - last_tracker_cleanup).num_seconds() > 60 {
            let now = Utc::now();
            market_trackers.retain(|_, tracker| {
                tracker.price_history.back()
                    .map_or(false, |(ts, _)| (now - *ts).num_seconds() < 1200)
            });
            last_trade_time.retain(|t, _| market_trackers.contains_key(t));
            last_tracker_cleanup = now;
        }

        // Get seconds to expiry
        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts) as i64;

        // Skip if expired or too close to expiry
        if secs_left < MIN_SECS_TO_EXPIRY {
            market_trackers.remove(&kalshi.ticker);
            last_trade_time.remove(&kalshi.ticker);
            continue;
        }

        // Skip if too far from expiry (asset-specific)
        let max_secs = get_max_secs_for_asset(&config.name);
        if secs_left > max_secs {
            continue;
        }

        // Freeze ObsSummary on first tick in trading phase
        {
            let tracker = market_trackers.get_mut(&kalshi.ticker).unwrap();
            if tracker.obs_summary.is_none() {
                if tracker.price_history.len() < 2 {
                    continue;
                }
                let first_ts = tracker.price_history.front().unwrap().0;
                let last_ts = tracker.price_history.back().unwrap().0;
                if (last_ts - first_ts).num_seconds() < 60 {
                    continue;
                }
                let prices: Vec<f64> = tracker.price_history.iter().map(|(_, p)| *p).collect();
                let n = prices.len() as f64;
                let mean = prices.iter().sum::<f64>() / n;
                let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let low = prices.iter().cloned().fold(f64::INFINITY, f64::min);
                let range = high - low;
                tracker.obs_summary = Some(ObsSummary { mean, high, low, range });
                tracker.obs_end_price = Some(spot);
                tracker.obs_end_time = Some(Utc::now());
            }
        }

        // Dedup on kalshi tick
        if last_kalshi_ts == Some(kalshi.ts) {
            continue;
        }

        // ============================================================
        // BUCKET 9 STRATEGY LOGIC
        // ============================================================

        // Check if YES or NO is in bucket 9 (asset-specific min price)
        let asset_name = &config.name;
        let yes_in_bucket9 = is_bucket_9(kalshi.yes_ask, asset_name);
        let no_in_bucket9 = is_bucket_9(kalshi.no_ask, asset_name);

        if !yes_in_bucket9 && !no_in_bucket9 {
            // Neither side is bucket 9, skip
            continue;
        }

        // Determine direction based on BTC price vs strike
        let btc_price = spot;
        let strike = kalshi.strike;

        // Asset-specific side selection based on win rate analysis:
        // - SOL: Both YES (99.3%) and NO (100%) work well - trade both with directional confirmation
        // - BTC/ETH/XRP: NO only (YES causes 98%+ of losses)
        let (side, entry_price) = if asset_name.to_lowercase().contains("sol") {
            // SOL: Trade both directions with directional confirmation
            if btc_price < strike && no_in_bucket9 {
                ("no".to_string(), kalshi.no_ask)
            } else if btc_price > strike && yes_in_bucket9 {
                ("yes".to_string(), kalshi.yes_ask)
            } else {
                continue;
            }
        } else {
            // BTC/ETH/XRP: NO trades only
            if btc_price < strike && no_in_bucket9 {
                ("no".to_string(), kalshi.no_ask)
            } else {
                continue;
            }
        };

        // Double-check bucket 9 (should always pass at this point)
        if !is_bucket_9(entry_price, asset_name) {
            continue;
        }

        // Spread check for execution quality
        let spread = kalshi.yes_ask - kalshi.yes_bid;
        if spread > 0.05 {
            eprintln!("[BUCKET9] {} ❌ Spread too wide: {:.3}", kalshi.ticker, spread);
            continue;
        }

        // Cooldown: don't trade same ticker more than once per 2 seconds
        if let Some(last_time) = last_trade_time.get(&kalshi.ticker) {
            if (Utc::now() - *last_time).num_seconds() < 2 {
                continue;
            }
        }

        // Calculate edge for logging
        let edge = (1.0 - entry_price) / entry_price;

        eprintln!(
            "[BUCKET9] {} ✅ {} at ${:.2} | {}s to expiry | BTC=${:.0} vs strike={:.0} | edge={:.1}%",
            kalshi.ticker, side.to_uppercase(), entry_price, secs_left, btc_price, strike, edge * 100.0
        );

        // Store model_prob for compatibility
        let model_prob = if btc_price > strike { 0.9 } else { 0.1 };
        state.lock().unwrap().model_prob = Some(model_prob);
        last_kalshi_ts = Some(kalshi.ts);

        if paper_mode {
            // ============================================================
            // POSITION GUARDS (SIMPLIFIED - NO LIMITS)
            // ============================================================
            {
                let app = state.lock().unwrap();
                let ticker_positions: Vec<_> = app.open_positions.iter()
                    .filter(|p| p.ticker == kalshi.ticker)
                    .collect();

                // ONLY check for opposite-direction positions
                // NO limit on number of same-side positions
                // NO MIN_PRICE_DELTA check
                let has_opposite_side = ticker_positions.iter().any(|p| p.side != side);
                if has_opposite_side {
                    eprintln!("[BUCKET9] {} ❌ Blocked {} - opposite position exists",
                              kalshi.ticker, side.to_uppercase());
                    continue;
                }

                // Log how many positions we have (for monitoring)
                let same_side_count = ticker_positions.len();
                if same_side_count > 0 {
                    eprintln!("[BUCKET9] {} Adding position #{} (same side: {})",
                              kalshi.ticker, same_side_count + 1, side.to_uppercase());
                }
            }

            // Capital check
            let available = state.lock().unwrap().available_capital;
            if available < bet_size {
                eprintln!("[BUCKET9] insufficient capital: ${available:.2} < ${bet_size:.2}");
                continue;
            }

            // Calculate contracts
            let count = (bet_size / entry_price).floor() as u64;
            if count == 0 {
                continue;
            }
            let count = count.min(MAX_CONTRACTS_PER_POSITION);

            // Fee + total cost
            let entry_fee = order_fee(count, entry_price);
            let total_cost = count as f64 * entry_price + entry_fee;

            // Debit capital
            state.lock().unwrap().available_capital -= total_cost;

            // Paper fill
            let pos = OpenPosition {
                ticker: kalshi.ticker.clone(),
                strike: kalshi.strike,
                side: side.clone(),
                entry_price,
                entry_yes_bid: kalshi.yes_bid,
                entry_yes_ask: kalshi.yes_ask,
                entry_no_bid: kalshi.no_bid,
                entry_no_ask: kalshi.no_ask,
                entry_fee,
                bet_size,
                count,
                expiry_ts: kalshi.expiry_ts,
                model_prob,
                edge,
                opened_at: Utc::now(),
                entry_rl_state: None,  // No RL state for bucket 9 bot
            };
            state.lock().unwrap().open_positions.push(pos);

            // Update last trade time
            last_trade_time.insert(kalshi.ticker.clone(), Utc::now());

            let position_count = state.lock().unwrap().open_positions.iter()
                .filter(|p| p.ticker == kalshi.ticker)
                .count();

            eprintln!(
                "[BUCKET9] PLACED {} #{} @ {:.2} | {} | secs_left={} | count={} | cost=${:.2} | capital=${:.2}",
                side.to_uppercase(), position_count, entry_price, kalshi.ticker, secs_left, count, total_cost,
                state.lock().unwrap().available_capital
            );

            let asset = extract_asset(&kalshi.ticker);
            let expiry = format_expiry(kalshi.expiry_ts);
            let msg = telegram_reporter::format_trade_entry(
                asset,
                kalshi.strike,
                &side.to_uppercase(),
                entry_price,
                count,
                total_cost,
                model_prob,
                edge,
                &expiry,
            );
            state.lock().unwrap().notifications.push_back(msg);
        } else {
            // Live order
            eprintln!("[BUCKET9] placing live order: {} {} @ ${:.2}", kalshi.ticker, side, entry_price);
            match place_order(
                &client,
                &api_key_id,
                &signing_key,
                &base_url,
                &kalshi,
                &side,
                entry_price,
            ).await {
                Ok(order_json) => {
                    let order_id = order_json
                        .get("order")
                        .and_then(|o| o.get("id"))
                        .and_then(|id| id.as_str())
                        .unwrap_or("?");
                    eprintln!("[BUCKET9] order created: {order_id}");

                    // Update last trade time
                    last_trade_time.insert(kalshi.ticker.clone(), Utc::now());

                    let asset = extract_asset(&kalshi.ticker);
                    let expiry = format_expiry(kalshi.expiry_ts);
                    let msg = telegram_reporter::format_trade_entry(
                        asset,
                        kalshi.strike,
                        &side.to_uppercase(),
                        entry_price,
                        1,
                        entry_price,
                        model_prob,
                        edge,
                        &expiry,
                    );
                    state.lock().unwrap().notifications.push_back(msg);
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if base_url == BASE_URL_PROD
                        && (err_str.contains("lookup") || err_str.contains("connect"))
                    {
                        eprintln!("[BUCKET9] production unreachable, falling back to demo");
                        base_url = BASE_URL_DEMO.to_string();
                    }
                    eprintln!("[BUCKET9] order failed: {e}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Settlement
// ---------------------------------------------------------------------------

async fn settle_expired(
    state: &Arc<Mutex<AppState>>,
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
) {
    let expired_tickers: Vec<(usize, String)> = {
        let app = state.lock().unwrap();
        app.open_positions
            .iter()
            .enumerate()
            .filter_map(|(i, pos)| {
                let secs = crate::model::seconds_to_expiry(pos.expiry_ts);
                if secs <= 0 {
                    Some((i, pos.ticker.clone()))
                } else {
                    None
                }
            })
            .collect()
    };

    if expired_tickers.is_empty() {
        return;
    }

    let mut settled_indices: Vec<(usize, bool)> = Vec::new();

    for (idx, ticker) in expired_tickers.iter().rev() {
        // Base URL logic needed here. Settle expired uses REST API.
        // For simplicity, we can default to config based URL or pass it in.
        // Since we don't have config here, let's assume PROD unless we change the signature again.
        // But wait, we *need* it to work on DEMO.
        // Best approach: Add base_url to settle_expired signature.
        let is_demo = std::env::var("KALSHI_ENV").unwrap_or_default().to_lowercase() == "demo";
         let base_url = if is_demo { "https://demo-api.kalshi.co" } else { "https://api.elections.kalshi.com" };

        match kalshi_listener::fetch_market_result(client, api_key_id, signing_key, ticker, base_url).await {
            Some((status, result)) => {
                if status == "finalized" || status == "settled" {
                    if let Some(result_str) = result {
                        let yes_won = result_str == "yes";
                        settled_indices.push((*idx, yes_won));
                        eprintln!(
                            "[BUCKET9] {} API result: {} (YES {})",
                            ticker, result_str, if yes_won { "WON" } else { "LOST" }
                        );
                    }
                }
            }
            None => {
                eprintln!("[BUCKET9] {} API error fetching settlement", ticker);
            }
        }
    }

    for (idx, yes_won) in settled_indices {
        let mut app = state.lock().unwrap();

        // Safety check
        if idx >= app.open_positions.len() {
            continue;
        }

        // Remove position (swap_remove is safe since we iterate indices in reverse)
        let pos = app.open_positions.swap_remove(idx);

        let won = match pos.side.as_str() {
            "yes" => yes_won,
            _     => !yes_won,
        };

        let exit_value   = if won { 1.0 } else { 0.0 };
        let count_f64    = pos.count as f64;
        let total_cost   = count_f64 * pos.entry_price + pos.entry_fee;
        let total_payout = count_f64 * exit_value;
        let pnl          = total_payout - total_cost;
        let roi_pct      = if total_cost > 0.0 { pnl / total_cost * 100.0 } else { 0.0 };

        // Update Account State
        app.available_capital += total_payout;
        
        // Only update session stats (since we are in paper mode)
        app.session_pnl  += pnl;
        app.total_trades += 1;
        if won {
            app.wins += 1;
        }

        eprintln!(
            "[BUCKET9] Settle {} {} | PnL: ${:.2} (ROI: {:.1}%) | Cap: ${:.2}",
            pos.ticker, if won { "WIN" } else { "LOSS" }, pnl, roi_pct, app.available_capital
        );

        // Queue for DB persistence (handled by data_recorder)
        app.pending_trades.push_back(crate::model::TradeRecord {
            opened_at:   pos.opened_at,
            closed_at:   chrono::Utc::now(),
            strike:      pos.strike,
            side:        pos.side.to_uppercase(),
            entry_price: pos.entry_price,
            exit_price:  exit_value,
            pnl,
            roi:         roi_pct,
        });

        // Notify Telegram
        let asset = extract_asset(&pos.ticker);
        let msg = crate::telegram_reporter::format_trade_result(
            asset,
            pos.strike,
            &pos.side,
            pos.entry_price,
            pos.count,
            pnl,
            roi_pct,
            won
        );
        app.notifications.push_back(msg);
    }
}

// ---------------------------------------------------------------------------
// Kalshi REST order placement
// ---------------------------------------------------------------------------

async fn place_order(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    kalshi: &MarketState,
    side: &str,
    entry_price: f64,
) -> anyhow::Result<serde_json::Value> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let path = REST_PATH;
    let message = format!("{}{}{}", timestamp_ms, "POST", path);
    // ... signature logic ...
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_vec());

    let price_cents = (entry_price * 100.0).round() as i64;
    // ... body construction ...
    let mut body = serde_json::json!({
        "ticker": kalshi.ticker,
        "action": "buy",
        "side": side,
        "count": 1,
        "type": "limit",
        "client_order_id": client_order_id(),
    });

    let price_key = if side == "yes" { "yes_price" } else { "no_price" };
    body[price_key] = serde_json::json!(price_cents);

    let url = format!("{}{}", base_url, path);

    let resp = client
        .post(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &sig_b64)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .json(&body)
        .send()
        .await?;
        
    let status = resp.status();
    if status == reqwest::StatusCode::CREATED {
        Ok(resp.json().await?)
    } else {
        let text = resp.text().await?;
        Err(anyhow::anyhow!("Kalshi {} — {}", status, text))
    }
}

fn sign_request(
    signing_key: &SigningKey<Sha256>,
    timestamp_ms: u64,
    method: &str,
    path: &str,
) -> String {
    let message = format!("{}{}{}", timestamp_ms, method, path);
    let signature = signing_key.sign_with_rng(&mut OsRng, message.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(signature.to_vec())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn client_order_id() -> String {
    let mut rng = OsRng;
    let a: u32 = rng.gen();
    let b: u16 = rng.gen();
    let c: u16 = (rng.gen::<u16>() & 0x0fff) | 0x4000;
    let d: u16 = (rng.gen::<u16>() & 0x3fff) | 0x8000;
    let e: u64 = rng.gen::<u64>() & 0x0000_ffff_ffff_ffff;
    format!("{a:08x}-{b:04x}-{c:04x}-{d:04x}-{e:012x}")
}

fn extract_asset(ticker: &str) -> &str {
    if ticker.contains("BTC") {
        "BTC-B9"  // Mark as bucket 9 bot in messages
    } else if ticker.contains("ETH") {
        "ETH-B9"
    } else if ticker.contains("SOL") {
        "SOL-B9"
    } else if ticker.contains("XRP") {
        "XRP-B9"
    } else {
        "?-B9"
    }
}

fn format_expiry(expiry_ts: Option<chrono::DateTime<chrono::Utc>>) -> String {
    match expiry_ts {
        Some(dt) => {
            let est = chrono::FixedOffset::west_opt(5 * 3600).unwrap();
            dt.with_timezone(&est).format("%H:%M EST").to_string()
        }
        None => "??:??".to_string(),
    }
}
