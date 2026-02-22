//! Rule-based trade executor for BTC 15-minute markets.
//!
//! Implements the OPTIMAL STRATEGY derived from 4 days of market data:
//! - Time window: 90-240 seconds before expiry
//! - Entry price: $0.15-$0.70 only
//! - Direction: BTC > strike → BUY YES, BTC < strike → BUY NO
//!
//! This is a separate bot from the RL-based executor for A/B testing.

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
// OPTIMAL STRATEGY PARAMETERS (from 4-day backtest)
// ============================================================

/// Minimum seconds to expiry for entry
const MIN_SECS_TO_EXPIRY: i64 = 90;

/// Maximum seconds to expiry for entry
const MAX_SECS_TO_EXPIRY: i64 = 240;

/// Minimum entry price (filters out extreme favorites)
const MIN_ENTRY_PRICE: f64 = 0.15;

/// Maximum entry price (filters out unfavorable risk/reward)
const MAX_ENTRY_PRICE: f64 = 0.70;

// ============================================================
// POSITION MANAGEMENT
// ============================================================

/// Minimum price movement required before a second same-side position
const MIN_PRICE_DELTA: f64 = 0.05;

/// Maximum positions per ticker (both sides combined)
const MAX_POSITIONS_PER_TICKER: usize = 3;

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

/// Entry point for the rule-based trade executor.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[RULEBASED] task started (asset: {})", config.name);
    eprintln!("[RULEBASED] Strategy: {}-{}s window, ${:.2}-${:.2} entry price",
              MIN_SECS_TO_EXPIRY, MAX_SECS_TO_EXPIRY, MIN_ENTRY_PRICE, MAX_ENTRY_PRICE);

    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);
    let bet_size: f64 = std::env::var("BET_SIZE")
        .unwrap_or_else(|_| "100".into())  // Default to $100 for rule-based
        .parse()
        .unwrap_or(100.0);

    if paper_mode {
        eprintln!("[RULEBASED] PAPER MODE — fills simulated, no orders sent");
    }

    let api_key_id = std::env::var("KALSHI_API_KEY_ID")?;
    let private_key_pem = std::env::var("KALSHI_PRIVATE_KEY")?;
    let private_key = RsaPrivateKey::from_pkcs1_pem(&private_key_pem)
        .context("Failed to parse KALSHI_PRIVATE_KEY as PKCS#1 PEM")?;
    let signing_key = SigningKey::<Sha256>::new(private_key);
    let client = reqwest::Client::new();

    let mut base_url: &str = BASE_URL_PROD;

    // Initialize paper-mode capital
    if paper_mode {
        state.lock().unwrap().available_capital = starting_capital;
        eprintln!("[RULEBASED] capital=${starting_capital:.2}  bet_size=${bet_size:.2}");
    }

    let mut market_trackers: HashMap<String, MarketTracker> = HashMap::new();
    let mut last_tracker_cleanup: chrono::DateTime<chrono::Utc> = Utc::now();
    let mut last_kalshi_ts: Option<chrono::DateTime<chrono::Utc>> = None;

    // Track which tickers we've already traded (one trade per market)
    let mut traded_tickers: std::collections::HashSet<String> = std::collections::HashSet::new();

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[RULEBASED] shutdown requested, exiting");
            return Ok(());
        }

        // Snapshot shared state
        let (kalshi, coinbase, _price_window) = {
            let app = state.lock().unwrap();
            (app.kalshi.clone(), app.coinbase.clone(), app.price_window.clone())
        };

        let (kalshi, coinbase) = match (kalshi, coinbase) {
            (Some(k), Some(c)) => (k, c),
            _ => continue, // wait for both feeds
        };

        // Settle expired paper positions
        if paper_mode {
            settle_expired(&state, &client, &api_key_id, &signing_key).await;
        }

        // Only trade 15-min markets for the configured asset
        let series_prefix = format!("{}-", config.kalshi_series);
        if !kalshi.ticker.starts_with(&series_prefix) {
            continue;
        }

        // Push BTC price into all active trackers
        let spot = coinbase.current_price;
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
            // Clean up traded tickers that are old
            traded_tickers.retain(|t| market_trackers.contains_key(t));
            last_tracker_cleanup = now;
        }

        // Get seconds to expiry
        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts) as i64;

        // Skip if expired
        if secs_left < 30 {
            market_trackers.remove(&kalshi.ticker);
            traded_tickers.remove(&kalshi.ticker);
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
        // OPTIMAL STRATEGY LOGIC
        // ============================================================

        // 1. Time window check (90-240 seconds)
        if secs_left < MIN_SECS_TO_EXPIRY || secs_left > MAX_SECS_TO_EXPIRY {
            continue;  // Outside optimal window
        }

        // Skip if we already traded this ticker
        if traded_tickers.contains(&kalshi.ticker) {
            continue;
        }

        // 2. Determine direction and entry price
        let btc_price = coinbase.current_price;
        let strike = kalshi.strike;
        let (side, entry_price) = if btc_price > strike {
            ("yes".to_string(), kalshi.yes_ask)  // Price above strike → BUY YES
        } else {
            ("no".to_string(), kalshi.no_ask)    // Price below strike → BUY NO
        };

        // 3. Entry price filter ($0.15-$0.70)
        if entry_price < MIN_ENTRY_PRICE || entry_price > MAX_ENTRY_PRICE {
            eprintln!(
                "[RULEBASED] {} ❌ Entry ${:.2} outside ${:.2}-${:.2} range (BTC=${:.0} strike={:.0})",
                kalshi.ticker, entry_price, MIN_ENTRY_PRICE, MAX_ENTRY_PRICE, btc_price, strike
            );
            continue;
        }

        // 4. Spread check for execution quality
        let spread = kalshi.yes_ask - kalshi.yes_bid;
        if spread > 0.05 {
            eprintln!("[RULEBASED] {} ❌ Spread too wide: {:.3}", kalshi.ticker, spread);
            continue;
        }

        // Calculate edge for logging
        let edge = (1.0 - entry_price) / entry_price;

        eprintln!(
            "[RULEBASED] {} ✅ {} at ${:.2} | {}s to expiry | BTC=${:.0} vs strike={:.0} | edge={:.1}%",
            kalshi.ticker, side.to_uppercase(), entry_price, secs_left, btc_price, strike, edge * 100.0
        );

        // Store model_prob for compatibility
        let model_prob = if btc_price > strike { 0.9 } else { 0.1 };
        state.lock().unwrap().model_prob = Some(model_prob);
        last_kalshi_ts = Some(kalshi.ts);

        if paper_mode {
            // Position guards
            {
                let app = state.lock().unwrap();
                let ticker_positions: Vec<_> = app.open_positions.iter()
                    .filter(|p| p.ticker == kalshi.ticker)
                    .collect();

                let same_side_count = ticker_positions.iter().filter(|p| p.side == side).count();
                if ticker_positions.len() >= MAX_POSITIONS_PER_TICKER || same_side_count >= 2 {
                    continue;
                }

                let position_conflict = ticker_positions.iter().any(|p| {
                    if p.side != side {
                        true
                    } else {
                        (p.entry_price - entry_price).abs() < MIN_PRICE_DELTA
                    }
                });
                if position_conflict {
                    continue;
                }
            }

            // Capital check
            let available = state.lock().unwrap().available_capital;
            if available < bet_size {
                eprintln!("[RULEBASED] insufficient capital: ${available:.2} < ${bet_size:.2}");
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
                entry_rl_state: None,  // No RL state for rule-based
            };
            state.lock().unwrap().open_positions.push(pos);

            // Mark this ticker as traded
            traded_tickers.insert(kalshi.ticker.clone());

            eprintln!(
                "[RULEBASED] PLACED {} @ {:.3} | {} strike={} | secs_left={} | count={} | cost=${:.2} | capital=${:.2}",
                side.to_uppercase(), entry_price, kalshi.ticker, kalshi.strike, secs_left, count, total_cost,
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
            eprintln!("[RULEBASED] placing live order: {} {} @ ${:.2}", kalshi.ticker, side, entry_price);
            match place_order(
                &client,
                &api_key_id,
                &signing_key,
                base_url,
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
                    eprintln!("[RULEBASED] order created: {order_id}");

                    // Mark as traded
                    traded_tickers.insert(kalshi.ticker.clone());

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
                        eprintln!("[RULEBASED] production unreachable, falling back to demo");
                        base_url = BASE_URL_DEMO;
                    }
                    eprintln!("[RULEBASED] order failed: {e}");
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

    let is_demo = std::env::var("KALSHI_ENV").unwrap_or_default().to_lowercase() == "demo";
    let base_url = if is_demo { "https://demo-api.kalshi.co" } else { "https://api.elections.kalshi.com" };

    for (idx, ticker) in expired_tickers.iter().rev() {
        match kalshi_listener::fetch_market_result(client, api_key_id, signing_key, ticker, base_url).await {
            Some((status, result)) => {
                if status == "finalized" || status == "settled" {
                    if let Some(result_str) = result {
                        let yes_won = result_str == "yes";
                        settled_indices.push((*idx, yes_won));
                        eprintln!(
                            "[RULEBASED] {} API result: {} (YES {})",
                            ticker, result_str, if yes_won { "WON" } else { "LOST" }
                        );
                    }
                }
            }
            None => {
                eprintln!("[RULEBASED] {} API error fetching settlement", ticker);
            }
        }
    }

    for (idx, yes_won) in settled_indices {
        let mut app = state.lock().unwrap();

        if idx >= app.open_positions.len() {
            continue;
        }

        let pos = app.open_positions.swap_remove(idx);

        let won = match pos.side.as_str() {
            "yes" => yes_won,
            _ => !yes_won,
        };

        let exit_value = if won { 1.0 } else { 0.0 };
        let total_cost = pos.count as f64 * pos.entry_price + pos.entry_fee;
        let total_payout = pos.count as f64 * exit_value;
        let pnl = total_payout - total_cost;
        let roi_pct = if total_cost > 0.0 { pnl / total_cost * 100.0 } else { 0.0 };

        app.available_capital += total_payout;
        app.session_pnl += pnl;
        app.total_trades += 1;
        if won {
            app.wins += 1;
        }

        app.pending_trades.push_back(TradeRecord {
            opened_at: pos.opened_at,
            closed_at: Utc::now(),
            strike: pos.strike,
            side: pos.side.to_uppercase(),
            entry_price: pos.entry_price,
            exit_price: exit_value,
            pnl,
            roi: roi_pct,
        });

        let msg = telegram_reporter::format_trade_settlement(
            won,
            pnl,
            roi_pct,
            app.wins,
            app.total_trades,
            app.session_pnl,
        );
        app.notifications.push_back(msg);

        eprintln!(
            "[RULEBASED] SETTLED {} {} strike={} — {} pnl={pnl:+.2} capital=${:.2}",
            pos.side, pos.ticker, pos.strike,
            if won { "WIN" } else { "LOSS" },
            app.available_capital
        );
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

    let signature = sign_request(signing_key, timestamp_ms, "POST", REST_PATH);
    let price_cents = (entry_price * 100.0).round() as i64;

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
        Ok(resp.json().await?)
    } else {
        let text = resp.text().await?;
        Err(anyhow::anyhow!("Kalshi {status} — {text}"))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn extract_asset(ticker: &str) -> &str {
    if ticker.contains("BTC") {
        "BTC"
    } else if ticker.contains("ETH") {
        "ETH"
    } else if ticker.contains("SOL") {
        "SOL"
    } else {
        "?"
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
