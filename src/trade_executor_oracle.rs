//! Oracle Trade Executor
//!
//! Strategy: Binary Black-Scholes fair value pricing with 4-gate entry system + asset filters.
//!
//! Gates (all must pass):
//! 1. Time window: 240-60 seconds to expiry (minutes 11-14, skip last minute)
//! 2. Fair value ≥ Asset-Specific Threshold (0.75 or 0.80)
//! 3. Edge (P_fair - Kalshi_Yes_Ask) ≥ 0.03
//! 4. Entry price ≤ 0.80 (optimized for NEW COINS performance)
//!
//! Asset Filters (based on entry price analysis 2026-03-15 onwards):
//! - SOL: Fully enabled (recent shadow analysis shows strong +$1.09 avg P&L)
//! - XRP: NO-only (YES remains in shadow mode pending better performance)
//! - BTC/ETH: All trades enabled (strong performance at 0.75-0.85 entry range)
//!
//! Order management:
//! - Dynamic re-pegging when price moves ≥ $0.01
//! - Emergency cancel when fair value drops below limit price
//! - Rate limit: max 1 amendment per 2 seconds per ticker
//!
//! Fixed 30 contracts per trade. Paper mode supported.

use std::collections::{HashSet, HashMap};
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

use crate::model::{AppState, AssetConfig, OpenPosition, TradeRecord, ActiveOrder, OracleRestingOrder};
use crate::telegram_reporter;
use crate::kalshi_listener;
use crate::black_scholes;
use crate::order_manager;
use crate::coinbase_advanced_listener;

// ============================================================
// ORACLE STRATEGY PARAMETERS
// ============================================================

/// Minimum seconds to expiry for entry (minute 14 of 15-min market, skip last minute)
const MIN_SECS_TO_EXPIRY: i64 = 60;

/// Maximum seconds to expiry for entry (minute 11 of 15-min market)
/// Tightened from 360s to 240s to reduce exposure to pre-convergence noise.
const MAX_SECS_TO_EXPIRY: i64 = 240;

/// Minimum fair value for entry (Gate 2) is now asset-specific and stored in AssetConfig.
/// - Tier 1: 0.80 (BTC, ETH, SOL, XRP, DOGE) for high safety.
/// - Tier 2: 0.75 (BNB, HYPE) for profitable volume.

/// Minimum edge (P_fair - ask) for entry (Gate 3)
/// Increased from 0.03 to 0.05 for Live mode to buffer against slippage.
const MIN_EDGE: f64 = 0.05;

/// Fixed edge discount for limit price (Gate 4)
const LIMIT_DISCOUNT: f64 = 0.03;

/// Fixed number of contracts per trade
const FIXED_COUNT: u64 = 10;


/// Maximum open positions at any time
const MAX_OPEN_POSITIONS: usize = 5;

/// Maximum allowed latency (in seconds) between market event and trade calculation.
const MAX_LATENCY: f64 = 0.50;

/// Capital reserved per position
const CAPITAL_PER_POSITION: f64 = 200.0;

/// Maximum contracts per position (safety cap)
const MAX_CONTRACTS_PER_POSITION: u64 = 1000;

// ============================================================
// KALSHI API
// ============================================================

const REST_PATH: &str = "/trade-api/v2/portfolio/orders";
const BALANCE_PATH: &str = "/trade-api/v2/portfolio/balance";
const BASE_URL_PROD: &str = "https://api.elections.kalshi.com";
const BASE_URL_DEMO: &str = "https://demo-api.kalshi.co";

/// Kalshi taker fee on an order of `count` contracts at `price` (0–1).
/// Kalshi taker fee estimation on an order of `count` contracts at `price` (0–1).
fn order_fee(_count: u64, _price: f64) -> f64 {
    // User requested zero fees as these are limit orders
    0.0
}

/// Extract market period from ticker (e.g., "KXBTC15M-26FEB221315-15" → "26FEB221315-15")
fn extract_market_period(ticker: &str) -> String {
    if let Some(idx) = ticker.find('-') {
        ticker[idx + 1..].to_string()
    } else {
        ticker.to_string()
    }
}

/// Entry point for the Oracle trade executor.
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[Oracle] task started (asset: {})", config.name);
    eprintln!("[Oracle] Strategy: Binary Black-Scholes fair value pricing");
    eprintln!("[Oracle] Time window: {}s - {}s to expiry (min 9 - min 14)", MIN_SECS_TO_EXPIRY, MAX_SECS_TO_EXPIRY);
    eprintln!("[Oracle] Gates: P_fair≥{}, Edge≥{}, Limit=P_fair-{}", config.min_fair_value, MIN_EDGE, LIMIT_DISCOUNT);
    eprintln!("[Oracle] Fixed position size: {} contracts", FIXED_COUNT);

    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);

    if paper_mode {
        eprintln!("[Oracle] PAPER MODE — fills simulated, no orders sent");
    } else {
        eprintln!("[Oracle] ⚠️  LIVE MODE — real money orders will be placed!");
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
        eprintln!("[Oracle] Paper capital initialized: ${starting_capital:.2}");
    }

    let mut last_kalshi_ts: Option<chrono::DateTime<chrono::Utc>> = None;

    // DEDUP: Track which market periods we've already traded
    let mut traded_periods: HashSet<String> = HashSet::new();
    let mut last_cleanup: chrono::DateTime<chrono::Utc> = Utc::now();
    let mut resting_orders: HashMap<String, OracleRestingOrder> = HashMap::new();

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[Oracle] shutdown requested, exiting");
            return Ok(());
        }

        // ============================================================
        // SETTLE EXPIRED POSITIONS
        // ============================================================
        settle_expired(&state, &client, &api_key_id, &signing_key, paper_mode, base_url).await;

        if !paper_mode {
            poll_resting_orders(
                &mut resting_orders,
                &state,
                &client,
                &api_key_id,
                &signing_key,
                base_url,
            )
            .await;
        }

        // ============================================================
        // ORDER AMENDMENT LOOP (check active orders for amend/cancel)
        // ============================================================
        {
            let active_tickers: Vec<String> = {
                let app = state.lock().unwrap();
                app.active_oracle_orders.keys().cloned().collect()
            };

            for ticker in active_tickers {
                let (current_order, l2_data, vol_window) = {
                    let app = state.lock().unwrap();
                    (
                        app.active_oracle_orders.get(&ticker).cloned(),
                        app.coinbase_l2.clone(),
                        app.volatility_window.clone(),
                    )
                };

                if let (Some(order), Some(l2)) = (current_order, l2_data) {
                    // Check staleness
                    if coinbase_advanced_listener::is_l2_stale(&state) {
                        continue;
                    }

                    // Get kalshi data for strike price
                    let kalshi = {
                        let app = state.lock().unwrap();
                        app.kalshi.clone()
                    };

                    if let Some(kalshi) = kalshi {
                        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts) as i64;

                        // If we're past the time window, emergency cancel
                        if secs_left < 60 {
                            eprintln!("[Oracle] ⏰ {} past expiry window — emergency cancel", ticker);
                            if !paper_mode {
                                let _ = order_manager::cancel_order(
                                    &client, &api_key_id, &signing_key, base_url, &order.order_id,
                                ).await;
                            }
                            state.lock().unwrap().active_oracle_orders.remove(&ticker);
                            continue;
                        }

                        // Calculate new fair value
                        let volatility = black_scholes::calculate_rolling_volatility(&vol_window);
                        let fair_value_yes = black_scholes::binary_black_scholes(
                            l2.mid_price, kalshi.strike, secs_left as f64, volatility,
                        );

                        // Get fair value for the side we're trading
                        let fair_value = if order.side == "yes" {
                            fair_value_yes
                        } else {
                            1.0 - fair_value_yes
                        };

                        // Emergency cancel: fair value dropped below limit (LIVE MODE ONLY)
                        // In paper mode, let positions ride to settlement
                        if !paper_mode && fair_value + 0.05 < order.limit_price {
                            eprintln!(
                                "[Oracle] 🚨 {} emergency cancel — fair={:.3} < limit={:.2}",
                                ticker, fair_value, order.limit_price
                            );
                            let _ = order_manager::cancel_order(
                                &client, &api_key_id, &signing_key, base_url, &order.order_id,
                            ).await;

                            state.lock().unwrap().active_oracle_orders.remove(&ticker);
                            // Allow re-entry since we cancelled
                            let market_period = extract_market_period(&ticker);
                            traded_periods.remove(&market_period);
                            continue;
                        }

                        // Amend check: price moved enough
                        let new_limit = fair_value - LIMIT_DISCOUNT;
                        if order_manager::should_amend(order.limit_price, new_limit, order.last_amend) {
                            let old_limit = order.limit_price;
                            match order_manager::amend_order(
                                &state, &client, &api_key_id, &signing_key, base_url,
                                &ticker, &kalshi, new_limit, &order.side, order.count, paper_mode,
                            ).await {
                                Ok(Some(_new_id)) => {
                                    // Successfully amended
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    eprintln!("[Oracle] amend error for {}: {e}", ticker);
                                }
                            }
                        }
                    }
                }
            }
        }

        // ============================================================
        // SNAPSHOT SHARED STATE
        // ============================================================
        let (kalshi, l2_data, vol_window, iteration_count) = {
            let app = state.lock().unwrap();
            (
                app.kalshi.clone(),
                app.coinbase_l2.clone(),
                app.volatility_window.clone(),
                0u64, // Will use for periodic logging
            )
        };

        // DEBUG: Log market state every 60 seconds
        static mut LAST_DEBUG_LOG: Option<std::time::Instant> = None;
        unsafe {
            if LAST_DEBUG_LOG.is_none() {
                LAST_DEBUG_LOG = Some(std::time::Instant::now());
            } else if LAST_DEBUG_LOG.unwrap().elapsed().as_secs() >= 60 {
                if let Some(ref k) = kalshi {
                    if let Some(ref l2) = l2_data {
                        eprintln!(
                            "[Oracle] 📊 Market State | Ticker: {} | Strike: ${:.0} | Yes Ask: ${:.2} | No Ask: ${:.2} | Coinbase Mid: ${:.2} | Vol Window: {} ticks",
                            k.ticker, k.strike, k.yes_ask, k.no_ask, l2.mid_price, vol_window.len()
                        );
                    } else {
                        eprintln!("[Oracle] 📊 Kalshi data present but NO Coinbase L2 data");
                    }
                } else {
                    eprintln!("[Oracle] 📊 NO Kalshi market data yet");
                }
                LAST_DEBUG_LOG = Some(std::time::Instant::now());
            }
        }

        let kalshi = match kalshi {
            Some(k) => k,
            None => continue,
        };

        // Skip if no L2 data or it's stale
        if coinbase_advanced_listener::is_l2_stale(&state) {
            // Fall back to coinbase REST price if L2 is not available
            let coinbase_price = state.lock().unwrap().coinbase_price;
            if coinbase_price.is_none() {
                continue;
            }
        }

        // Only trade 15-min markets for this asset
        let series_prefix = format!("{}-", config.kalshi_series);
        if !kalshi.ticker.starts_with(&series_prefix) {
            continue;
        }

        // ============================================================
        // TOXICITY PAUSE: 8-9 AM CDT (13:00 - 14:00 UTC)
        // High-volatility window identified as toxic for this bot suite.
        // ============================================================
        let hour_utc = chrono::Timelike::hour(&Utc::now());
        if hour_utc == 13 {
            continue;
        }

        // Calculate weighted average of Coinbase L2 and Kraken to approximate BRTI
        // BRTI settlement uses CF Benchmarks index which aggregates multiple exchanges
        // including both Coinbase and Kraken. Using 50/50 weighted average reduces
        // basis risk compared to using Coinbase alone.
        //
        // Safety: Falls back to Coinbase-only if Kraken data is stale, invalid, or diverges too much.
        let mid_price = if let Some(ref l2) = l2_data {
            // Try to get Kraken price with safety checks
            let kraken_result = {
                let app = state.lock().unwrap();
                app.coinbase.as_ref().and_then(|idx| {
                    // Check 1: Price must be positive and reasonable (within 50% of Coinbase)
                    if idx.current_price <= 0.0 {
                        return None;
                    }

                    // Check 2: Kraken data must be fresh (< 5 seconds old)
                    let age_secs = (Utc::now() - idx.ts).num_seconds();
                    if age_secs > 5 {
                        return None;
                    }

                    // Check 3: Sanity check - Kraken and Coinbase shouldn't diverge by > 5%
                    let divergence_pct = ((idx.current_price - l2.mid_price) / l2.mid_price).abs();
                    if divergence_pct > 0.05 {
                        eprintln!("[Oracle] ⚠️  Kraken divergence too high: CB=${:.2}, KR=${:.2} ({:.1}% diff) - using CB only",
                            l2.mid_price, idx.current_price, divergence_pct * 100.0);
                        return None;
                    }

                    Some(idx.current_price)
                })
            };

            if let Some(kraken) = kraken_result {
                // Weighted average: 50% Coinbase L2, 50% Kraken
                (l2.mid_price + kraken) / 2.0
            } else {
                // Fallback to Coinbase L2 only if Kraken unavailable or invalid
                l2.mid_price
            }
        } else if let Some(cp) = state.lock().unwrap().coinbase_price {
            cp
        } else {
            continue;
        };

        // Calculate volatility (O(1) from stateful tracker)
        let volatility = if std::env::var("USE_EWMA").map_or(true, |v| v == "1") {
            let app_lock = state.lock().unwrap();
            let tracker = app_lock.multi_volatility_trackers
                .get(&config.name);

            match tracker {
                Some(t) => {
                    let v = t.current_volatility();
                    if v > 0.0 { v } else { black_scholes::MIN_VOLATILITY }
                },
                None => {
                    // Fallback to rolling, or MIN_VOLATILITY if no data yet
                    let v = black_scholes::calculate_rolling_volatility(&vol_window);
                    if v > 0.0 { v } else { black_scholes::MIN_VOLATILITY }
                }
            }
        } else {
            let v = black_scholes::calculate_rolling_volatility(&vol_window);
            if v > 0.0 { v } else { black_scholes::MIN_VOLATILITY }
        };

        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts) as i64;

        let fair_value_yes = black_scholes::binary_black_scholes(
            mid_price, kalshi.strike, secs_left as f64, volatility,
        );
        let fair_value_no = 1.0 - fair_value_yes;

        // Structured debug log for verification (helpful for user's offline analysis)
        let d2 = black_scholes::calculate_d2(mid_price, kalshi.strike, secs_left as f64 / (365.25 * 24.0 * 3600.0), volatility);
        let latency = (Utc::now() - kalshi.ts).num_milliseconds() as f64 / 1000.0;
        
        // GATE 0: Latency Filter (Ensures data fresh < 0.5s)
        if latency > MAX_LATENCY {
            if latency > 1.0 { // Only log high-latency rejections to avoid log spam
                eprintln!("[Oracle] 🛑 LATENCY REJECTED: {:.2}s for {}", latency, kalshi.ticker);
            }
            continue;
        }
        
        if (Utc::now() - *state.lock().unwrap().last_debug_log.get_or_insert(Utc::now() - Duration::from_secs(60))).num_seconds() > 30 {
             // Show price components for BRTI comparison
             let (coinbase_l2_price, kraken_price, kraken_age) = {
                 let app = state.lock().unwrap();
                 let cb = app.coinbase_l2.as_ref().map(|l2| l2.mid_price);
                 let kr = app.coinbase.as_ref().map(|idx| (idx.current_price, (Utc::now() - idx.ts).num_seconds()));
                 (cb, kr.map(|(p, _)| p), kr.map(|(_, a)| a))
             };

             if let (Some(cb), Some(kr)) = (coinbase_l2_price, kraken_price) {
                 let divergence_pct = ((kr - cb) / cb).abs() * 100.0;
                 let age = kraken_age.unwrap_or(0);
                 let source_label = if (mid_price - cb).abs() < 0.01 {
                     format!("CB-only (KR stale/invalid)")
                 } else {
                     format!("Weighted (CB+KR)")
                 };
                 eprintln!("[Oracle-Debug] {} | {} ${:.2} [CB: ${:.2}, KR: ${:.2} age={}s div={:.2}%] | Strike: ${:.2} | T: {}s | Vol: {:.2} | d2: {:.4} | P_fair: {:.4} | Latency: {:.2}s",
                    kalshi.ticker, source_label, mid_price, cb, kr, age, divergence_pct, kalshi.strike, secs_left, volatility, d2, fair_value_yes, latency);
             } else {
                 eprintln!("[Oracle-Debug] {} | CB-only ${:.2} (Kraken unavailable) | Strike: ${:.2} | T: {}s | Vol: {:.2} | d2: {:.4} | P_fair: {:.4} | Latency: {:.2}s",
                    kalshi.ticker, mid_price, kalshi.strike, secs_left, volatility, d2, fair_value_yes, latency);
             }
             // Note: last_debug_log is updated below in the price log
        }

        // ============================================================
        // GATE 1: Time window (600-840 seconds to expiry)
        // ============================================================
        if secs_left < MIN_SECS_TO_EXPIRY || secs_left > MAX_SECS_TO_EXPIRY {
            continue;
        }

        // Dedup on kalshi tick timestamp
        if last_kalshi_ts == Some(kalshi.ts) {
            continue;
        }

        // Dedup: one trade per market period
        let market_period = extract_market_period(&kalshi.ticker);
        if traded_periods.contains(&market_period) {
            continue;
        }

        // ============================================================
        // GATE 2: Fair value ≥ 0.60
        // ============================================================

        // ============================================================
        // SKIP: No active orderbook (empty or illiquid market)
        // ============================================================
        // yes_ask=0 means no one is selling YES contracts — can't trade.
        // yes_bid=0 and yes_ask=0 means completely empty book.
        if kalshi.yes_ask < 0.01 || (kalshi.yes_bid < 0.01 && kalshi.no_bid < 0.01) {
            if (Utc::now() - *state.lock().unwrap().last_debug_log.get_or_insert(Utc::now() - Duration::from_secs(60))).num_seconds() > 30 {
                eprintln!("[Oracle] {} 🧊 Illiquid: YES ${:.2}/${:.2} | NO ${:.2}/${:.2}", 
                    kalshi.ticker, kalshi.yes_bid, kalshi.yes_ask, kalshi.no_bid, kalshi.no_ask);
                state.lock().unwrap().last_debug_log = Some(Utc::now());
            }
            continue;
        }

        // Periodic price logging for user verification
        if (Utc::now() - *state.lock().unwrap().last_debug_log.get_or_insert(Utc::now() - Duration::from_secs(60))).num_seconds() > 30 {
            eprintln!("[Oracle] {} 📊 Prices: YES ${:.2}/${:.2} | NO ${:.2}/${:.2} | Latency: {:.2}s", 
                kalshi.ticker, kalshi.yes_bid, kalshi.yes_ask, kalshi.no_bid, kalshi.no_ask, latency);
            state.lock().unwrap().last_debug_log = Some(Utc::now());
        }

        // Determine which side has edge
        let edge_yes = fair_value_yes - kalshi.yes_ask;
        let edge_no = fair_value_no - kalshi.no_ask;

        let (side, fair_value, edge, ask_price) = if edge_yes >= edge_no && edge_yes >= MIN_EDGE {
            ("yes", fair_value_yes, edge_yes, kalshi.yes_ask)
        } else if edge_no >= MIN_EDGE {
            ("no", fair_value_no, edge_no, kalshi.no_ask)
        } else {
            // No edge on either side
            continue;
        };

        // Gate 2: Asset-specific Minimum Fair Value check
        if fair_value < config.min_fair_value {
            continue;
        }

        // SHADOW LOGGING: XRP YES trades (high settlement risk)
        // Mark them as "shadow" to log performance without real capital in LIVE mode.
        let is_shadow_trade = !paper_mode && side == "yes" && config.name == "oracle-xrp";


        // ============================================================
        // GATE 3: Edge ≥ 0.03
        // ============================================================
        if edge < MIN_EDGE {
            continue;
        }

        // ============================================================
        // GATE 4: Place limit order at P_fair - 0.03
        // ============================================================
        let limit_price = fair_value - LIMIT_DISCOUNT;

        // Ensure limit price is valid (at least $0.01)
        if limit_price < 0.01 || limit_price > 0.99 {
            continue;
        }

        let minute = (900 - secs_left) / 60;
        let count = FIXED_COUNT;

        // Determine if this should be a real fill or a shadow/paper fill
        let effective_paper_mode = paper_mode || is_shadow_trade;

        eprintln!(
            "[Oracle] {} ✅ GATES PASSED: {} {} at ${:.2} (fair={:.3}, edge={:.3}, σ={:.3}, mid=${:.2}, strike=${:.0}) | min{} ({}s)",
            kalshi.ticker, 
            if is_shadow_trade { "🧪 SHADOW" } else { "✅" },
            side.to_uppercase(), limit_price, fair_value, edge, volatility, mid_price, kalshi.strike, minute, secs_left
        );

        // Mark this period as traded
        traded_periods.insert(market_period.clone());
        last_kalshi_ts = Some(kalshi.ts);

        // Fee + total cost (use ask_price for aggressive fills)
        let fill_price = ask_price;
        let entry_fee = order_fee(count, fill_price);
        let total_cost = count as f64 * fill_price + entry_fee;

        // Check max positions
        {
            let app = state.lock().unwrap();
            if app.open_positions.len() >= MAX_OPEN_POSITIONS {
                eprintln!("[Oracle] {} ❌ max positions reached ({})", kalshi.ticker, MAX_OPEN_POSITIONS);
                traded_periods.remove(&market_period);
                continue;
            }
        }

        if effective_paper_mode {
            // ============================================================
            // PAPER / SHADOW MODE
            // ============================================================

            // Capital check
            let available = state.lock().unwrap().available_capital;
            if available < total_cost {
                eprintln!("[Oracle] insufficient capital: ${available:.2} < ${total_cost:.2}");
                traded_periods.remove(&market_period);
                continue;
            }

            // Debit capital
            state.lock().unwrap().available_capital -= total_cost;

            // Create paper position (immediate fill at the current ask price)
            let market_entry_price = ask_price;

            let pos = OpenPosition {
                ticker: kalshi.ticker.clone(),
                side: side.to_string(),
                count,
                entry_price: market_entry_price,
                entry_yes_bid: kalshi.yes_bid,
                entry_yes_ask: kalshi.yes_ask,
                entry_no_bid: kalshi.no_bid,
                entry_no_ask: kalshi.no_ask,
                entry_fee,
                bet_size: total_cost,
                opened_at: Utc::now(),
                expiry_ts: kalshi.expiry_ts,
                strike: kalshi.strike,
                model_prob: fair_value,
                edge,
                entry_rl_state: None,
            };

            state.lock().unwrap().open_positions.push(pos);

            let asset_key = extract_asset(&kalshi.ticker);
            let principal = ask_price * count as f64;
            
            // Telegram notification for paper/shadow fill
            let mut msg = telegram_reporter::format_oracle_trade_entry(
                asset_key,
                side,
                ask_price,
                count,
                principal,
                fair_value,
                volatility,
            );
            if is_shadow_trade {
                msg = format!("🧪 [SHADOW] {}", msg);
            }
            state.lock().unwrap().notifications.push_back(msg);
            
            if is_shadow_trade {
                eprintln!(
                    "[Oracle] 🧪 SHADOW FILLED {} @ ${:.2} | {} | min{} | count={} | cost=${:.2}",
                    side.to_uppercase(), ask_price, kalshi.ticker, minute, count, total_cost
                );
            } else {
                eprintln!(
                    "[Oracle] FILLED {} @ ${:.2} (LIMIT AT ASK) | {} | min{} | count={} | cost=${:.2} | capital=${:.2}",
                    side.to_uppercase(), ask_price, kalshi.ticker, minute, count, total_cost, state.lock().unwrap().available_capital
                );
            }
        } else {
            // ============================================================
            // LIVE MODE
            // ============================================================

            // Safety checks
            {
                let app = state.lock().unwrap();
                let has_ticker = app.open_positions.iter()
                    .any(|p| p.ticker == kalshi.ticker);
                if has_ticker {
                    eprintln!("[Oracle] {} ❌ Already have position in this market", kalshi.ticker);
                    traded_periods.remove(&market_period); // Ensure this market is not re-traded
                    continue;
                }
            }

            eprintln!("[Oracle] placing LIVE AGGRESSIVE LIMIT order: {} {} x {} @ ${:.2}", kalshi.ticker, side, count, ask_price);

            match order_manager::place_aggressive_limit_order(
                &client,
                &api_key_id,
                &signing_key,
                base_url,
                &kalshi.ticker,
                side,
                ask_price, // Price at the current ask for immediate fill
                count,
            )
            .await
            {
                Ok(order_obj) => {
                    let order_id = order_obj["order_id"].as_str().unwrap_or("unknown").to_string();
                    let order_status = order_obj["status"].as_str().unwrap_or("unknown");

                    if order_status == "filled" || order_status == "executed" {
                        let mut app = state.lock().unwrap();
                        app.open_positions.push(OpenPosition {
                            ticker: kalshi.ticker.clone(),
                            side: side.to_string(),
                            count,
                            entry_price: ask_price,
                            entry_yes_bid: kalshi.yes_bid,
                            entry_yes_ask: kalshi.yes_ask,
                            entry_no_bid: kalshi.no_bid,
                            entry_no_ask: kalshi.no_ask,
                            entry_fee,
                            bet_size: total_cost,
                            opened_at: Utc::now(),
                            expiry_ts: kalshi.expiry_ts,
                            strike: kalshi.strike,
                            model_prob: fair_value,
                            edge,
                            entry_rl_state: None,
                        });

                        let asset_key = extract_asset(&kalshi.ticker);
                        let principal = ask_price * count as f64;
                        
                        let msg = telegram_reporter::format_oracle_trade_entry(
                            asset_key,
                            side,
                            ask_price,
                            count,
                            principal,
                            fair_value,
                            volatility,
                        );
                        app.notifications.push_back(msg);

                        eprintln!(
                            "[Oracle] LIVE FILLED {} @ ${:.2} (INSTANT) | {}",
                            side.to_uppercase(), ask_price, kalshi.ticker
                        );
                    } else {
                        eprintln!("[Oracle] 📋 tracking resting order {order_id} (status={order_status}) for polling");
                        resting_orders.insert(order_id.clone(), OracleRestingOrder {
                            order_id,
                            ticker: kalshi.ticker.clone(),
                            side: side.to_string(),
                            count,
                            limit_price: ask_price,
                            fair_value,
                            edge,
                            entry_yes_bid: kalshi.yes_bid,
                            entry_yes_ask: kalshi.yes_ask,
                            entry_no_bid: kalshi.no_bid,
                            entry_no_ask: kalshi.no_ask,
                            entry_fee,
                            total_cost,
                            expiry_ts: kalshi.expiry_ts,
                            strike: kalshi.strike,
                            volatility,
                            placed_at: Utc::now(),
                        });
                    }
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if base_url == BASE_URL_PROD
                        && (err_str.contains("lookup") || err_str.contains("connect"))
                    {
                        eprintln!("[Oracle] production unreachable, falling back to demo");
                        base_url = BASE_URL_DEMO;
                    }
                    eprintln!("[Oracle] ❌ order failed: {e}");
                    traded_periods.remove(&market_period);
                }
            }
        }

        // Periodic cleanup
        if (Utc::now() - last_cleanup).num_seconds() > 120 {
            if traded_periods.len() > 200 {
                traded_periods.clear();
                eprintln!("[Oracle] Cleared traded_periods cache");
            }
            last_cleanup = Utc::now();
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
    paper_mode: bool,
    base_url: &str,
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

    let mut settled_indices: Vec<(usize, bool, String)> = Vec::new();

    for (idx, ticker) in expired_tickers.iter().rev() {
        match kalshi_listener::fetch_market_result(client, api_key_id, signing_key, &ticker, base_url).await {
            Some((status, result)) => {
                if status == "finalized" || status == "settled" {
                    if let Some(result_str) = result {
                        let yes_won = result_str == "yes";
                        let settlement = result_str.to_uppercase();
                        settled_indices.push((*idx, yes_won, settlement));
                        eprintln!(
                            "[Oracle] {} API result: {} (YES {})",
                            ticker, result_str, if yes_won { "WON" } else { "LOST" }
                        );
                    }
                }
            }
            None => {
                eprintln!("[Oracle] {} API error fetching settlement", ticker);
            }
        }
    }

    for (idx, yes_won, settlement) in settled_indices {
        let (pos, won, pnl, roi_pct, wins, total_trades, session_pnl, paper_capital) = {
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

            // Remove from active oracle orders
            app.active_oracle_orders.remove(&pos.ticker);

            let asset = extract_asset(&pos.ticker);
            app.pending_trades.push_back(TradeRecord {
                opened_at: pos.opened_at,
                closed_at: Utc::now(),
                asset: asset.to_string(),
                strike: pos.strike,
                side: pos.side.to_uppercase(),
                entry_price: pos.entry_price,
                exit_price: exit_value,
                pnl,
                roi: roi_pct,
                settlement: settlement.clone(),
                fair_value: pos.model_prob,
            });

            (pos, won, pnl, roi_pct, app.wins, app.total_trades, app.session_pnl, app.available_capital)
        };

        // Fetch Kalshi balance
        let display_balance = if !paper_mode {
            get_kalshi_balance(client, api_key_id, signing_key, base_url)
                .await
                .unwrap_or(paper_capital)
        } else {
            paper_capital
        };

        let asset = extract_asset(&pos.ticker);
        let outcome = if won { "WIN" } else { "LOSS" };
        let emoji = if won { "✅" } else { "❌" };
        let mode_tag = if paper_mode { "PAPER" } else { "LIVE" };

        let msg = format!(
            "{} *Oracle {} {} {}*\n\
             {} > ${:.0} | {} @ ${:.2}\n\
             💵 *P&L:* ${:+.2} ({:+.0}%)\n\
             📊 *Session:* {}/{} ({:.0}%) | ${:+.2}\n\
             💰 *Balance:* ${:.2}",
            emoji, mode_tag, asset, outcome,
            asset, pos.strike, pos.side.to_uppercase(), pos.entry_price,
            pnl, roi_pct,
            wins, total_trades, wins as f64 / total_trades as f64 * 100.0, session_pnl,
            display_balance
        );
        state.lock().unwrap().notifications.push_back(msg);

        eprintln!(
            "[Oracle] {} SETTLED {} {} strike={} — {} pnl={pnl:+.2} | Balance=${:.2}",
            emoji, pos.side, pos.ticker, pos.strike,
            if won { "WIN" } else { "LOSS" },
            display_balance
        );

        if !won {
            eprintln!("[Oracle] 🚨 LOSS on {} — reviewing strategy parameters", pos.ticker);
        }
    }
}

// ---------------------------------------------------------------------------
// Kalshi Balance API
// ---------------------------------------------------------------------------

async fn get_kalshi_balance(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
) -> anyhow::Result<f64> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_request(signing_key, timestamp_ms, "GET", BALANCE_PATH);
    let url = format!("{base_url}{BALANCE_PATH}");

    let resp = client
        .get(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &signature)
        .header("KALSHI-ACCESS-TIMESTAMP", timestamp_ms.to_string())
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("Balance API HTTP {}", resp.status()));
    }

    let json: serde_json::Value = resp.json().await?;
    let balance_cents = json.get("balance")
        .and_then(|b| b.as_i64())
        .unwrap_or(0);
    Ok(balance_cents as f64 / 100.0)
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
    } else if ticker.contains("XRP") {
        "XRP"
    } else if ticker.contains("HYPE") {
        "HYPE"
    } else if ticker.contains("DOGE") {
        "DOGE"
    } else if ticker.contains("BNB") {
        "BNB"
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

async fn poll_resting_orders(
    resting_orders: &mut HashMap<String, OracleRestingOrder>,
    state: &Arc<Mutex<AppState>>,
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
) {
    if resting_orders.is_empty() {
        return;
    }

    let order_ids: Vec<String> = resting_orders.keys().cloned().collect();

    for order_id in order_ids {
        let order = match resting_orders.get(&order_id) {
            Some(o) => o,
            None => continue,
        };

        match order_manager::fetch_order_status(client, api_key_id, signing_key, base_url, &order_id).await {
            Ok(order_obj) => {
                let status = order_obj["status"].as_str().unwrap_or("unknown");
                if status == "filled" || status == "executed" {
                    let mut app = state.lock().unwrap();
                    app.open_positions.push(OpenPosition {
                        ticker: order.ticker.clone(),
                        side: order.side.clone(),
                        count: order.count,
                        entry_price: order.limit_price,
                        entry_yes_bid: order.entry_yes_bid,
                        entry_yes_ask: order.entry_yes_ask,
                        entry_no_bid: order.entry_no_bid,
                        entry_no_ask: order.entry_no_ask,
                        entry_fee: order.entry_fee,
                        bet_size: order.total_cost,
                        opened_at: Utc::now(),
                        expiry_ts: order.expiry_ts,
                        strike: order.strike,
                        model_prob: order.fair_value,
                        edge: order.edge,
                        entry_rl_state: None,
                    });

                    let asset_key = extract_asset(&order.ticker);
                    let principal = order.limit_price * order.count as f64;
                    
                    let msg = telegram_reporter::format_oracle_trade_entry(
                        asset_key,
                        &order.side,
                        order.limit_price,
                        order.count,
                        principal,
                        order.fair_value,
                        order.volatility,
                    );
                    app.notifications.push_back(msg);

                    eprintln!(
                        "[Oracle] 🎯 resting order FILLED: {} {} @ ${:.2} | {}",
                        order.side.to_uppercase(), order.limit_price, order.ticker, order_id
                    );
                    resting_orders.remove(&order_id);
                } else if status == "canceled" || status == "rejected" {
                    eprintln!("[Oracle] ❌ resting order {}: {} ", order_id, status);
                    resting_orders.remove(&order_id);
                }
            }
            Err(e) => {
                eprintln!("[Oracle] ❌ poll error for {order_id}: {e}");
            }
        }
    }
}
