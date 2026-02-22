//! Trade executor — polls shared state, runs the edge model, and places limit
//! orders on Kalshi when the edge exceeds the minimum threshold.
//!
//! In paper mode (`PAPER_MODE=1`) fills are simulated locally and positions
//! settle against the 60-second rolling average at expiry.  No network call is
//! made for paper fills; the bot still connects to the **production** WebSocket
//! for real market data.
//!
//! ## Flow
//! 1. Both price feeds must be live (`kalshi` + `coinbase` in AppState).
//! 2. Open paper positions are checked for settlement (expiry reached).
//! 3. `run_inference` produces a model probability.  In paper mode it falls
//!    back to a simple sigmoid heuristic when the ONNX model is not loaded.
//! 4. A uniform random roll selects YES (if roll < model_prob) or NO; the
//!    selected side's edge is computed using the Kalshi fee formula.
//! 5. If `edge > MINIMUM_EDGE`:
//!    - Paper mode: record an `OpenPosition`; no network call.
//!    - Live mode:  POST a limit order to Kalshi REST.
//! 6. On successful entry a Trade Entry notification is pushed for Telegram.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::path::Path;

use anyhow::Context;
use rusqlite::{Connection, params};
use base64::Engine;
use chrono::{Timelike, Utc};
use ort::session::Session;
use ort::value::Tensor;
use rand::rngs::OsRng;
use rand::Rng;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use sha2::Sha256;
use tokio::time::Duration;

use crate::model::{AppState, AssetConfig, Experience, IndexState, MarketState, MarketTracker, ObsSummary, OpenPosition, RlState, TradeRecord};
use crate::telegram_reporter;
use crate::kalshi_listener;

/// Minimum net edge (after fees) required to place a trade.
const MINIMUM_EDGE: f64 = 0.05;

/// Minimum ask-price movement (in dollars) required before a second same-side
/// position is allowed on the same ticker.
const MIN_PRICE_DELTA: f64 = 0.05;

/// Hard cap on total open positions per ticker (both sides combined).
const MAX_POSITIONS_PER_TICKER: usize = 3;

/// Minimum entry price to avoid penny options with explosive contract counts.
const MIN_ENTRY_PRICE: f64 = 0.10;  // $0.10 minimum (10 cents)

/// Maximum edge cap to filter fake edges from model miscalibration.
const MAX_EDGE: f64 = 0.30;  // 30% maximum

/// Maximum probability mismatch between model and market price.
const MAX_PROB_MISMATCH: f64 = 0.20;  // 20 percentage points

/// Maximum contracts per position as safety net.
const MAX_CONTRACTS_PER_POSITION: u64 = 1000;

/// Number of model features — must match the trained ONNX model and scaler.json.
const N_FEATURES: usize = 18;

/// Kalshi REST path for order creation.
const REST_PATH: &str = "/trade-api/v2/portfolio/orders";

const BASE_URL_PROD: &str = "https://api.elections.kalshi.com";
const BASE_URL_DEMO: &str = "https://demo-api.kalshi.co";

/// RL Policy Structs ------------------------------------------------------
#[derive(serde::Deserialize, Debug, Clone)]
struct RlBuckets {
    distance: Vec<f64>,
    time:     Vec<i64>,
    price:    Vec<f64>,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct RlPolicyData {
    buckets:   RlBuckets,
    q_table:   HashMap<String, Vec<f64>>,
    timestamp: f64,
}

fn load_rl_policy(asset: &str) -> Option<RlPolicyData> {
    let policy_path = format!("model/{}/rl_policy.json", asset);
    let content = match std::fs::read_to_string(&policy_path) {
        Ok(s) => s,
        Err(e) => { eprintln!("[trade_executor] rl_policy.json missing: {e}"); return None; }
    };
    match serde_json::from_str::<RlPolicyData>(&content) {
        Ok(p) => {
            eprintln!("[trade_executor] RL policy loaded (ts={})", p.timestamp);
            Some(p)
        }
        Err(e) => { eprintln!("[trade_executor] rl_policy.json parse error: {e}"); None }
    }
}

/// Discretize continuous state into RlState struct matching Python logic.
/// momentum: price change over last 30s (positive = up, negative = down)
fn discretize_rl_state(policy: &RlPolicyData, dist_pct: f64, secs: i64, price: f64, is_above: bool, spread: f64, momentum: f64) -> RlState {
    let d_idx = policy.buckets.distance.iter().filter(|&&b| dist_pct >= b).count();
    let t_idx = policy.buckets.time.iter().filter(|&&b| secs <= b).count();
    let p_idx = policy.buckets.price.iter().filter(|&&b| price >= b).count();
    let dir = if is_above { 1 } else { 0 };
    let sp_idx = if spread > 0.05 { 1 } else { 0 };

    // Momentum bucket: 0=down (<-0.1%), 1=neutral, 2=up (>+0.1%)
    let mom_idx = if momentum < -0.1 { 0 } else if momentum > 0.1 { 2 } else { 1 };

    RlState {
        dist_bucket: d_idx,
        time_bucket: t_idx,
        price_bucket: p_idx,
        direction: dir,
        spread_bucket: sp_idx as u8,
        momentum_bucket: mom_idx,
        time_of_day_bucket: get_time_of_day_bucket(),
    }
}

fn get_rl_action(policy: &RlPolicyData, dist_pct: f64, secs: i64, price: f64, is_above: bool, spread: f64, momentum: f64) -> usize {
    // Epsilon-greedy exploration
    // Read from env var EPSILON (default 0.5 for exploration)
    let epsilon: f64 = std::env::var("EPSILON")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    
    let mut rng = rand::thread_rng();
    
    // With probability epsilon, take random action for exploration
    if rng.gen::<f64>() < epsilon {
        let random_action = rng.gen_range(0..3);  // 0=HOLD, 1=YES, 2=NO
        eprintln!("[RL] epsilon={:.2} → random action {}", epsilon, random_action);
        return random_action;
    }
    
    // Otherwise, use Q-table (exploit)
    // 1. Discretize match Python logic exactly
    let state = discretize_rl_state(policy, dist_pct, secs, price, is_above, spread, momentum);

    // 2. Lookup key "d,t,p,dir,sp,mom,tod"
    let key = format!("{},{},{},{},{},{},{}", state.dist_bucket, state.time_bucket, state.price_bucket, state.direction, state.spread_bucket, state.momentum_bucket, state.time_of_day_bucket);

    if let Some(q_vals) = policy.q_table.get(&key) {
        // Argmax: 0=HOLD, 1=BUY_YES, 2=BUY_NO
        let mut max_q = -f64::INFINITY;
        let mut best_action = 0;
        for (i, &q) in q_vals.iter().enumerate() {
            if q > max_q {
                max_q = q;
                best_action = i;
            }
        }
        best_action
    } else {
        // State not in Q-table: random exploration
        let random_action = rng.gen_range(0..3);
        eprintln!("[RL] state {} not in Q-table → random action {}", key, random_action);
        random_action
    }
}

// ============================================================================
// ACTUAL TRADES POLICY (Online Q-Learning from trades.db)
// ============================================================================

/// Q-table entry from actual trades learning
#[derive(serde::Deserialize, Debug, Clone)]
struct ActualTradesEntry {
    avg_pnl: f64,
    count: i32,
    confidence: String,
}

/// Thresholds for actual trades policy
#[derive(serde::Deserialize, Debug, Clone)]
struct ActualTradesThresholds {
    min_avg_pnl: f64,
    min_observations: i32,
    avoid_losing_states: bool,
}

/// Actual trades policy data
#[derive(serde::Deserialize, Debug, Clone)]
struct ActualTradesPolicy {
    #[serde(rename = "type")]
    policy_type: String,
    q_table: HashMap<String, ActualTradesEntry>,
    thresholds: ActualTradesThresholds,
    timestamp: f64,
}

/// Load actual trades policy from JSON
fn load_actual_trades_policy(asset: &str) -> Option<ActualTradesPolicy> {
    let policy_path = format!("model/{}/rl_policy_actual.json", asset);
    let content = match std::fs::read_to_string(&policy_path) {
        Ok(s) => s,
        Err(_) => return None,  // Silent failure - policy may not exist yet
    };
    match serde_json::from_str::<ActualTradesPolicy>(&content) {
        Ok(p) => {
            eprintln!("[ACTUAL_TRADES] Loaded policy ({} states, ts={})", p.q_table.len(), p.timestamp);
            Some(p)
        }
        Err(e) => {
            eprintln!("[ACTUAL_TRADES] Parse error: {e}");
            None
        }
    }
}

/// Check if we should trade based on actual trades history
/// Returns (should_trade, reason)
fn should_trade_actual(policy: &ActualTradesPolicy, price_bucket: usize, side_dir: usize, time_bucket: usize) -> (bool, String) {
    let key = format!("{},{},{}", price_bucket, side_dir, time_bucket);

    match policy.q_table.get(&key) {
        Some(entry) => {
            if entry.count < policy.thresholds.min_observations {
                (true, format!("insufficient_data (n={})", entry.count))
            } else if entry.avg_pnl < policy.thresholds.min_avg_pnl {
                (false, format!("losing_state (avg=${:.2}, n={})", entry.avg_pnl, entry.count))
            } else {
                (true, format!("profitable (avg=${:.2}, n={})", entry.avg_pnl, entry.count))
            }
        }
        None => (true, "unknown_state".to_string())
    }
}

/// Get price bucket (0-9) from entry price
fn get_price_bucket(price: f64) -> usize {
    ((price * 10.0).floor() as usize).min(9)
}

/// Get time of day bucket (0-3) from hour
fn get_time_bucket_from_hour(hour: u32) -> usize {
    if hour < 6 { 0 }       // night
    else if hour < 12 { 1 } // morning
    else if hour < 18 { 2 } // afternoon
    else { 3 }              // evening
}

/// Get current time-of-day bucket (0-3) for RL state
fn get_time_of_day_bucket() -> u8 {
    let hour = chrono::Utc::now().hour();
    get_time_bucket_from_hour(hour) as u8
}

// ============================================================================
// FILTERED TRADES LOGGING (for future analysis of what would have happened)
// ============================================================================

/// Initialize filtered trades database, creating table if it doesn't exist
fn init_filtered_trades_db(asset: &str) -> Option<Connection> {
    let db_dir = format!("data/{}", asset);
    let db_path = format!("{}/filtered_trades.db", db_dir);

    // Create directory if it doesn't exist
    if !Path::new(&db_dir).exists() {
        if let Err(e) = std::fs::create_dir_all(&db_dir) {
            eprintln!("[FILTERED_DB] Failed to create directory {}: {}", db_dir, e);
            return None;
        }
    }

    let conn = match Connection::open(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[FILTERED_DB] Failed to open {}: {}", db_path, e);
            return None;
        }
    };

    // Create table if it doesn't exist
    let create_sql = r#"
        CREATE TABLE IF NOT EXISTS filtered_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            price_bucket INTEGER NOT NULL,
            time_bucket INTEGER NOT NULL,
            side_dir INTEGER NOT NULL,
            filter_type TEXT NOT NULL,
            reason TEXT NOT NULL,
            state_avg_pnl REAL,
            secs_remaining INTEGER NOT NULL
        )
    "#;

    if let Err(e) = conn.execute(create_sql, []) {
        eprintln!("[FILTERED_DB] Failed to create table: {}", e);
        return None;
    }

    eprintln!("[FILTERED_DB] Initialized database: {}", db_path);
    Some(conn)
}

/// Log a filtered trade to the database
fn log_filtered_trade(
    conn: &Connection,
    ticker: &str,
    side: &str,
    entry_price: f64,
    price_bucket: usize,
    time_bucket: usize,
    side_dir: usize,
    filter_type: &str,
    reason: &str,
    state_avg_pnl: Option<f64>,
    secs_remaining: i64,
) {
    let timestamp = Utc::now().to_rfc3339();

    let result = conn.execute(
        "INSERT INTO filtered_trades (timestamp, ticker, side, entry_price, price_bucket, time_bucket, side_dir, filter_type, reason, state_avg_pnl, secs_remaining) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            timestamp,
            ticker,
            side,
            entry_price,
            price_bucket as i64,
            time_bucket as i64,
            side_dir as i64,
            filter_type,
            reason,
            state_avg_pnl,
            secs_remaining,
        ],
    );

    if let Err(e) = result {
        eprintln!("[FILTERED_DB] Failed to insert: {}", e);
    }
}

// ============================================================================

/// StandardScaler parameters loaded from model/scaler.json.
/// Applied as: scaled = (raw − mean) / std, per feature.
#[derive(serde::Deserialize)]
struct ScalerParams {
    mean: Vec<f64>,
    std:  Vec<f64>,
}

/// ONNX session + scaler, loaded once at startup.
struct OnnxModel {
    session: Session,
    scaler:  ScalerParams,
}

/// DQN ONNX model for deep Q-learning inference.
/// Takes 6D state features and outputs Q-values for 3 actions.
struct DqnModel {
    session: Session,
    scaler:  ScalerParams,
}

/// Attempts to load the DQN ONNX model from disk.
/// Returns None if DQN model doesn't exist (falls back to Q-table).
fn load_dqn_model(asset: &str) -> Option<DqnModel> {
    let scaler_path = format!("model/{}/scaler.json", asset);
    let scaler_json = match std::fs::read_to_string(&scaler_path) {
        Ok(s)  => s,
        Err(e) => { eprintln!("[trade_executor] DQN scaler.json missing: {e}"); return None; }
    };
    let scaler: ScalerParams = match serde_json::from_str(&scaler_json) {
        Ok(s)  => s,
        Err(e) => { eprintln!("[trade_executor] DQN scaler.json parse error: {e}"); return None; }
    };

    let onnx_path = format!("model/{}/dqn_model.onnx", asset);
    let session = match Session::builder()
        .and_then(|b| b.commit_from_file(&onnx_path))
    {
        Ok(s)  => s,
        Err(e) => { eprintln!("[trade_executor] DQN ONNX model not found or failed: {e}"); return None; }
    };

    eprintln!("[trade_executor] DQN ONNX model loaded successfully");
    Some(DqnModel { session, scaler })
}

/// Get action from DQN model.
/// Converts discrete state to continuous features, normalizes, runs inference.
fn get_dqn_action(model: &mut DqnModel, dist_pct: f64, secs: i64, price: f64, is_above: bool, spread: f64, momentum: f64) -> Option<usize> {
    // Convert to 6D continuous state (matching Python state_tuple_to_continuous)
    let dist_bucket = if dist_pct < 0.1 { 0 } else if dist_pct < 0.2 { 1 } else if dist_pct < 0.3 { 2 } else if dist_pct < 0.4 { 3 } else if dist_pct < 0.5 { 4 } else { 5 };
    let time_bucket = if secs <= 30 { 0 } else if secs <= 60 { 1 } else if secs <= 90 { 2 } else if secs <= 120 { 3 } else if secs <= 150 { 4 } else if secs <= 180 { 5 } else if secs <= 210 { 6 } else if secs <= 240 { 7 } else if secs <= 270 { 8 } else if secs <= 300 { 9 } else { 10 };
    let price_bucket = ((price * 10.0) as usize).min(9);
    let direction = if is_above { 1.0 } else { 0.0 };
    let spread_bucket = if spread > 0.05 { 1 } else { 0 };
    let momentum_bucket = if momentum < -0.1 { 0 } else if momentum > 0.1 { 2 } else { 1 };

    // Convert buckets to continuous features (matching Python)
    let state = [
        dist_bucket as f64 * 0.1,           // dist_pct
        time_bucket as f64 / 10.0,          // time_frac
        price_bucket as f64 * 0.1 + 0.05,   // price
        direction,                           // direction
        spread_bucket as f64 * 0.05,        // spread
        (momentum_bucket as f64 - 1.0) * 0.15, // momentum
    ];

    // Normalize with scaler
    if model.scaler.mean.len() != 6 || model.scaler.std.len() != 6 {
        eprintln!("[trade_executor] DQN scaler dimension mismatch");
        return None;
    }

    let mut normalized = [0.0f32; 6];
    for i in 0..6 {
        let std = model.scaler.std[i];
        normalized[i] = if std.abs() > 1e-8 {
            ((state[i] - model.scaler.mean[i]) / std) as f32
        } else {
            0.0
        };
    }

    // Run ONNX inference
    let input_tensor = match Tensor::from_array(([1usize, 6], normalized.to_vec())) {
        Ok(t) => t,
        Err(e) => { eprintln!("[trade_executor] DQN tensor creation failed: {e}"); return None; }
    };

    let inputs = ort::inputs!["state" => input_tensor];
    let outputs = match model.session.run(inputs) {
        Ok(o) => o,
        Err(e) => { eprintln!("[trade_executor] DQN inference failed: {e}"); return None; }
    };

    // Get Q-values and find argmax
    let q_values = match outputs.get("q_values") {
        Some(v) => match v.try_extract_tensor::<f32>() {
            Ok(t) => t,
            Err(e) => { eprintln!("[trade_executor] DQN output extraction failed: {e}"); return None; }
        },
        None => { eprintln!("[trade_executor] DQN output 'q_values' not found"); return None; }
    };

    // q_values is (&Shape, &[f32]) - get the slice
    let (_shape, data) = q_values;
    if data.len() < 3 {
        eprintln!("[trade_executor] DQN output too short: {}", data.len());
        return None;
    }

    // Argmax over 3 actions
    let mut best_action = 0;
    let mut best_q = data[0];
    for i in 1..3 {
        if data[i] > best_q {
            best_q = data[i];
            best_action = i;
        }
    }

    Some(best_action)
}

/// Attempts to load the ONNX session and scaler.json from disk.
/// Returns None (with a logged warning) on any failure — the bot continues
/// in degraded mode rather than crashing at startup.
fn load_model(asset: &str) -> Option<OnnxModel> {
    let scaler_path = format!("model/{}/scaler.json", asset);
    let scaler_json = match std::fs::read_to_string(&scaler_path) {
        Ok(s)  => s,
        Err(e) => { eprintln!("[trade_executor] scaler.json missing: {e}"); return None; }
    };
    let scaler: ScalerParams = match serde_json::from_str(&scaler_json) {
        Ok(s)  => s,
        Err(e) => { eprintln!("[trade_executor] scaler.json parse error: {e}"); return None; }
    };

    let onnx_path = format!("model/{}/strategy_model.onnx", asset);
    let session = match Session::builder()
        .and_then(|b| b.commit_from_file(&onnx_path))
    {
        Ok(s)  => s,
        Err(e) => { eprintln!("[trade_executor] ONNX model load failed: {e}"); return None; }
    };

    eprintln!("[trade_executor] ONNX model loaded successfully");
    Some(OnnxModel { session, scaler })
}

/// Computes the 18 model features from live tick data and the frozen
/// observation summary stored in the per-market tracker.
/// Returns None if any required state is missing.
fn compute_features(
    kalshi:       &MarketState,
    coinbase:     &IndexState,
    price_window: &VecDeque<(chrono::DateTime<chrono::Utc>, f64)>,
    tracker:      &MarketTracker,
    volatility_baseline: f64,
) -> Option<[f64; N_FEATURES]> {
    let obs      = tracker.obs_summary.as_ref()?;
    let obs_end  = tracker.obs_end_price?;
    if price_window.len() < 2 { return None; }

    let spot   = coinbase.current_price;
    let strike = kalshi.strike;

    // 0: distance_to_strike
    let f0 = (spot - strike) / strike;

    // 1: avg_distance_to_strike (frozen obs mean)
    let f1 = (obs.mean - strike) / strike;

    // 2: time_remaining_frac — clamped to [0, 1]
    let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts) as f64;
    let f2 = (secs_left / 900.0).clamp(0.0, 1.0);

    // 3: volatility_60 = population_std / mean  (coefficient of variation)
    let prices: Vec<f64> = price_window.iter().map(|(_, p)| *p).collect();
    let n    = prices.len() as f64;
    let mean = prices.iter().sum::<f64>() / n;
    let var  = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
    let f3   = if mean != 0.0 { var.sqrt() / mean } else { 0.0 };

    // 4: momentum_window (from obs_end / t=10 reference)
    let f4 = if obs_end != 0.0 { (spot - obs_end) / obs_end } else { 0.0 };

    // 5: momentum_short — 180-second lookback from tracker price buffer
    let now_ts    = chrono::Utc::now();
    let target_ts = now_ts - chrono::Duration::seconds(180);
    let short_ref = tracker.price_history.iter()
        .find(|(ts, _)| *ts >= target_ts)
        .map(|(_, p)| *p)
        .unwrap_or(obs_end);   // fallback to obs_end if buffer too short
    let f5 = if short_ref != 0.0 { (spot - short_ref) / short_ref } else { 0.0 };

    // 6: obs_high_dist
    let f6 = (spot - obs.high) / strike;

    // 7: obs_low_dist
    let f7 = (spot - obs.low) / strike;

    // 8: obs_range_norm (frozen)
    let f8 = obs.range / strike;

    // 9: dist_to_obs_mean
    let f9 = (spot - obs.mean) / strike;

    // 10: abs_distance_to_strike
    let f10 = f0.abs();

    // 11: strike_offset_from_start
    let start_price = tracker.price_history.front().map(|(_, p)| *p).unwrap_or(obs_end);
    let f11 = if start_price != 0.0 { (strike - start_price) / start_price } else { 0.0 };

    // --- New predictive features (f12-f17) ---

    // 12: momentum_5min (5-minute lookback momentum)
    let target_5min = now_ts - chrono::Duration::seconds(300);
    let price_5min_ago = tracker.price_history.iter()
        .find(|(ts, _)| *ts >= target_5min)
        .map(|(_, p)| *p)
        .unwrap_or(obs_end);
    let f12 = if price_5min_ago != 0.0 { (spot - price_5min_ago) / price_5min_ago } else { 0.0 };

    // 13: momentum_accel (difference between 5min and 10min momentum)
    let target_10min = now_ts - chrono::Duration::seconds(600);
    let price_10min_ago = tracker.price_history.iter()
        .find(|(ts, _)| *ts >= target_10min)
        .map(|(_, p)| *p)
        .unwrap_or(obs_end);
    let momentum_10min = if price_10min_ago != 0.0 { (spot - price_10min_ago) / price_10min_ago } else { 0.0 };
    let f13 = f12 - momentum_10min;

    // 14: reversion_pressure (z-score from obs_mean)
    let obs_std = obs.range / 4.0;  // approximate std from range
    let f14 = if obs_std > 0.0 { (spot - obs.mean) / obs_std } else { 0.0 };

    // 15: vol_vs_baseline (volatility relative to historical baseline)
    let f15 = f3 / volatility_baseline;

    // 16: distance_velocity (rate of change of distance_to_strike)
    let prev_spot = tracker.price_history.back().map(|(_, p)| *p).unwrap_or(spot);
    let prev_distance = (prev_spot - strike) / strike;
    let f16 = f0 - prev_distance;

    // 17: strike_crossing_momentum (momentum aligned with strike crossing)
    let f17 = f12 * if f0 > 0.0 { 1.0 } else { -1.0 };

    Some([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17])
}

/// Scales raw features with the StandardScaler, runs the ONNX session, and
/// extracts the scalar sigmoid probability.
fn run_onnx(model: &mut OnnxModel, raw_features: &[f64; N_FEATURES]) -> Option<f64> {
    // StandardScaler: (x − mean) / std.  All math in f64; single cast to f32.
    let mut scaled = vec![0.0f32; N_FEATURES];
    for i in 0..N_FEATURES {
        scaled[i] = ((raw_features[i] - model.scaler.mean[i]) / model.scaler.std[i]) as f32;
    }

    // Tensor input: shape [1, N_FEATURES], data is the scaled f32 vector.
    let tensor = match Tensor::from_array(([1usize, N_FEATURES], scaled)) {
        Ok(t)  => t,
        Err(e) => { eprintln!("[trade_executor] Tensor::from_array failed: {e}"); return None; }
    };

    let outputs = match model.session.run(ort::inputs![tensor]) {
        Ok(o)  => o,
        Err(e) => { eprintln!("[trade_executor] ONNX session.run failed: {e}"); return None; }
    };

    // Output is [1, 1] f32.  Extract as raw slice.
    let (_shape, data) = match outputs[0].try_extract_tensor::<f32>() {
        Ok(d)  => d,
        Err(e) => { eprintln!("[trade_executor] output extract failed: {e}"); return None; }
    };
    let prob = data[0] as f64;

    // Sanity: sigmoid output must be in [0, 1].
    // Note: We allow exact 0.0 and 1.0 because deep ITM/OTM situations
    // can legitimately produce near-certainty predictions.
    if prob.is_nan() || prob.is_infinite() || prob < 0.0 || prob > 1.0 {
        eprintln!("[trade_executor] ONNX output out of range: {prob}");
        return None;
    }

    Some(prob)
}

/// Entry point.  Parses the RSA key once, then enters the polling loop.
/// Falls back to the demo REST endpoint automatically on DNS failure (mirrors
/// the behaviour in `kalshi_listener`).
pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[trade_executor] task started (asset: {})", config.name);

    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);
    let bet_size: f64 = std::env::var("BET_SIZE")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .unwrap_or(10.0);

    if paper_mode {
        eprintln!("[trade_executor] PAPER MODE — fills simulated, no orders sent");
    }

    let api_key_id = std::env::var("KALSHI_API_KEY_ID")?;
    let private_key_pem = std::env::var("KALSHI_PRIVATE_KEY")?;
    let private_key = RsaPrivateKey::from_pkcs1_pem(&private_key_pem)
        .context("Failed to parse KALSHI_PRIVATE_KEY as PKCS#1 PEM")?;
    let signing_key = SigningKey::<Sha256>::new(private_key);
    let client = reqwest::Client::new();

    let mut base_url: String = config.api_base_url.clone();

    // --- Initialise paper-mode capital ---------------------------------------
    if paper_mode {
        state.lock().unwrap().available_capital = starting_capital;
        eprintln!("[trade_executor] capital=${starting_capital:.2}  bet_size=${bet_size:.2}");
    }

    // --- Load ONNX model once at startup -------------------------------------
    let mut onnx_model = load_model(&config.name);
    if onnx_model.is_none() {
        eprintln!(
            "[trade_executor] WARNING: model not loaded — {}",
            if paper_mode { "using heuristic fallback" } else { "live mode will not trade" }
        );
    }

    // --- Load RL Policy ------------------------------------------------------
    let mut rl_policy = load_rl_policy(&config.name);
    let mut last_policy_check = std::time::Instant::now();

    // --- Load DQN Model (if available) ---------------------------------------
    let mut dqn_model = load_dqn_model(&config.name);
    if dqn_model.is_some() {
        eprintln!("[trade_executor] DQN model available for inference");
    } else {
        eprintln!("[trade_executor] DQN model not found, using Q-table policy only");
    }

    // --- Load Actual Trades Policy (online learning from trades.db) ----------
    let mut actual_trades_policy = load_actual_trades_policy(&config.name);
    let mut last_actual_policy_check = std::time::Instant::now();

    // --- Initialize Filtered Trades Database ---------------------------------
    let filtered_trades_db = init_filtered_trades_db(&config.name);

    let mut market_trackers: HashMap<String, MarketTracker> = HashMap::new();
    let mut last_tracker_cleanup: chrono::DateTime<chrono::Utc> = Utc::now();

    let mut last_kalshi_ts: Option<chrono::DateTime<chrono::Utc>> = None;

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[trade_executor] shutdown requested, exiting");
            return Ok(());
        }

        // --- Snapshot shared state --------------------------------------------
        let (kalshi, coinbase, price_window) = {
            let app = state.lock().unwrap();
            (app.kalshi.clone(), app.coinbase.clone(), app.price_window.clone())
        };

        let (kalshi, coinbase) = match (kalshi, coinbase) {
            (Some(k), Some(c)) => (k, c),
            _ => continue, // wait for both feeds
        };

        // --- Settle expired paper positions ------------------------------------
        if paper_mode {
            settle_expired(&state, &client, &api_key_id, &signing_key).await;
        }

        // Only trade 15-min markets for the configured asset
        let series_prefix = format!("{}-", config.kalshi_series);
        if !kalshi.ticker.starts_with(&series_prefix) {
            continue;
        }

        // --- Push BTC price into all active trackers -------------------------
        let spot = coinbase.current_price;
        market_trackers.entry(kalshi.ticker.clone()).or_insert_with(MarketTracker::new);
        for tracker in market_trackers.values_mut() {
            tracker.price_history.push_back((Utc::now(), spot));
            // Cap buffer at 600 ticks (~10 minutes at 1 tick/sec) to prevent unbounded growth
            if tracker.price_history.len() > 600 {
                tracker.price_history.pop_front();
            }
        }

        // --- Periodic cleanup of stale trackers ------------------------------
        if (Utc::now() - last_tracker_cleanup).num_seconds() > 60 {
            let now = Utc::now();
            market_trackers.retain(|_, tracker| {
                tracker.price_history.back()
                    .map_or(false, |(ts, _)| (now - *ts).num_seconds() < 1200)
            });
            last_tracker_cleanup = now;
        }

        // --- Phase detection -------------------------------------------------
        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts);

        if secs_left < 30 {
            market_trackers.remove(&kalshi.ticker);
            continue;   // expiry cliff
        }

        // --- Trade phase (30 ≤ secs_left ≤ 900) ------------------------------
        // Removed observation phase gate - trust RL to learn optimal timing
        {
            let tracker = market_trackers.get_mut(&kalshi.ticker).unwrap();

            // Freeze ObsSummary on first tick in trading phase
            if tracker.obs_summary.is_none() {
                if tracker.price_history.len() < 2 {
                    continue;
                }
                let first_ts = tracker.price_history.front().unwrap().0;
                let last_ts  = tracker.price_history.back().unwrap().0;
                if (last_ts - first_ts).num_seconds() < 60 {
                    eprintln!(
                        "[trade_executor] {} skipping obs transition — only {}s of history",
                        kalshi.ticker, (last_ts - first_ts).num_seconds()
                    );
                    continue;
                }
                let prices: Vec<f64> = tracker.price_history.iter().map(|(_, p)| *p).collect();
                let n     = prices.len() as f64;
                let mean  = prices.iter().sum::<f64>() / n;
                let high  = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let low   = prices.iter().cloned().fold(f64::INFINITY, f64::min);
                let range = high - low;
                tracker.obs_summary    = Some(ObsSummary { mean, high, low, range });
                tracker.obs_end_price  = Some(spot);
                tracker.obs_end_time   = Some(Utc::now());
                eprintln!(
                    "[trade_executor] {} obs_summary computed: mean={mean:.2} high={high:.2} low={low:.2} range={range:.2}",
                    kalshi.ticker
                );
            }

            // Obs-phase cooldown: skip first 30s after transition
            if let Some(obs_end_time) = tracker.obs_end_time {
                if (Utc::now() - obs_end_time).num_seconds() < 30 {
                    continue;
                }
            }
        }

        // --- Spread sanity ---------------------------------------------------
        let yes_spread = kalshi.yes_ask - kalshi.yes_bid;
        let no_spread  = kalshi.no_ask  - kalshi.no_bid;
        if yes_spread > 0.05 || no_spread > 0.05 {
            eprintln!("[trade_executor] {} spread too wide (yes={yes_spread:.3} no={no_spread:.3})", kalshi.ticker);
            continue;
        }

        // --- Dedup on kalshi tick --------------------------------------------
        if last_kalshi_ts == Some(kalshi.ts) {
            continue;
        }

        // --- RL Policy Reload -----------------------------------------------
        if last_policy_check.elapsed() > Duration::from_secs(60) {
            if let Some(p) = load_rl_policy(&config.name) {
                rl_policy = Some(p);
            }
            last_policy_check = std::time::Instant::now();
        }

        // --- Actual Trades Policy Reload (every 5 minutes) ------------------
        if last_actual_policy_check.elapsed() > Duration::from_secs(300) {
            actual_trades_policy = load_actual_trades_policy(&config.name);
            last_actual_policy_check = std::time::Instant::now();
        }

        // --- RL Decision Logic ----------------------------------------------
        let btc_price = coinbase.current_price;
        let distance_pct = ((btc_price - kalshi.strike) / kalshi.strike).abs() * 100.0;
        let is_above_strike = btc_price > kalshi.strike;
        let spread = kalshi.yes_ask - kalshi.yes_bid;

        // Only trade in last 5 minutes (300 seconds) - matches time bucket design [30-300s]
        if secs_left > 300 {
            continue;  // Too early, wait until last 5 minutes
        }

        // Keep spread check for execution quality
        if spread > 0.05 {
            eprintln!("[trade_executor] {} RL: spread={:.3} > 0.05, skipping", kalshi.ticker, spread);
            continue;
        }

        // Use loaded policy for decision and capture RL state for experience logging
        let (side, entry_price, entry_rl_state) = if let Some(ref policy) = rl_policy {
             // Calculate momentum from price history (% change over last 30 data points)
             let momentum = if let Some(tracker) = market_trackers.get(&kalshi.ticker) {
                 if tracker.price_history.len() >= 30 {
                     let recent: Vec<_> = tracker.price_history.iter().rev().take(30).collect();
                     // recent.first() = index 0 = most recent price (newest)
                     // recent.last() = index 29 = 30 ticks ago (oldest)
                     if let (Some((_, newest)), Some((_, oldest))) = (recent.first(), recent.last()) {
                         (newest - oldest) / oldest * 100.0  // % change: positive = upward momentum
                     } else { 0.0 }
                 } else { 0.0 }
             } else { 0.0 };
             
             let action = get_rl_action(policy, distance_pct, secs_left as i64, kalshi.yes_ask, is_above_strike, spread, momentum);
             let rl_state = discretize_rl_state(policy, distance_pct, secs_left as i64, kalshi.yes_ask, is_above_strike, spread, momentum);

             // --- DQN Shadow Mode: Log what DQN would decide (no actual trading) ---
             if let Some(ref mut dqn) = dqn_model {
                 if let Some(dqn_action) = get_dqn_action(dqn, distance_pct, secs_left as i64, kalshi.yes_ask, is_above_strike, spread, momentum) {
                     let action_names = ["HOLD", "YES", "NO"];
                     let agrees = action == dqn_action;
                     let symbol = if agrees { "✓" } else { "✗" };
                     eprintln!(
                         "[DQN_SHADOW] {} Q-table={} DQN={} {} (dist={:.2}% t={}s p=${:.2})",
                         kalshi.ticker,
                         action_names[action],
                         action_names[dqn_action],
                         symbol,
                         distance_pct,
                         secs_left,
                         kalshi.yes_ask
                     );
                 }
             }

             // ============================================================
             // BUCKET-BASED FILTERS (from real trade data, last 3 days)
             // ============================================================

             // YES: Only allow buckets 4, 6, 9 (block 0, 1, 2, 3, 5, 7, 8)
             // Based on actual PnL: bucket 4 +$156, bucket 6 +$62, bucket 9 +$221
             if action == 1 {  // BUY_YES
                 let dominated_buckets = [0, 1, 2, 3, 5, 7, 8];
                 if dominated_buckets.contains(&rl_state.price_bucket) {
                     eprintln!("[RL_FILTER] {} ❌ Filtered YES at bucket {} (losing bucket per real trades)",
                               kalshi.ticker, rl_state.price_bucket);

                     // Log filtered trade to database
                     if let Some(ref db) = filtered_trades_db {
                         let time_bucket = get_time_bucket_from_hour(Utc::now().format("%H").to_string().parse().unwrap_or(12));
                         log_filtered_trade(
                             db,
                             &kalshi.ticker,
                             "yes",
                             kalshi.yes_ask,
                             rl_state.price_bucket,
                             time_bucket,
                             1, // side_dir for YES
                             "rl_bucket_filter",
                             &format!("losing YES bucket {} per real trades", rl_state.price_bucket),
                             None,
                             secs_left as i64,
                         );
                     }

                     let pending = crate::model::PendingCounterfactual {
                         ticker: kalshi.ticker.clone(),
                         state: rl_state.clone(),
                         yes_price: kalshi.yes_ask,
                         no_price: kalshi.no_ask,
                         bet_size,
                         timestamp: Utc::now(),
                     };
                     state.lock().unwrap().pending_counterfactuals.push(pending);
                     continue;
                 }
             }

             // NO: Block buckets 2, 3, 7 only (allow 0, 1, 4, 5, 6, 8, 9)
             // Based on actual PnL: bucket 2 -$928, bucket 3 -$540, bucket 7 -$210
             if action == 2 {  // BUY_NO
                 if rl_state.price_bucket == 2 || rl_state.price_bucket == 3 || rl_state.price_bucket == 7 {
                     eprintln!("[RL_FILTER] {} ❌ Filtered NO at bucket {} (losing bucket per real trades)",
                               kalshi.ticker, rl_state.price_bucket);

                     // Log filtered trade to database
                     if let Some(ref db) = filtered_trades_db {
                         let time_bucket = get_time_bucket_from_hour(Utc::now().format("%H").to_string().parse().unwrap_or(12));
                         log_filtered_trade(
                             db,
                             &kalshi.ticker,
                             "no",
                             kalshi.no_ask,
                             rl_state.price_bucket,
                             time_bucket,
                             0, // side_dir for NO
                             "rl_bucket_filter",
                             &format!("losing NO bucket {} per real trades", rl_state.price_bucket),
                             None,
                             secs_left as i64,
                         );
                     }

                     let pending = crate::model::PendingCounterfactual {
                         ticker: kalshi.ticker.clone(),
                         state: rl_state.clone(),
                         yes_price: kalshi.yes_ask,
                         no_price: kalshi.no_ask,
                         bet_size,
                         timestamp: Utc::now(),
                     };
                     state.lock().unwrap().pending_counterfactuals.push(pending);
                     continue;
                 }
             }

             match action {
                1 => ("yes".to_string(), kalshi.yes_ask, Some(rl_state)),
                2 => ("no".to_string(), kalshi.no_ask, Some(rl_state)),
                _ => {
                    eprintln!("[trade_executor] {} RL: HOLD (dist={:.2}% secs={} price={:.2})",
                        kalshi.ticker, distance_pct, secs_left as i64, kalshi.yes_ask);

                    // Store pending counterfactual for HER (Hindsight Experience Replay)
                    // When this market settles, we'll generate experiences for what
                    // WOULD have happened if we bought YES or NO
                    let pending = crate::model::PendingCounterfactual {
                        ticker: kalshi.ticker.clone(),
                        state: rl_state.clone(),
                        yes_price: kalshi.yes_ask,
                        no_price: kalshi.no_ask,
                        bet_size,  // Capture budget for counterfactual calculations
                        timestamp: Utc::now(),
                    };
                    state.lock().unwrap().pending_counterfactuals.push(pending);

                    continue;
                }
             }
        } else {
             // Fallback: If no policy loaded, use conservative hardcoded rule
             if kalshi.yes_ask <= 0.10 && distance_pct >= 0.3 && is_above_strike {
                 ("yes".to_string(), kalshi.yes_ask, None)
             } else {
                 eprintln!("[trade_executor] RL policy not loaded, skipping");
                 continue;
             }
        };
        
        // NOTE: Removed extreme price filter to allow full exploration
        // The RL will learn through experience which prices are profitable
        if entry_price >= 1.0 || entry_price <= 0.0 {
            // Only skip truly impossible prices (exactly 0 or 1)
            continue;
        }
        
        // Calculate edge for logging (how much we profit if we win)
        let edge = (1.0 - entry_price) / entry_price;
        
        eprintln!(
            "[trade_executor] RL decision: {} {} @ ${:.2} (dist={:.2}% secs={} edge={:.1}%)",
            kalshi.ticker, side, entry_price, distance_pct, secs_left, edge * 100.0
        );
        
        // Store model_prob for compatibility (use direction as probability)
        let model_prob = if is_above_strike { 0.9 } else { 0.1 };
        state.lock().unwrap().model_prob = Some(model_prob);
        last_kalshi_ts = Some(kalshi.ts);

        // --- HARD FILTER: Block long-shot trades < $0.10 for ALL assets --------
        // These trades have 0% win rate across all assets (SOL lost $489 on 8 trades today)
        if entry_price < 0.10 {
            eprintln!(
                "[LONGSHOT_FILTER] {} ❌ Blocked {} @ ${:.2} - extreme long-shot",
                kalshi.ticker, side.to_uppercase(), entry_price
            );
            // Log filtered trade to database for future analysis
            if let Some(ref db) = filtered_trades_db {
                let price_bucket = get_price_bucket(entry_price);
                let side_dir = if side == "yes" { 1 } else { 0 };
                let time_bucket = get_time_bucket_from_hour(Utc::now().format("%H").to_string().parse().unwrap_or(12));
                log_filtered_trade(
                    db,
                    &kalshi.ticker,
                    &side,
                    entry_price,
                    price_bucket,
                    time_bucket,
                    side_dir,
                    "longshot_filter",
                    "extreme long-shot < $0.10",
                    None,
                    secs_left as i64,
                );
            }
            continue;
        }

        // --- ACTUAL TRADES POLICY FILTER (Online Learning) -------------------
        // Check if this trade configuration is historically profitable
        if let Some(ref policy) = actual_trades_policy {
            let price_bucket = get_price_bucket(entry_price);
            let side_dir = if side == "yes" { 1 } else { 0 };
            let time_bucket = get_time_bucket_from_hour(Utc::now().format("%H").to_string().parse().unwrap_or(12));

            let (should_trade, reason) = should_trade_actual(policy, price_bucket, side_dir, time_bucket);

            if !should_trade {
                eprintln!(
                    "[ACTUAL_TRADES] {} ❌ Filtered {} @ ${:.2} bucket={} - {}",
                    kalshi.ticker, side.to_uppercase(), entry_price, price_bucket, reason
                );
                // Log filtered trade to database for future analysis
                if let Some(ref db) = filtered_trades_db {
                    // Get the avg_pnl from the policy entry if it exists
                    let key = format!("{},{},{}", price_bucket, side_dir, time_bucket);
                    let state_avg_pnl = policy.q_table.get(&key).map(|e| e.avg_pnl);
                    log_filtered_trade(
                        db,
                        &kalshi.ticker,
                        &side,
                        entry_price,
                        price_bucket,
                        time_bucket,
                        side_dir,
                        "actual_trades_filter",
                        &reason,
                        state_avg_pnl,
                        secs_left as i64,
                    );
                }
                continue;
            } else {
                eprintln!(
                    "[ACTUAL_TRADES] {} ✅ {} @ ${:.2} bucket={} - {}",
                    kalshi.ticker, side.to_uppercase(), entry_price, price_bucket, reason
                );
            }
        }

        if paper_mode {
            // --- Position guards -------------------------------------------
            {
                let app = state.lock().unwrap();
                let ticker_positions: Vec<_> = app.open_positions.iter()
                    .filter(|p| p.ticker == kalshi.ticker)
                    .collect();

                // Hard cap: max 3 positions per ticker, max 2 per side.
                let same_side_count = ticker_positions.iter().filter(|p| p.side == side).count();
                if ticker_positions.len() >= MAX_POSITIONS_PER_TICKER || same_side_count >= 2 {
                    continue;
                }

                // Block opposite-side fills entirely — prevents self-cancellation.
                // Same-side: require price moved >= MIN_PRICE_DELTA.
                let position_conflict = ticker_positions.iter().any(|p| {
                    if p.side != side {
                        true  // opposite side — always block
                    } else {
                        (p.entry_price - entry_price).abs() < MIN_PRICE_DELTA
                    }
                });
                if position_conflict {
                    continue;
                }
            }

            // --- Capital check ---------------------------------------------
            let available = state.lock().unwrap().available_capital;
            if available < bet_size {
                eprintln!("[trade_executor] insufficient capital: ${available:.2} < ${bet_size:.2}");
                continue;
            }

            // --- Filter 5: Calculate contracts with cap -----------------------
            let count = (bet_size / entry_price).floor() as u64;
            if count == 0 {
                continue;   // entry_price > bet_size — can't buy even 1
            }
            let capped_count = count.min(MAX_CONTRACTS_PER_POSITION);

            if capped_count < count {
                eprintln!(
                    "[trade_executor] {} contract limit: capped {count} → {capped_count} (entry_price=${:.4})",
                    kalshi.ticker, entry_price
                );
            }
            let count = capped_count;  // Use capped value for all subsequent calculations

            // --- Fee + total cost ------------------------------------------
            let entry_fee  = order_fee(count, entry_price);
            let total_cost = count as f64 * entry_price + entry_fee;

            // --- Debit capital ---------------------------------------------
            state.lock().unwrap().available_capital -= total_cost;

            // --- Paper fill ------------------------------------------------
            let pos = OpenPosition {
                ticker:      kalshi.ticker.clone(),
                strike:      kalshi.strike,
                side:        side.clone(),
                entry_price,
                entry_yes_bid: kalshi.yes_bid,  // Capture for reward shaping
                entry_yes_ask: kalshi.yes_ask,  // Capture for reward shaping
                entry_no_bid: kalshi.no_bid,    // Capture for counterfactuals
                entry_no_ask: kalshi.no_ask,    // Capture for counterfactuals
                entry_fee,
                bet_size,    // Capture budget for counterfactual calculations
                count,
                expiry_ts:   kalshi.expiry_ts,
                model_prob,
                edge,
                opened_at:   Utc::now(),
                entry_rl_state: entry_rl_state.clone(),
            };
            state.lock().unwrap().open_positions.push(pos);

            eprintln!(
                "[trade_executor] PLACED {} @ {:.3} | {} strike={} | secs_left={} | edge={:.3} | count={} | cost=${:.2} | capital=${:.2}",
                side.to_uppercase(), entry_price, kalshi.ticker, kalshi.strike, secs_left, edge, count, total_cost,
                state.lock().unwrap().available_capital
            );

            let asset  = extract_asset(&kalshi.ticker);
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
            // --- Live order ------------------------------------------------
            eprintln!("[trade_executor] edge={edge:.4} side={side} strike={} — placing order", kalshi.strike);
            match place_order(
                &client,
                &api_key_id,
                &signing_key,
                &base_url,
                &kalshi,
                &side,
                entry_price,
            )
            .await
            {
                Ok(order_json) => {
                    let order_id = order_json
                        .get("order")
                        .and_then(|o| o.get("id"))
                        .and_then(|id| id.as_str())
                        .unwrap_or("?");
                    eprintln!("[trade_executor] order created: {order_id}");

                    let asset  = extract_asset(&kalshi.ticker);
                    let expiry = format_expiry(kalshi.expiry_ts);
                    let msg = telegram_reporter::format_trade_entry(
                        asset,
                        kalshi.strike,
                        &side.to_uppercase(),
                        entry_price,
                        1,              // live mode: 1 contract
                        entry_price,    // total_cost = 1 × entry_price
                        model_prob,
                        edge,
                        &expiry,
                    );
                    state.lock().unwrap().notifications.push_back(msg);
                }
                Err(e) => {
                    let err_str = e.to_string();

                    // Auto-fallback to demo on DNS/connect failure.
                    if base_url == BASE_URL_PROD
                        && (err_str.contains("lookup") || err_str.contains("connect"))
                    {
                        eprintln!(
                            "[trade_executor] production unreachable, falling back to demo"
                        );
                        base_url = BASE_URL_DEMO.to_string();
                    }

                    eprintln!("[trade_executor] order failed: {e}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Settlement
// ---------------------------------------------------------------------------

/// Checks all open paper positions; settles any whose ticker has expired AND
/// has received official settlement data from Kalshi.
/// 
/// **CRITICAL FOR RL**: Uses Kalshi API to get official "yes"/"no" result
/// instead of calculating from BTC price. This ensures Q-values are trained
/// on accurate settlement data, not predictions that may differ from Kalshi.
async fn settle_expired(
    state: &Arc<Mutex<AppState>>,
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
) {
    // First pass: collect expired positions that need settlement check
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

    // Second pass: fetch settlement results from Kalshi API for each expired position
    // We process in reverse order to avoid index shifting issues with swap_remove
    let mut settled_indices: Vec<(usize, bool)> = Vec::new(); // (index, yes_won)
    
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
                            "[settlement] {} API result: status={} result={} (YES {})",
                            ticker, status, result_str, if yes_won { "WON" } else { "LOST" }
                        );
                    } else {
                        eprintln!(
                            "[settlement] {} status={} but no result yet, waiting...",
                            ticker, status
                        );
                    }
                } else {
                    eprintln!(
                        "[settlement] {} waiting for Kalshi settlement (status={})",
                        ticker, status
                    );
                }
            }
            None => {
                eprintln!(
                    "[settlement] {} API error fetching settlement, will retry...",
                    ticker
                );
            }
        }
    }

    // Third pass: process settled positions
    for (idx, yes_won) in settled_indices {
        let mut app = state.lock().unwrap();
        
        // Safety check: make sure index is still valid
        if idx >= app.open_positions.len() {
            continue;
        }

        // swap_remove is O(1)
        let pos = app.open_positions.swap_remove(idx);

        let won = match pos.side.as_str() {
            "yes" => yes_won,
            _     => !yes_won,  // "no" wins when yes_won is false
        };

        let exit_value   = if won { 1.0 } else { 0.0 };
        let total_cost   = pos.count as f64 * pos.entry_price + pos.entry_fee;
        let total_payout = pos.count as f64 * exit_value;
        let pnl          = total_payout - total_cost;
        let roi_pct      = if total_cost > 0.0 { pnl / total_cost * 100.0 } else { 0.0 };

        // === REWARD CALCULATION ===
        // Use raw PnL as reward (no shaping)
        let final_reward = pnl;

        // Credit capital with payout
        app.available_capital += total_payout;

        app.session_pnl  += pnl;
        app.total_trades += 1;
        if won {
            app.wins += 1;
        }

        app.pending_trades.push_back(TradeRecord {
            opened_at:   pos.opened_at,
            closed_at:   Utc::now(),
            strike:      pos.strike,
            side:        pos.side.to_uppercase(),
            entry_price: pos.entry_price,
            exit_price:  exit_value,
            pnl,
            roi:         roi_pct,
        });

        // Log experience for RL training (if RL state was captured at entry)
        if let Some(entry_state) = pos.entry_rl_state.clone() {
            let action = match pos.side.as_str() {
                "yes" => 1,
                "no"  => 2,
                _     => 0,  // Should not happen
            };

            // Clone data we need before mutable borrows
            let ticker = pos.ticker.clone();
            let side = pos.side.clone();

            // Record actual action taken
            let experience = Experience {
                timestamp: Utc::now(),
                ticker: ticker.clone(),
                state: entry_state.clone(),
                action,
                reward: final_reward,  // Use raw PnL as reward
                next_state: entry_state.clone(),  // Terminal state (same as entry)
            };
            app.experience_queue.push_back(experience);

            // HINDSIGHT EXPERIENCE REPLAY (HER):
            // Generate counterfactual experiences for actions we DIDN'T take

            let opposite_price = if side == "yes" {
                pos.entry_no_ask
            } else {
                pos.entry_yes_ask
            };

            let opposite_count = (pos.bet_size / opposite_price).floor() as u64;
            let opposite_count = opposite_count.min(1000);

            // Calculate counterfactual rewards based on official settlement
            let (cf_yes_reward, cf_no_reward, cf_hold_reward) = if won {
                match side.as_str() {
                    "yes" => (
                        pnl,
                        -(opposite_price * opposite_count as f64) - order_fee(opposite_count, opposite_price),
                        0.0,
                    ),
                    "no" => (
                        -(opposite_price * opposite_count as f64) - order_fee(opposite_count, opposite_price),
                        pnl,
                        0.0,
                    ),
                    _ => (0.0, 0.0, 0.0),
                }
            } else {
                match side.as_str() {
                    "yes" => (
                        pnl,
                        ((1.0 - opposite_price) * opposite_count as f64) - order_fee(opposite_count, opposite_price),
                        0.0,
                    ),
                    "no" => (
                        ((1.0 - opposite_price) * opposite_count as f64) - order_fee(opposite_count, opposite_price),
                        pnl,
                        0.0,
                    ),
                    _ => (0.0, 0.0, 0.0),
                }
            };

            // Generate counterfactual experiences for actions NOT taken
            if action != 1 {
                app.experience_queue.push_back(Experience {
                    timestamp: Utc::now(),
                    ticker: ticker.clone(),
                    state: entry_state.clone(),
                    action: 1,
                    reward: cf_yes_reward,
                    next_state: entry_state.clone(),
                });
            }
            if action != 2 {
                app.experience_queue.push_back(Experience {
                    timestamp: Utc::now(),
                    ticker: ticker.clone(),
                    state: entry_state.clone(),
                    action: 2,
                    reward: cf_no_reward,
                    next_state: entry_state.clone(),
                });
            }
            if action != 0 {
                app.experience_queue.push_back(Experience {
                    timestamp: Utc::now(),
                    ticker: ticker.clone(),
                    state: entry_state.clone(),
                    action: 0,
                    reward: cf_hold_reward,
                    next_state: entry_state.clone(),
                });
            }

            eprintln!(
                "[settlement] {} reward: pnl=${:.2} (API-verified)",
                ticker, pnl
            );
            eprintln!(
                "[HER] {} counterfactuals: YES=${:.2} NO=${:.2} HOLD=${:.2}",
                ticker, cf_yes_reward, cf_no_reward, cf_hold_reward
            );
        }

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
            "[trade_executor] SETTLED (API) {} {} strike={} — {} pnl={pnl:+.2} count={} capital=${:.2}",
            pos.side, pos.ticker, pos.strike,
            if won { "WIN" } else { "LOSS" },
            pos.count, app.available_capital
        );

        // HINDSIGHT EXPERIENCE REPLAY (HER) FOR HOLD STATES:
        // Process pending counterfactuals for this ticker using official settlement
        let pending_to_process: Vec<_> = app.pending_counterfactuals
            .iter()
            .filter(|p| p.ticker == pos.ticker)
            .cloned()
            .collect();

        for pending in &pending_to_process {
            // Use official settlement result
            let yes_contracts = (pending.bet_size / pending.yes_price).floor() as u64;
            let yes_contracts = yes_contracts.min(1000);
            let no_contracts = (pending.bet_size / pending.no_price).floor() as u64;
            let no_contracts = no_contracts.min(1000);

            let (cf_yes_reward, cf_no_reward) = if yes_won {
                (
                    ((1.0 - pending.yes_price) * yes_contracts as f64) - order_fee(yes_contracts, pending.yes_price),
                    -(pending.no_price * no_contracts as f64) - order_fee(no_contracts, pending.no_price),
                )
            } else {
                (
                    -(pending.yes_price * yes_contracts as f64) - order_fee(yes_contracts, pending.yes_price),
                    ((1.0 - pending.no_price) * no_contracts as f64) - order_fee(no_contracts, pending.no_price),
                )
            };

            app.experience_queue.push_back(Experience {
                timestamp: Utc::now(),
                ticker: pending.ticker.clone(),
                state: pending.state.clone(),
                action: 1,
                reward: cf_yes_reward,
                next_state: pending.state.clone(),
            });
            app.experience_queue.push_back(Experience {
                timestamp: Utc::now(),
                ticker: pending.ticker.clone(),
                state: pending.state.clone(),
                action: 2,
                reward: cf_no_reward,
                next_state: pending.state.clone(),
            });
            app.experience_queue.push_back(Experience {
                timestamp: Utc::now(),
                ticker: pending.ticker.clone(),
                state: pending.state.clone(),
                action: 0,
                reward: 0.0,
                next_state: pending.state.clone(),
            });

            eprintln!(
                "[HER-HOLD] {} generated counterfactuals (API): YES=${:.2} NO=${:.2} HOLD=$0.00",
                pending.ticker, cf_yes_reward, cf_no_reward
            );
        }

        // Remove processed counterfactuals
        app.pending_counterfactuals.retain(|p| p.ticker != pos.ticker);
    }
}

// ---------------------------------------------------------------------------
// Model inference
// ---------------------------------------------------------------------------

/// Returns the model probability and the inference source tag for the current
/// market tick.
///
/// Priority: ONNX model → paper heuristic (paper mode only) → None.
fn run_inference(
    model:        &mut Option<OnnxModel>,
    kalshi:       &MarketState,
    coinbase:     &IndexState,
    price_window: &VecDeque<(chrono::DateTime<chrono::Utc>, f64)>,
    tracker:      &MarketTracker,
    paper_mode:   bool,
    volatility_baseline: f64,
) -> Option<(f64, &'static str)> {
    // --- Try ONNX path first --------------------------------------------------
    if let Some(onnx) = model {
        if let Some(features) = compute_features(kalshi, coinbase, price_window, tracker, volatility_baseline) {
            if let Some(prob) = run_onnx(onnx, &features) {
                // NOTE: Removed extreme probability filter (prob < 0.001 || prob > 0.999)
                // High confidence is EXPECTED for deep ITM/OTM late-game scenarios.
                // The dynamic edge threshold already handles entry selection.
                return Some((prob, "onnx"));
            }
            // run_onnx already logged the error; fall through.
        }
        // price_window too small; fall through silently.
    }

    // --- Filter 4: Heuristic fallback disabled (USER DECISION) ---------------
    // ONNX failed - skip trade entirely (no heuristic fallback in paper or live mode)
    eprintln!("[trade_executor] {} ONNX failed, skipping trade (heuristic disabled)", kalshi.ticker);
    None
}

/// Simple sigmoid heuristic for paper mode.  Maps the distance between the
/// 60-second rolling average and the strike into a 0–1 probability.
///
/// `k = 20` means a 1 % price move relative to strike shifts the probability
/// by roughly 10 percentage points — aggressive enough to produce tradeable
/// edges against a market that occasionally misprices near-the-money contracts.
fn paper_heuristic(kalshi: &MarketState, coinbase: &IndexState) -> f64 {
    let k = 20.0;
    let x = k * (coinbase.rolling_avg_60s - kalshi.strike) / kalshi.strike;
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Edge & fee calculation  (spec §5)
// ---------------------------------------------------------------------------

/// Fee impact in probability space (0.0–1.0) for a single contract.
///
/// Kalshi's documented fee formula (applied per-contract at execution):
///   `⌈0.07 × price_cents × (100 − price_cents) / 100⌉` cents
///
/// With `price` already in the 0.0–1.0 range this simplifies to
///   `⌈7 × price × (1 − price)⌉` cents
/// which is then divided by 100 to stay in probability space.
fn fee_impact(price: f64) -> f64 {
    let fee_cents = (7.0 * price * (1.0 - price)).ceil();
    fee_cents / 100.0
}

/// Kalshi taker fee on an order of `count` contracts at `price` (0–1).
/// Kalshi applies ceil to the *total* order (not per-contract), rounding up
/// to the nearest cent.
fn order_fee(count: u64, price: f64) -> f64 {
    (0.07 * count as f64 * price * (1.0 - price) * 100.0).ceil() / 100.0
}

// ---------------------------------------------------------------------------
// Kalshi REST order placement
// ---------------------------------------------------------------------------

/// POSTs a limit-buy order to Kalshi.  Returns the parsed JSON response body
/// on HTTP 201; returns an error for any other status.
async fn place_order(
    client: &reqwest::Client,
    api_key_id: &str,
    signing_key: &SigningKey<Sha256>,
    base_url: &str,
    kalshi: &MarketState,
    side: &str,         // "yes" | "no"
    entry_price: f64,   // 0.0–1.0
) -> anyhow::Result<serde_json::Value> {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let signature = sign_request(signing_key, timestamp_ms, "POST", REST_PATH);

    // Kalshi prices are in whole cents (integer 1–99).
    let price_cents = (entry_price * 100.0).round() as i64;

    let mut body = serde_json::json!({
        "ticker":          kalshi.ticker,
        "action":          "buy",
        "side":            side,
        "count":           1,
        "type":            "limit",
        "client_order_id": client_order_id(),
    });

    // Price field name is side-specific.
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

/// Signs a Kalshi API request.  The signed message is
/// `"{timestamp_ms}{METHOD}{path}"`.  The request body is NOT included in the
/// signature (per Kalshi auth spec).
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

/// Generates a UUID v4 string for use as `client_order_id` (deduplication).
fn client_order_id() -> String {
    let mut rng = OsRng;
    let a: u32 = rng.gen();
    let b: u16 = rng.gen();
    let c: u16 = (rng.gen::<u16>() & 0x0fff) | 0x4000; // version 4
    let d: u16 = (rng.gen::<u16>() & 0x3fff) | 0x8000; // variant 1
    let e: u64 = rng.gen::<u64>() & 0x0000_ffff_ffff_ffff;
    format!("{a:08x}-{b:04x}-{c:04x}-{d:04x}-{e:012x}")
}

/// Extracts the asset symbol (BTC / ETH / SOL) from a Kalshi crypto ticker.
fn extract_asset(ticker: &str) -> &str {
    if ticker.contains("BTC") {
        "BTC"
    } else if ticker.contains("ETH") {
        "ETH"
    } else if ticker.contains("SOL") {
        "SOL"
    } else if ticker.contains("XRP") {
        "XRP"
    } else {
        "?"
    }
}

/// Formats expiry as "HH:MM EST" for Telegram notifications using the actual
/// expiry timestamp from the REST API (avoids parsing the ticker string).
fn format_expiry(expiry_ts: Option<chrono::DateTime<chrono::Utc>>) -> String {
    match expiry_ts {
        Some(dt) => {
            let est = chrono::FixedOffset::west_opt(5 * 3600).unwrap();
            dt.with_timezone(&est).format("%H:%M EST").to_string()
        }
        None => "??:??".to_string(),
    }
}
