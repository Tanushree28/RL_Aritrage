//! Ensemble supervisor executor — polls shared state and only trades when
//! multiple policies agree (consensus voting).
//!
//! ## Voting
//! Each tick, up to 2 policies vote on the best action:
//!   Voter 1: RL Q-table (`model/{asset}/rl_policy.json`)
//!   Voter 2: Actual trades policy (`model/{asset}/rl_policy_actual.json`)
//!
//! Consensus thresholds:
//!   - Entry price $0.90–$0.99 → require ALL voters to agree (high-certainty range)
//!   - All other prices       → require min(3, voters.len()) agreement
//!
//! ## Bet sizing
//! Two methods computed simultaneously; the smaller (more conservative) is used:
//!   - Graduated:            base_bet × (agreeing / total)
//!   - Confidence-weighted:  base_bet × avg_win_rate_of_agreeing_voters

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use base64::Engine;
use chrono::{Timelike, Utc};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use sha2::Sha256;
use tokio::time::Duration;

use crate::model::{AppState, AssetConfig, OpenPosition, RlState, TradeRecord};
use crate::telegram_reporter;

// ============================================================================
// Constants (same as trade_executor)
// ============================================================================

const MIN_ENTRY_PRICE: f64 = 0.10;
const MAX_CONTRACTS_PER_POSITION: u64 = 1000;
const MAX_POSITIONS_PER_TICKER: usize = 3;
const MIN_PRICE_DELTA: f64 = 0.05;
const REST_PATH: &str = "/trade-api/v2/portfolio/orders";
const BASE_URL_PROD: &str = "https://api.elections.kalshi.com";
const BASE_URL_DEMO: &str = "https://demo-api.kalshi.co";

// ============================================================================
// RL Policy types (mirrors trade_executor structs)
// ============================================================================

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
    #[allow(dead_code)]
    timestamp: f64,
}

/// Actual trades policy entry
#[derive(serde::Deserialize, Debug, Clone)]
struct ActualTradesEntry {
    avg_pnl:    f64,
    count:      i32,
    #[allow(dead_code)]
    confidence: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct ActualTradesThresholds {
    min_avg_pnl:          f64,
    min_observations:     i32,
    #[allow(dead_code)]
    avoid_losing_states:  bool,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct ActualTradesPolicy {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    policy_type: String,
    q_table:     HashMap<String, ActualTradesEntry>,
    thresholds:  ActualTradesThresholds,
    #[allow(dead_code)]
    timestamp:   f64,
}

// ============================================================================
// Voter abstraction
// ============================================================================

enum PolicyKind {
    RlQTable(RlPolicyData),
    ActualTrades(ActualTradesPolicy),
}

struct Voter {
    name:     String,
    policy:   PolicyKind,
    win_rate: f64, // used for confidence-weighted bet sizing
}

/// Strip "ensemble-" prefix to get underlying asset name for model file paths.
fn underlying_asset(asset: &str) -> &str {
    asset.strip_prefix("ensemble-").unwrap_or(asset)
}

/// Load the RL Q-table voter. Returns None on missing/parse error.
fn load_rl_voter(asset: &str) -> Option<Voter> {
    let path = format!("model/{}/rl_policy.json", underlying_asset(asset));
    let text = std::fs::read_to_string(&path).ok()?;
    let policy: RlPolicyData = serde_json::from_str(&text).ok()?;
    let win_rate = load_win_rate(asset, "rl");
    eprintln!("[ensemble] ✅ voter 'rl_policy' loaded (win_rate={:.2})", win_rate);
    Some(Voter { name: "rl_policy".to_string(), policy: PolicyKind::RlQTable(policy), win_rate })
}

/// Load the actual trades voter. Returns None on missing/parse error.
fn load_actual_voter(asset: &str) -> Option<Voter> {
    let path = format!("model/{}/rl_policy_actual.json", underlying_asset(asset));
    let text = std::fs::read_to_string(&path).ok()?;
    let policy: ActualTradesPolicy = serde_json::from_str(&text).ok()?;
    let win_rate = load_win_rate(asset, "actual");
    eprintln!("[ensemble] ✅ voter 'rl_policy_actual' loaded (win_rate={:.2})", win_rate);
    Some(Voter { name: "rl_policy_actual".to_string(), policy: PolicyKind::ActualTrades(policy), win_rate })
}

/// Try to read win_rate from model_registry/{asset}/current/metadata.json.
/// Falls back to 0.5 if unavailable.
fn load_win_rate(asset: &str, _policy_name: &str) -> f64 {
    let path = format!("model_registry/{}/current/metadata.json", underlying_asset(asset));
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return 0.5,
    };
    let json: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return 0.5,
    };
    json.get("win_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5)
        .clamp(0.0, 1.0)
}

// ============================================================================
// State discretization (mirrors trade_executor exactly)
// ============================================================================

fn get_time_of_day_bucket() -> u8 {
    let hour = Utc::now().hour();
    if hour < 6 { 0 } else if hour < 12 { 1 } else if hour < 18 { 2 } else { 3 }
}

fn discretize_state(policy: &RlPolicyData, dist_pct: f64, secs: i64, price: f64, is_above: bool, spread: f64, momentum: f64) -> RlState {
    let d_idx  = policy.buckets.distance.iter().filter(|&&b| dist_pct >= b).count();
    let t_idx  = policy.buckets.time.iter().filter(|&&b| secs <= b).count();
    let p_idx  = policy.buckets.price.iter().filter(|&&b| price >= b).count();
    let dir    = if is_above { 1 } else { 0 };
    let sp_idx = if spread > 0.05 { 1u8 } else { 0u8 };
    let mom    = if momentum < -0.1 { 0u8 } else if momentum > 0.1 { 2u8 } else { 1u8 };
    RlState {
        dist_bucket: d_idx,
        time_bucket: t_idx,
        price_bucket: p_idx,
        direction: dir,
        spread_bucket: sp_idx,
        momentum_bucket: mom,
        time_of_day_bucket: get_time_of_day_bucket(),
    }
}

fn state_key(s: &RlState) -> String {
    // 5D key — must match Python rl_strategy.py format: "d,t,p,dir,mom"
    format!("{},{},{},{},{}", s.dist_bucket, s.time_bucket, s.price_bucket, s.direction, s.momentum_bucket)
}

fn get_price_bucket(price: f64) -> usize {
    ((price * 10.0).floor() as usize).min(9)
}

fn get_time_bucket_from_hour(hour: u32) -> usize {
    if hour < 6 { 0 } else if hour < 12 { 1 } else if hour < 18 { 2 } else { 3 }
}

// ============================================================================
// Per-voter action derivation
// ============================================================================

/// Derive an action (0=HOLD, 1=BUY_YES, 2=BUY_NO) from a voter given the
/// current market state. Always deterministic (no epsilon exploration).
fn get_vote(voter: &Voter, rl_state: &RlState, entry_price: f64, side: &str) -> u8 {
    match &voter.policy {
        PolicyKind::RlQTable(policy) => {
            let key = state_key(rl_state);
            if let Some(q_vals) = policy.q_table.get(&key) {
                // Argmax — but skip action 0 (HOLD) unless it's the only choice
                let mut best_action = 0usize;
                let mut best_q = f64::NEG_INFINITY;
                for (i, &q) in q_vals.iter().enumerate() {
                    if q > best_q {
                        best_q = q;
                        best_action = i;
                    }
                }
                best_action as u8
            } else {
                // State not in Q-table → conservative HOLD
                0
            }
        }
        PolicyKind::ActualTrades(policy) => {
            let price_bucket = get_price_bucket(entry_price);
            let hour = Utc::now().hour();
            let time_bucket = get_time_bucket_from_hour(hour);

            // Check if the candidate side (from RL voter) is historically profitable
            let side_dir: usize = if side == "yes" { 1 } else { 2 };
            let key = format!("{},{},{}", price_bucket, side_dir, time_bucket);

            match policy.q_table.get(&key) {
                Some(entry) if entry.count >= policy.thresholds.min_observations => {
                    if entry.avg_pnl >= policy.thresholds.min_avg_pnl {
                        // Historically profitable state — agree with the side
                        if side == "yes" { 1 } else { 2 }
                    } else {
                        0 // Losing state — vote HOLD
                    }
                }
                _ => {
                    // Unknown state or insufficient data — give benefit of doubt, vote with side
                    if side == "yes" { 1 } else { 2 }
                }
            }
        }
    }
}

// ============================================================================
// Consensus voting
// ============================================================================

struct VoteResult {
    action:    u8,    // 1=BUY_YES or 2=BUY_NO
    bet_size:  f64,
    vote_count: usize,
    total_voters: usize,
}

/// Run all voters and return Some(VoteResult) if consensus is reached.
/// The candidate action comes from voter 1 (RL Q-table); other voters
/// confirm or veto that action.
fn run_vote(voters: &[Voter], rl_state: &RlState, entry_price: f64, base_bet: f64) -> Option<VoteResult> {
    if voters.is_empty() {
        return None;
    }

    // Voter 1 (RL Q-table) determines the candidate action
    let candidate_action = get_vote(&voters[0], rl_state, entry_price, "yes");

    // HOLD from voter 1 → no trade
    if candidate_action == 0 {
        return None;
    }

    let candidate_side = if candidate_action == 1 { "yes" } else { "no" };

    // Count how many voters agree on this action
    let mut agree_count = 0usize;
    let mut agree_win_rate_sum = 0.0f64;
    let total = voters.len();

    for voter in voters {
        let vote = get_vote(voter, rl_state, entry_price, candidate_side);
        if vote == candidate_action {
            agree_count += 1;
            agree_win_rate_sum += voter.win_rate;
        }
    }

    // Threshold: require all voters for $0.90-$0.99 range, else min(3, total)
    let threshold = if entry_price >= 0.90 {
        total // unanimous required in bucket9 range
    } else {
        3_usize.min(total)
    };

    if agree_count < threshold {
        eprintln!(
            "[ensemble] no consensus — action={} agree={}/{} threshold={}",
            candidate_action, agree_count, total, threshold
        );
        return None;
    }

    // Bet sizing: graduated and confidence-weighted, take the smaller
    let graduated   = base_bet * (agree_count as f64 / total as f64);
    let avg_wr      = agree_win_rate_sum / agree_count as f64;
    let conf_sized  = base_bet * avg_wr;
    let final_bet   = graduated.min(conf_sized).max(1.0); // floor at $1

    eprintln!(
        "[ensemble] ✅ consensus={}/{} action={} bet=${:.2} (grad=${:.2} conf=${:.2})",
        agree_count, total, candidate_action, final_bet, graduated, conf_sized
    );

    Some(VoteResult {
        action: candidate_action,
        bet_size: final_bet,
        vote_count: agree_count,
        total_voters: total,
    })
}

// ============================================================================
// Order fee (Kalshi taker fee formula, same as trade_executor)
// ============================================================================

fn order_fee(count: u64, price: f64) -> f64 {
    let raw = 0.07 * (count as f64) * price * (1.0 - price);
    raw.min(0.35 * (count as f64) * price)
}

// ============================================================================
// Live order placement (mirrors trade_executor::place_order)
// ============================================================================

async fn place_order(
    client:      &reqwest::Client,
    api_key_id:  &str,
    signing_key: &SigningKey<Sha256>,
    base_url:    &str,
    ticker:      &str,
    side:        &str,
    price:       f64,
) -> anyhow::Result<serde_json::Value> {
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let nonce = format!("{}", ts_ms);
    let message = format!("{}{}{}", ts_ms, "POST", REST_PATH);
    let mut rng = rand::rngs::OsRng;
    let sig_bytes = signing_key.sign_with_rng(&mut rng, message.as_bytes());
    let sig_b64 = base64::engine::general_purpose::STANDARD.encode(sig_bytes.to_bytes());
    let price_int = (price * 100.0).round() as u64;

    let body = serde_json::json!({
        "ticker": ticker,
        "action": "buy",
        "side":   side,
        "count":  1,
        "type":   "limit",
        "yes_price": if side == "yes" { price_int } else { 100 - price_int },
        "no_price":  if side == "no"  { price_int } else { 100 - price_int },
        "expiration_ts": 0,
        "sell_position_floor": 0,
        "buy_max_cost": (price * 100.0).round() as u64,
    });

    let url = format!("{}{}", base_url, REST_PATH);
    let resp = client
        .post(&url)
        .header("KALSHI-ACCESS-KEY", api_key_id)
        .header("KALSHI-ACCESS-SIGNATURE", &sig_b64)
        .header("KALSHI-ACCESS-TIMESTAMP", &nonce)
        .json(&body)
        .send()
        .await
        .context("order POST failed")?;

    let status = resp.status();
    let json: serde_json::Value = resp.json().await.context("order response parse failed")?;

    if !status.is_success() {
        anyhow::bail!("order rejected ({status}): {json}");
    }
    Ok(json)
}

// ============================================================================
// Settle expired paper positions (mirrors trade_executor)
// ============================================================================

async fn settle_expired(state: &Arc<Mutex<AppState>>, config_name: &str) {
    let now = Utc::now();
    let expired: Vec<OpenPosition> = {
        let app = state.lock().unwrap();
        app.open_positions
            .iter()
            .filter(|p| p.expiry_ts.map_or(false, |e| e <= now))
            .cloned()
            .collect()
    };

    for pos in expired {
        let settle_price = {
            let app = state.lock().unwrap();
            app.coinbase.as_ref().map(|c| c.current_price)
        };

        let Some(spot) = settle_price else { continue };

        // Determine win/loss: YES wins if spot > strike, NO wins if spot <= strike
        let win = (pos.side == "yes" && spot > pos.strike)
            || (pos.side == "no"  && spot <= pos.strike);

        let exit_price = if win { 1.0 } else { 0.0 };
        let gross      = pos.count as f64 * exit_price;
        let fee        = order_fee(pos.count, exit_price);
        let net        = gross - fee - (pos.count as f64 * pos.entry_price) - pos.entry_fee;
        let roi        = net / (pos.count as f64 * pos.entry_price + pos.entry_fee);

        let settlement = if spot > pos.strike { "YES" } else { "NO" }.to_string();
        let record = TradeRecord {
            opened_at:   pos.opened_at,
            closed_at:   now,
            asset:       config_name.to_string(),
            strike:      pos.strike,
            side:        pos.side.clone(),
            entry_price: pos.entry_price,
            exit_price,
            pnl:         net,
            roi,
            settlement,
            fair_value:  pos.model_prob,
        };

        {
            let mut app = state.lock().unwrap();
            app.open_positions.retain(|p| p.opened_at != pos.opened_at || p.ticker != pos.ticker);
            app.pending_trades.push_back(record.clone());
            app.session_pnl += net;
            app.available_capital += gross - fee;
            if win { app.wins += 1; }
            app.total_trades += 1;
        }

        eprintln!(
            "[ensemble] settled {} {} @ entry={:.3} exit={:.3} pnl=${:.2}",
            pos.ticker, pos.side.to_uppercase(), pos.entry_price, exit_price, net
        );
    }
}

// ============================================================================
// Helper: extract asset label from ticker (e.g. "KXBTC15M-..." → "BTC")
// ============================================================================

fn extract_asset(ticker: &str) -> &str {
    if ticker.contains("BTC") { "BTC" }
    else if ticker.contains("ETH") { "ETH" }
    else if ticker.contains("SOL") { "SOL" }
    else if ticker.contains("XRP") { "XRP" }
    else { "?" }
}

fn format_expiry(ts: Option<chrono::DateTime<Utc>>) -> String {
    ts.map_or_else(|| "unknown".to_string(), |t| t.format("%H:%M UTC").to_string())
}

// ============================================================================
// Main task
// ============================================================================

pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[ensemble] task started (asset: {})", config.name);

    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));
    let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);
    let base_bet: f64 = std::env::var("BET_SIZE")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .unwrap_or(10.0);

    if paper_mode {
        state.lock().unwrap().available_capital = starting_capital;
        eprintln!("[ensemble] PAPER MODE — fills simulated  capital=${starting_capital:.2}  base_bet=${base_bet:.2}");
    }

    let api_key_id = std::env::var("KALSHI_API_KEY_ID")?;
    let private_key_pem = std::env::var("KALSHI_PRIVATE_KEY")?;
    let private_key = RsaPrivateKey::from_pkcs1_pem(&private_key_pem)
        .context("Failed to parse KALSHI_PRIVATE_KEY as PKCS#1 PEM")?;
    let signing_key = SigningKey::<Sha256>::new(private_key);
    let client = reqwest::Client::new();
    let mut base_url = config.api_base_url.clone();

    // --- Load voters at startup -----------------------------------------------
    let mut voters: Vec<Voter> = Vec::new();
    if let Some(v) = load_rl_voter(&config.name)     { voters.push(v); }
    if let Some(v) = load_actual_voter(&config.name) { voters.push(v); }

    if voters.is_empty() {
        anyhow::bail!("[ensemble] No policy files found — cannot run without at least one voter");
    }
    eprintln!("[ensemble] {} voter(s) loaded", voters.len());

    let mut last_policy_reload = std::time::Instant::now();
    let mut market_trackers: HashMap<String, crate::model::MarketTracker> = HashMap::new();
    let mut last_tracker_cleanup = Utc::now();
    let mut last_kalshi_ts: Option<chrono::DateTime<Utc>> = None;

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[ensemble] shutdown requested, exiting");
            return Ok(());
        }

        // --- Snapshot shared state --------------------------------------------
        let (kalshi, coinbase) = {
            let app = state.lock().unwrap();
            (app.kalshi.clone(), app.coinbase.clone())
        };
        let (kalshi, coinbase) = match (kalshi, coinbase) {
            (Some(k), Some(c)) => (k, c),
            _ => continue,
        };

        // --- Settle expired paper positions -----------------------------------
        if paper_mode {
            settle_expired(&state, &config.name).await;
        }

        // Only trade the configured asset's 15-min series
        let series_prefix = format!("{}-", config.kalshi_series);
        if !kalshi.ticker.starts_with(&series_prefix) {
            continue;
        }

        // --- Update market tracker with latest price --------------------------
        let spot = coinbase.current_price;
        market_trackers.entry(kalshi.ticker.clone()).or_insert_with(crate::model::MarketTracker::new);
        for tracker in market_trackers.values_mut() {
            tracker.price_history.push_back((Utc::now(), spot));
            if tracker.price_history.len() > 600 {
                tracker.price_history.pop_front();
            }
        }

        // --- Periodic cleanup of stale trackers ------------------------------
        if (Utc::now() - last_tracker_cleanup).num_seconds() > 60 {
            let now = Utc::now();
            market_trackers.retain(|_, t| {
                t.price_history.back().map_or(false, |(ts, _)| (now - *ts).num_seconds() < 1200)
            });
            last_tracker_cleanup = now;
        }

        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts);

        // Only trade in last 5 minutes (matches RL training window)
        if secs_left < 30 || secs_left > 300 {
            if secs_left < 30 { market_trackers.remove(&kalshi.ticker); }
            continue;
        }

        // --- Spread sanity check (same as trade_executor) --------------------
        let spread = kalshi.yes_ask - kalshi.yes_bid;
        if spread > 0.05 {
            continue;
        }

        // --- Dedup on kalshi tick --------------------------------------------
        if last_kalshi_ts == Some(kalshi.ts) {
            continue;
        }

        // --- Periodic voter reload (every 5 min) ----------------------------
        if last_policy_reload.elapsed() > Duration::from_secs(300) {
            voters.clear();
            if let Some(v) = load_rl_voter(&config.name)     { voters.push(v); }
            if let Some(v) = load_actual_voter(&config.name) { voters.push(v); }
            eprintln!("[ensemble] voters reloaded ({} voters)", voters.len());
            last_policy_reload = std::time::Instant::now();
        }

        if voters.is_empty() {
            continue;
        }

        // --- Compute momentum (% change over last 30 ticks) ------------------
        let momentum = market_trackers
            .get(&kalshi.ticker)
            .and_then(|t| {
                if t.price_history.len() >= 30 {
                    let recent: Vec<_> = t.price_history.iter().rev().take(30).collect();
                    if let (Some((_, newest)), Some((_, oldest))) = (recent.first(), recent.last()) {
                        Some((newest - oldest) / oldest * 100.0)
                    } else { None }
                } else { None }
            })
            .unwrap_or(0.0);

        // --- Ensure obs_summary is populated ---------------------------------
        {
            let tracker = market_trackers.get_mut(&kalshi.ticker).unwrap();
            if tracker.obs_summary.is_none() {
                if tracker.price_history.len() < 2 { continue; }
                let first_ts = tracker.price_history.front().unwrap().0;
                let last_ts  = tracker.price_history.back().unwrap().0;
                if (last_ts - first_ts).num_seconds() < 60 { continue; }
                let prices: Vec<f64> = tracker.price_history.iter().map(|(_, p)| *p).collect();
                let n     = prices.len() as f64;
                let mean  = prices.iter().sum::<f64>() / n;
                let high  = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let low   = prices.iter().cloned().fold(f64::INFINITY, f64::min);
                tracker.obs_summary   = Some(crate::model::ObsSummary { mean, high, low, range: high - low });
                tracker.obs_end_price = Some(spot);
                tracker.obs_end_time  = Some(Utc::now());
            }
        }

        // --- Obs cooldown: skip first 30s after phase transition -------------
        {
            let tracker = market_trackers.get(&kalshi.ticker).unwrap();
            if let Some(obs_end_time) = tracker.obs_end_time {
                if (Utc::now() - obs_end_time).num_seconds() < 30 { continue; }
            }
        }

        // --- Discretize RL state (needs RL voter loaded) ---------------------
        let rl_state = {
            // Get rl policy buckets from voter 0 to discretize (same buckets for all)
            let rl_policy_data = match &voters[0].policy {
                PolicyKind::RlQTable(p) => p,
                PolicyKind::ActualTrades(_) => continue, // voter 0 must be RL Q-table
            };

            let dist_pct     = ((spot - kalshi.strike) / kalshi.strike).abs() * 100.0;
            let is_above     = spot > kalshi.strike;
            discretize_state(rl_policy_data, dist_pct, secs_left as i64, kalshi.yes_ask, is_above, spread, momentum)
        };

        // --- Hard filter: block markets where BOTH sides are cheap -----------
        // Use max(yes_ask, no_ask) so we don't block valid NO trades when YES is cheap
        let best_tradeable_price = kalshi.yes_ask.max(kalshi.no_ask);
        if best_tradeable_price < MIN_ENTRY_PRICE {
            continue;
        }

        // --- Run ensemble vote -----------------------------------------------
        let Some(result) = run_vote(&voters, &rl_state, kalshi.yes_ask, base_bet) else {
            continue;
        };

        let (side, entry_price) = match result.action {
            1 => ("yes", kalshi.yes_ask),
            2 => ("no",  kalshi.no_ask),
            _ => continue,
        };

        let bet_size   = result.bet_size;
        let edge       = (1.0 - entry_price) / entry_price;
        let model_prob = if spot > kalshi.strike { 0.9 } else { 0.1 };

        last_kalshi_ts = Some(kalshi.ts);
        state.lock().unwrap().model_prob = Some(model_prob);

        eprintln!(
            "[ensemble] DECISION {} {} @ ${:.2} | votes={}/{} | bet=${:.2} | secs={}",
            kalshi.ticker, side.to_uppercase(), entry_price,
            result.vote_count, result.total_voters, bet_size, secs_left
        );

        if entry_price >= 1.0 || entry_price <= 0.0 { continue; }

        if paper_mode {
            // --- Position guards -------------------------------------------
            {
                let app = state.lock().unwrap();
                let ticker_positions: Vec<_> = app.open_positions
                    .iter()
                    .filter(|p| p.ticker == kalshi.ticker)
                    .collect();
                let same_side = ticker_positions.iter().filter(|p| p.side == side).count();
                if ticker_positions.len() >= MAX_POSITIONS_PER_TICKER || same_side >= 2 {
                    continue;
                }
                let conflict = ticker_positions.iter().any(|p| {
                    if p.side != side { true }
                    else { (p.entry_price - entry_price).abs() < MIN_PRICE_DELTA }
                });
                if conflict { continue; }
            }

            // --- Capital check ---------------------------------------------
            let available = state.lock().unwrap().available_capital;
            if available < bet_size {
                eprintln!("[ensemble] insufficient capital: ${available:.2} < ${bet_size:.2}");
                continue;
            }

            let count = (bet_size / entry_price).floor() as u64;
            if count == 0 { continue; }
            let count = count.min(MAX_CONTRACTS_PER_POSITION);

            let entry_fee  = order_fee(count, entry_price);
            let total_cost = count as f64 * entry_price + entry_fee;

            state.lock().unwrap().available_capital -= total_cost;

            let pos = OpenPosition {
                ticker:        kalshi.ticker.clone(),
                strike:        kalshi.strike,
                side:          side.to_string(),
                entry_price,
                entry_yes_bid: kalshi.yes_bid,
                entry_yes_ask: kalshi.yes_ask,
                entry_no_bid:  kalshi.no_bid,
                entry_no_ask:  kalshi.no_ask,
                entry_fee,
                bet_size,
                count,
                expiry_ts:     kalshi.expiry_ts,
                model_prob,
                edge,
                opened_at:     Utc::now(),
                entry_rl_state: Some(rl_state.clone()),
            };
            state.lock().unwrap().open_positions.push(pos);

            let asset  = extract_asset(&kalshi.ticker);
            let expiry = format_expiry(kalshi.expiry_ts);
            let msg = telegram_reporter::format_trade_entry(
                asset, kalshi.strike, &side.to_uppercase(),
                entry_price, count, total_cost, model_prob, edge, &expiry,
            );
            state.lock().unwrap().notifications.push_back(msg);

            eprintln!(
                "[ensemble] PLACED {} {} @ {:.3} | count={} cost=${:.2} | capital=${:.2}",
                kalshi.ticker, side.to_uppercase(), entry_price, count, total_cost,
                state.lock().unwrap().available_capital
            );
        } else {
            // --- Live order ------------------------------------------------
            match place_order(&client, &api_key_id, &signing_key, &base_url, &kalshi.ticker, side, entry_price).await {
                Ok(order_json) => {
                    let order_id = order_json
                        .get("order").and_then(|o| o.get("id"))
                        .and_then(|id| id.as_str()).unwrap_or("?");
                    eprintln!("[ensemble] live order created: {order_id}");

                    let asset  = extract_asset(&kalshi.ticker);
                    let expiry = format_expiry(kalshi.expiry_ts);
                    let msg = telegram_reporter::format_trade_entry(
                        asset, kalshi.strike, &side.to_uppercase(),
                        entry_price, 1, entry_price, model_prob, edge, &expiry,
                    );
                    state.lock().unwrap().notifications.push_back(msg);
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if base_url == BASE_URL_PROD
                        && (err_str.contains("lookup") || err_str.contains("connect"))
                    {
                        eprintln!("[ensemble] production unreachable, falling back to demo");
                        base_url = BASE_URL_DEMO.to_string();
                    }
                    eprintln!("[ensemble] live order failed: {e}");
                }
            }
        }
    }
}
