use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use crate::black_scholes::EwmaVolatilityTracker;

/// Asset-specific configuration for trading different cryptocurrencies
#[derive(Debug, Clone)]
pub struct AssetConfig {
    pub name: String,
    pub kalshi_series: String,
    pub kraken_symbol: String,
    pub coinbase_pair: String,
    pub volatility_baseline: f64,
    pub api_base_url: String, // "https://api.elections.kalshi.com" or "https://demo-api.kalshi.co"
    pub min_fair_value: f64,  // Minimum B-S fair value for Oracle bot entry
}

/// Returns the number of seconds remaining until the contract's expiry.
/// The expiry timestamp is fetched from the Kalshi REST API when the event
/// is first seen and stored in `MarketState::expiry_ts`.
pub fn seconds_to_expiry(expiry: Option<DateTime<Utc>>) -> i32 {
    match expiry {
        Some(exp) => (exp - Utc::now()).num_seconds() as i32,
        None      => 0,
    }
}

/// A single tick from the Kalshi market WebSocket.
#[derive(Debug, Clone, Serialize)]
pub struct MarketState {
    /// Full market ticker, e.g. `KXBTCUSD15M-260131-1545-96000`.
    pub ticker:  String,
    pub strike:  f64,
    pub yes_ask: f64,
    pub no_ask:  f64,
    pub yes_bid: f64,
    pub no_bid:    f64,
    /// Contract close time from the Kalshi REST API (set once per event).
    pub expiry_ts: Option<DateTime<Utc>>,
    pub ts:        DateTime<Utc>,
}

/// A completed trade, ready for SQLite persistence.  Pushed onto
/// `AppState::pending_trades` by the trade executor; drained and inserted
/// by `data_recorder`.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub opened_at:   DateTime<Utc>,
    pub closed_at:   DateTime<Utc>,
    pub asset:       String,   // "btc", "eth", "sol", "xrp", "oracle-btc", etc.
    pub strike:      f64,
    pub side:        String,   // "YES" or "NO"
    pub entry_price: f64,
    pub exit_price:  f64,
    pub pnl:         f64,
    pub roi:         f64,
    pub settlement:  String,   // "YES" or "NO" - actual Kalshi settlement result
    pub fair_value:  f64,      // Model's fair value at trade entry (for analysis)
}

/// An open paper-traded position awaiting settlement.
#[derive(Debug, Clone)]
pub struct OpenPosition {
    pub ticker:      String,
    pub strike:      f64,
    pub side:        String,   // "yes" | "no"
    pub entry_price: f64,
    pub entry_yes_bid: f64,  // YES bid at entry (for reward shaping)
    pub entry_yes_ask: f64,  // YES ask at entry (for reward shaping)
    pub entry_no_bid: f64,   // NO bid at entry (for counterfactuals)
    pub entry_no_ask: f64,   // NO ask at entry (for counterfactuals)
    pub entry_fee:   f64,     // total Kalshi taker fee paid at entry
    pub bet_size:    f64,     // Budget allocated to this trade (for counterfactuals)
    pub count:       u64,     // number of contracts in this position
    pub expiry_ts:   Option<DateTime<Utc>>,
    pub model_prob:  f64,
    pub edge:        f64,
    pub opened_at:   DateTime<Utc>,
    /// RL state at entry (for experience logging)
    pub entry_rl_state: Option<RlState>,
}

/// Discretized RL state for Q-learning.
/// Matches the state representation in rl_strategy.py for consistency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlState {
    pub dist_bucket: usize,      // Distance to strike bucket index
    pub time_bucket: usize,      // Time remaining bucket index
    pub price_bucket: usize,     // Entry price bucket index
    pub direction: u8,           // 0 = below strike, 1 = above strike
    pub spread_bucket: u8,       // 0 = tight spread, 1 = wide spread
    pub momentum_bucket: u8,     // 0 = down, 1 = neutral, 2 = up
    pub time_of_day_bucket: u8,  // 0=night(0-6), 1=morning(6-12), 2=afternoon(12-18), 3=evening(18-24)
}

/// Experience tuple for RL training (state, action, reward, next_state).
/// Pushed to experience_queue in AppState, drained by experience_recorder.
#[derive(Debug, Clone)]
pub struct Experience {
    pub timestamp: DateTime<Utc>,
    pub ticker: String,
    pub state: RlState,
    pub action: u8,      // 0=HOLD, 1=BUY_YES, 2=BUY_NO
    pub reward: f64,     // PnL from settlement
    pub next_state: RlState,
}

/// Pending counterfactual for Hindsight Experience Replay (HER).
/// When we HOLD, we store the state and prices to generate counterfactual
/// experiences later when the market settles.
#[derive(Debug, Clone)]
pub struct PendingCounterfactual {
    pub ticker: String,
    pub state: RlState,
    pub yes_price: f64,  // Entry price if we had bought YES
    pub no_price: f64,   // Entry price if we had bought NO
    pub bet_size: f64,   // Budget for counterfactual calculations
    pub timestamp: DateTime<Utc>,
}

/// Metrics event for performance tracking.
/// Logged throughout the system for monitoring and analysis.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: String,  // "trade", "action", "settlement", etc.
    pub data: serde_json::Value,
}

/// Synthetic index state derived from Coinbase + Kraken price feeds.
#[derive(Debug, Clone, Serialize)]
pub struct IndexState {
    pub current_price:   f64,
    pub rolling_avg_60s: f64,
    pub ts:              DateTime<Utc>,
}

/// Coinbase Advanced Trade L2 order book state (for Oracle bot).
#[derive(Debug, Clone)]
pub struct CoinbaseL2State {
    pub mid_price: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub ts: DateTime<Utc>,
}

/// Active order state for Oracle bot order management.
#[derive(Debug, Clone)]
pub struct ActiveOrder {
    pub order_id: String,
    pub ticker: String,
    pub side: String,
    pub limit_price: f64,
    pub count: u64,
    pub placed_at: DateTime<Utc>,
    pub last_amend: DateTime<Utc>,
}

/// Tracking structure for Oracle orders that are not yet confirmed filled.
#[derive(Debug, Clone)]
pub struct OracleRestingOrder {
    pub order_id: String,
    pub ticker: String,
    pub side: String,
    pub count: u64,
    pub limit_price: f64,
    pub fair_value: f64,
    pub edge: f64,
    pub volatility: f64,
    pub entry_yes_bid: f64,
    pub entry_yes_ask: f64,
    pub entry_no_bid: f64,
    pub entry_no_ask: f64,
    pub entry_fee: f64,
    pub total_cost: f64,
    pub expiry_ts: Option<DateTime<Utc>>,
    pub strike: f64,
    pub placed_at: DateTime<Utc>,
}

/// Frozen summary statistics from the observation window (t=0..9).
/// Computed once when the market transitions from Observe → Trade.
#[derive(Debug, Clone)]
pub struct ObsSummary {
    pub mean:  f64,   // mean of all prices during observation
    pub high:  f64,   // max price during observation
    pub low:   f64,   // min price during observation
    pub range: f64,   // high - low
}

/// Per-market state for the observe / trade lifecycle.
/// One instance per active KXBTC15M ticker, stored in a local HashMap
/// inside trade_executor::run() (not in AppState — avoids lock contention
/// with the Kraken price feed).
#[derive(Debug, Clone)]
pub struct MarketTracker {
    /// Frozen observation summary. None until phase transitions to Trade.
    pub obs_summary: Option<ObsSummary>,
    /// Price at the moment obs_summary was computed (≈ t=10 price).
    pub obs_end_price: Option<f64>,
    /// Timestamp when obs_summary was computed (used for trade-phase cooldown).
    pub obs_end_time: Option<DateTime<Utc>>,
    /// Continuous price buffer: (timestamp, price) pairs pushed every loop tick.
    /// Used for obs_summary computation and short-momentum lookback.
    pub price_history: VecDeque<(DateTime<Utc>, f64)>,
}

impl MarketTracker {
    pub fn new() -> Self {
        Self {
            obs_summary:    None,
            obs_end_price:  None,
            obs_end_time:   None,
            price_history:  VecDeque::new(),
        }
    }
}

/// Central shared state. Passed as Arc<Mutex<AppState>> to every task.
#[derive(Debug)]
pub struct AppState {
    /// Latest Kalshi tick — written by kalshi_listener.
    pub kalshi: Option<MarketState>,

    /// Latest synthetic index state — written by coinbase_listener (Kraken).
    pub coinbase: Option<IndexState>,

    /// Latest price from Bitstamp WebSocket feed.
    pub binance_price: Option<f64>,

    /// Rolling 60-second window of prices for settlement average calculation.
    /// Each entry is (timestamp, price). Entries older than 60s are evicted.
    pub price_window: VecDeque<(DateTime<Utc>, f64)>,

    /// Accumulated PnL for the current session.
    pub session_pnl: f64,

    /// Pending Telegram messages. Any task may push a pre-formatted string;
    /// telegram_reporter drains this queue and delivers them.
    pub notifications: VecDeque<String>,

    /// Completed trades pending SQLite insertion.  Pushed by the trade
    /// executor, drained by data_recorder.
    pub pending_trades: VecDeque<TradeRecord>,

    /// Paper positions awaiting settlement (only used when PAPER_MODE is set).
    pub open_positions: Vec<OpenPosition>,

    /// Session win counter (paper mode).
    pub wins: u32,

    /// Session total-trade counter (paper mode).
    pub total_trades: u32,

    /// Most recent model inference output.  Written by trade_executor after
    /// each successful ONNX run; read by data_recorder for CSV logging.
    pub model_prob: Option<f64>,

    /// Cash available for new paper positions (debited on entry, credited on settle).
    pub available_capital: f64,

    /// Latest price from the Coinbase Exchange feed (None until first tick).
    pub coinbase_price: Option<f64>,

    /// Set to true by any task to signal graceful shutdown of all others.
    pub shutdown: bool,

    /// Experience queue for RL training. Pushed by trade_executor on settlement,
    /// drained by experience_recorder for persistence to SQLite.
    pub experience_queue: VecDeque<Experience>,

    /// Pending counterfactuals for Hindsight Experience Replay (HER).
    /// Stores HOLD states to generate counterfactual experiences on settlement.
    pub pending_counterfactuals: Vec<PendingCounterfactual>,

    /// Metrics queue for performance tracking. Pushed by various tasks,
    /// drained by metrics_recorder for monitoring and analysis.
    pub metrics_queue: VecDeque<MetricsEvent>,

    /// Current RL model version (loaded from model_registry/current/).
    pub model_version: Option<String>,

    // --- Data source health tracking ---
    /// Timestamp of last Kraken price update.
    pub kraken_last_update: Option<DateTime<Utc>>,
    /// Timestamp of last Coinbase price update.
    pub coinbase_last_update: Option<DateTime<Utc>>,
    /// Timestamp of last Bitstamp price update.
    pub bitstamp_last_update: Option<DateTime<Utc>>,

    // --- Oracle bot support ---
    /// Coinbase Advanced Trade L2 order book state (for Oracle bot).
    pub coinbase_l2: Option<CoinbaseL2State>,
    /// Last time we logged a debug message (used for throttling).
    pub last_debug_log: Option<DateTime<Utc>>,
    /// Active Oracle bot orders (ticker → order state).
    pub active_oracle_orders: HashMap<String, ActiveOrder>,
    /// Rolling volatility calculation window (Oracle bot).
    pub volatility_window: VecDeque<(DateTime<Utc>, f64)>,
    /// Stateful volatility trackers per asset (Oracle bot).
    pub multi_volatility_trackers: HashMap<String, EwmaVolatilityTracker>,
}

/// Maximum age (in seconds) before a data source is considered stale/down.
const SOURCE_STALE_SECS: i64 = 30;

impl AppState {
    pub fn new() -> Self {
        // Load initial capital from environment
        let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
            .unwrap_or_else(|_| "1000".into())
            .parse()
            .unwrap_or(1000.0);

        Self {
            kalshi:        None,
            coinbase:      None,
            binance_price: None,
            price_window:  VecDeque::new(),
            session_pnl:   0.0,
            notifications: VecDeque::new(),
            pending_trades: VecDeque::new(),
            open_positions: Vec::new(),
            wins:          0,
            total_trades:  0,
            model_prob:    None,
            available_capital: starting_capital,
            coinbase_price: None,
            shutdown:       false,
            experience_queue: VecDeque::new(),
            pending_counterfactuals: Vec::new(),
            metrics_queue: VecDeque::new(),
            model_version: None,
            kraken_last_update:   None,
            coinbase_last_update: None,
            bitstamp_last_update: None,
            coinbase_l2:              None,
            last_debug_log:           None,
            active_oracle_orders:     HashMap::new(),
            volatility_window:        VecDeque::new(),
            multi_volatility_trackers: HashMap::new(),
        }
    }

    /// Returns `true` if Kraken data is fresh (updated within the last 30s).
    pub fn is_kraken_alive(&self) -> bool {
        self.kraken_last_update
            .map_or(false, |ts| (Utc::now() - ts).num_seconds() < SOURCE_STALE_SECS)
    }

    /// Returns `true` if Coinbase data is fresh (updated within the last 30s).
    pub fn is_coinbase_alive(&self) -> bool {
        self.coinbase_last_update
            .map_or(false, |ts| (Utc::now() - ts).num_seconds() < SOURCE_STALE_SECS)
    }

    /// Returns `true` if Bitstamp data is fresh (updated within the last 30s).
    pub fn is_bitstamp_alive(&self) -> bool {
        self.bitstamp_last_update
            .map_or(false, |ts| (Utc::now() - ts).num_seconds() < SOURCE_STALE_SECS)
    }

    /// Number of currently alive data sources.
    pub fn alive_source_count(&self) -> u8 {
        let mut count = 0;
        if self.is_kraken_alive()   { count += 1; }
        if self.is_coinbase_alive() { count += 1; }
        if self.is_bitstamp_alive() { count += 1; }
        count
    }

    /// Compute a hybrid spot price from all non-stale data sources.
    /// Returns `None` only if ALL three sources are stale/missing.
    pub fn best_available_price(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0.0;

        // Source 1: Kraken (WebSocket)
        if self.is_kraken_alive() {
            if let Some(ref idx) = self.coinbase {
                if idx.current_price > 0.0 {
                    sum += idx.current_price;
                    count += 1.0;
                }
            }
        }

        // Source 2: Coinbase (REST)
        if self.is_coinbase_alive() {
            if let Some(p) = self.coinbase_price {
                if p > 0.0 {
                    sum += p;
                    count += 1.0;
                }
            }
        }

        // Source 3: Bitstamp (WebSocket)
        if self.is_bitstamp_alive() {
            if let Some(p) = self.binance_price {
                if p > 0.0 {
                    sum += p;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            Some(sum / count)
        } else {
            None
        }
    }

    /// Build a synthetic `IndexState` from the best available price.
    /// Falls back to constructing one from Coinbase or Bitstamp if Kraken is down.
    pub fn best_available_index(&self) -> Option<IndexState> {
        // If Kraken is alive and has a full IndexState, prefer it
        if self.is_kraken_alive() {
            if let Some(ref idx) = self.coinbase {
                return Some(idx.clone());
            }
        }

        // Otherwise, construct a synthetic IndexState from whatever price we have
        if let Some(price) = self.best_available_price() {
            Some(IndexState {
                current_price:   price,
                rolling_avg_60s: price, // approximate — no rolling window without Kraken
                ts:              Utc::now(),
            })
        } else {
            None
        }
    }
}
