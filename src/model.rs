use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Asset-specific configuration for trading different cryptocurrencies
#[derive(Debug, Clone)]
pub struct AssetConfig {
    pub name: String,
    pub kalshi_series: String,
    pub kraken_symbol: String,
    pub coinbase_pair: String,
    pub volatility_baseline: f64,
    pub api_base_url: String, // "https://api.elections.kalshi.com" or "https://demo-api.kalshi.co"
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
    pub strike:      f64,
    pub side:        String,   // "YES" or "NO"
    pub entry_price: f64,
    pub exit_price:  f64,
    pub pnl:         f64,
    pub roi:         f64,
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

    /// Latest synthetic index state — written by coinbase_listener.
    pub coinbase: Option<IndexState>,

    /// Latest price from Binance.US (or Global) WebSocket feed.
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
}

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
        }
    }
}
