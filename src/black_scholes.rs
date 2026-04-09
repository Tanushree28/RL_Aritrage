//! Binary Black-Scholes fair value calculator for Kalshi crypto futures.
//!
//! Computes the probability that spot > strike at expiry using the
//! risk-neutral formula for a digital (binary) call option:
//!
//!   P_fair = Φ( (ln(S/K) - (σ²/2)T) / (σ√T) )
//!
//! Also provides rolling volatility estimation from tick data.

use chrono::{DateTime, Utc};
use std::collections::VecDeque;

// ============================================================
// BINARY BLACK-SCHOLES
// ============================================================

/// Minimum annualized volatility (25% - realistic floor for BTC/crypto)
/// Prevents model overconfidence on brief calm periods in ultra-short time horizons
pub const MIN_VOLATILITY: f64 = 0.40;

/// Fallback annualized volatility when insufficient data (35% - conservative for crypto)
pub const FALLBACK_VOLATILITY: f64 = 0.35;

/// Seconds in a year (for annualization).
pub const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 3600.0;

/// Computes the Binary Black-Scholes fair value (probability that spot > strike).
///
/// # Arguments
/// * `spot` - Current spot price (e.g., BTC price in USD)
/// * `strike` - Contract strike price (e.g., 96000.0)
/// * `time_to_expiry_secs` - Seconds remaining until contract expiry
/// * `volatility` - Annualized volatility (e.g., 0.50 = 50%)
///
/// # Returns
/// Probability in [0.0, 1.0] that spot > strike at expiry.
pub fn binary_black_scholes(spot: f64, strike: f64, time_to_expiry_secs: f64, volatility: f64) -> f64 {
    // Edge case: expired or at expiry
    if time_to_expiry_secs <= 0.0 {
        return if spot > strike { 1.0 } else { 0.0 };
    }

    // Edge case: invalid prices
    if spot <= 0.0 || strike <= 0.0 {
        return 0.0;
    }

    // Clamp volatility to minimum
    let sigma = volatility.max(MIN_VOLATILITY);

    // Convert seconds to years
    let t = time_to_expiry_secs / SECONDS_PER_YEAR;

    let d2 = calculate_d2(spot, strike, t, sigma);

    // P_fair = Φ(d2)
    let p_fair = fast_normal_cdf(d2);

    // Clamp to [0.0, 1.0] for safety and handle extreme tails
    if d2 > 6.0 { 1.0 }
    else if d2 < -6.0 { 0.0 }
    else { p_fair.clamp(0.0, 1.0) }
}

/// d2 = (ln(S/K) - (σ²/2)T) / (σ√T)
pub fn calculate_d2(spot: f64, strike: f64, t_years: f64, sigma_annual: f64) -> f64 {
    let sqrt_t = t_years.sqrt();
    if sqrt_t < 1e-9 {
        if spot > strike { 10.0 } else { -10.0 }
    } else {
        ((spot / strike).ln() - (sigma_annual * sigma_annual / 2.0) * t_years) / (sigma_annual * sqrt_t)
    }
}

/// Extremely fast polynomial approximation of the standard normal CDF.
/// Max absolute error: 7.5e-8, which is well within the tolerance needed 
/// for options priced in cents.
pub fn fast_normal_cdf(x: f64) -> f64 {
    let sign = x < 0.0;
    let x_abs = x.abs();

    // Abramowitz and Stegun constants
    const P: f64 = 0.2316419;
    const A1: f64 = 0.319381530;
    const A2: f64 = -0.356563782;
    const A3: f64 = 1.781477937;
    const A4: f64 = -1.821255978;
    const A5: f64 = 1.330274429;
    
    // 1.0 / sqrt(2.0 * PI)
    const INV_SQRT_2PI: f64 = 0.3989422804014327;

    let t = 1.0 / (1.0 + P * x_abs);
    
    // Standard normal PDF
    let pdf = INV_SQRT_2PI * (-0.5 * x_abs * x_abs).exp();

    // Horner's method for polynomial evaluation
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    let cdf_abs = 1.0 - pdf * poly;

    if sign {
        1.0 - cdf_abs
    } else {
        cdf_abs
    }
}

#[derive(Debug, Clone)]
pub struct EwmaVolatilityTracker {
    lambda: f64,
    variance: f64,
    last_price: Option<f64>,
    last_timestamp: Option<DateTime<Utc>>,
    is_initialized: bool,
}

impl EwmaVolatilityTracker {
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            variance: 0.0,
            last_price: None,
            last_timestamp: None,
            is_initialized: false,
        }
    }

    /// Feeds a new price into the tracker and returns the current annualized volatility in O(1) time.
    pub fn update(&mut self, price: f64, timestamp: DateTime<Utc>) -> f64 {
        if let (Some(prev_price), Some(prev_ts)) = (self.last_price, self.last_timestamp) {
            if price > 0.0 && prev_price > 0.0 {
                let dt = (timestamp - prev_ts).num_milliseconds() as f64 / 1000.0;
                
                // Only update if time has actually moved forward to prevent division by zero
                if dt > 0.0 {
                    let log_return = (price / prev_price).ln();
                    
                    if self.is_initialized {
                        // O(1) EWMA variance update
                        self.variance = self.lambda * self.variance + (1.0 - self.lambda) * (log_return * log_return);
                    } else {
                        // First valid return initializes the variance
                        self.variance = log_return * log_return;
                        self.is_initialized = true;
                    }

                    self.last_price = Some(price);
                    self.last_timestamp = Some(timestamp);

                    // Annualize based on the time elapsed
                    let annualized_vol = (self.variance * (SECONDS_PER_YEAR / dt)).sqrt();
                    return annualized_vol.clamp(MIN_VOLATILITY, 5.0);
                }
            }
        } else {
            // First tick seen
            self.last_price = Some(price);
            self.last_timestamp = Some(timestamp);
        }

        // Return fallback while we wait for a second valid tick to compute a return
        if self.is_initialized && self.last_timestamp.is_some() {
            // If we have an old variance but a zero dt on this specific tick, just scale the old one
            // Assuming ~1 second standard tick interval as a rough fallback if data is jammed
            let annualized_vol = (self.variance * SECONDS_PER_YEAR).sqrt();
            annualized_vol.clamp(MIN_VOLATILITY, 5.0)
        } else {
            FALLBACK_VOLATILITY
        }
    }

    pub fn current_volatility(&self) -> f64 {
        if !self.is_initialized {
            return FALLBACK_VOLATILITY;
        }
        // Annualize assuming ~1s interval if we need a quick peek without an update
        (self.variance * SECONDS_PER_YEAR).sqrt().clamp(MIN_VOLATILITY, 5.0)
    }
}

/// Calculates annualized volatility from a rolling window of (timestamp, price) pairs.
///
/// Uses log returns over adjacent ticks, then annualizes based on the
/// average time interval between ticks.
///
/// # Arguments
/// * `prices` - VecDeque of (timestamp, price) pairs, most recent last.
///
/// # Returns
/// Annualized volatility. Returns `FALLBACK_VOLATILITY` if insufficient data.
pub fn calculate_rolling_volatility(prices: &VecDeque<(DateTime<Utc>, f64)>) -> f64 {
    if prices.len() < 10 {
        return FALLBACK_VOLATILITY;
    }

    let mut log_returns: Vec<f64> = Vec::with_capacity(prices.len() - 1);
    
    // Total time span of the window in seconds
    let t_first = prices.front().unwrap().0;
    let t_last = prices.back().unwrap().0;
    let total_dt = (t_last - t_first).num_milliseconds() as f64 / 1000.0;

    if total_dt <= 1.0 { // Need at least 1 second of data
        return FALLBACK_VOLATILITY;
    }

    for i in 1..prices.len() {
        let (_, p0) = &prices[i - 1];
        let (_, p1) = &prices[i];
        if *p0 <= 0.0 || *p1 <= 0.0 { continue; }
        log_returns.push((p1 / p0).ln());
    }

    if log_returns.len() < 5 {
        return FALLBACK_VOLATILITY;
    }

    // Variance of log returns
    let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
    let variance = log_returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / (log_returns.len() as f64); // Population variance for short window

    // Annualize based on total time span:
    // σ_annual = √(Variance * (TicksPerYear))
    // TicksPerYear ≈ log_returns.len() * (SecondsPerYear / total_dt)
    let annualized_vol = (variance * (log_returns.len() as f64 * SECONDS_PER_YEAR / total_dt)).sqrt();

    // Clamp to reasonable range [1%, 500%]
    annualized_vol.clamp(MIN_VOLATILITY, 5.0)
}

/// Calculates EWMA (Exponentially Weighted Moving Average) volatility.
pub fn calculate_ewma_volatility(prices: &VecDeque<(DateTime<Utc>, f64)>, lambda: f64) -> f64 {
    if prices.len() < 10 {
        return FALLBACK_VOLATILITY;
    }

    let mut variance = 0.0;
    let mut weight_sum = 0.0;
    
    let t_first = prices.front().unwrap().0;
    let t_last = prices.back().unwrap().0;
    let total_dt = (t_last - t_first).num_milliseconds() as f64 / 1000.0;

    if total_dt <= 1.0 { return FALLBACK_VOLATILITY; }

    for i in 1..prices.len() {
        let (_, p0) = &prices[i - 1];
        let (_, p1) = &prices[i];
        if *p0 <= 0.0 || *p1 <= 0.0 { continue; }
        
        let lr = (p1 / p0).ln();
        let weight = lambda.powi((prices.len() - 1 - i) as i32);
        variance += weight * lr * lr;
        weight_sum += weight;
    }

    if weight_sum <= 0.0 { return FALLBACK_VOLATILITY; }
    
    let weighted_variance = variance / weight_sum;
    let annualized_vol = (weighted_variance * ((prices.len() - 1) as f64 * SECONDS_PER_YEAR / total_dt)).sqrt();

    annualized_vol.clamp(MIN_VOLATILITY, 5.0)
}

// ============================================================
// UNIT TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    #[test]
    fn test_atm_option() {
        // ATM option should be close to 0.50 (slightly below due to drift term)
        let p = binary_black_scholes(100.0, 100.0, 900.0, 0.50);
        assert!(p > 0.45 && p < 0.55, "ATM P_fair={p}, expected ~0.50");
    }

    #[test]
    fn test_deep_itm() {
        // Deep in-the-money: spot >> strike
        let p = binary_black_scholes(100.0, 50.0, 900.0, 0.20);
        assert!(p > 0.95, "Deep ITM P_fair={p}, expected >0.95");
    }

    #[test]
    fn test_deep_otm() {
        // Deep out-of-the-money: spot << strike
        let p = binary_black_scholes(50.0, 100.0, 900.0, 0.20);
        assert!(p < 0.05, "Deep OTM P_fair={p}, expected <0.05");
    }

    #[test]
    fn test_expired_itm() {
        // At expiry, ITM should return 1.0
        let p = binary_black_scholes(100.0, 99.0, 0.0, 0.50);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_expired_otm() {
        // At expiry, OTM should return 0.0
        let p = binary_black_scholes(99.0, 100.0, 0.0, 0.50);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_negative_time() {
        // Negative time should behave like expired
        let p = binary_black_scholes(100.0, 99.0, -10.0, 0.50);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_zero_volatility_clamped() {
        // σ = 0 should be clamped to MIN_VOLATILITY, not panic
        let p = binary_black_scholes(100.0, 99.0, 900.0, 0.0);
        assert!(p >= 0.0 && p <= 1.0, "Zero vol P_fair={p}, should be valid");
    }

    #[test]
    fn test_output_range() {
        // Test many parameter combos for output in [0, 1]
        for spot in [50.0, 100.0, 200.0] {
            for strike in [50.0, 100.0, 200.0] {
                for t in [60.0, 300.0, 900.0] {
                    for vol in [0.01, 0.20, 1.0, 3.0] {
                        let p = binary_black_scholes(spot, strike, t, vol);
                        assert!(
                            p >= 0.0 && p <= 1.0,
                            "Out of range: spot={spot}, strike={strike}, t={t}, vol={vol}, p={p}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_rolling_volatility_insufficient_data() {
        let prices: VecDeque<(DateTime<Utc>, f64)> = VecDeque::new();
        let vol = calculate_rolling_volatility(&prices);
        assert_eq!(vol, FALLBACK_VOLATILITY);
    }

    #[test]
    fn test_rolling_volatility_with_data() {
        // Create 60 ticks at 1-second intervals with some price movement
        let mut prices = VecDeque::new();
        let base = Utc::now();
        let base_price = 100_000.0;

        for i in 0..60 {
            let ts = base + TimeDelta::seconds(i);
            // Slight random-ish movement using a deterministic pattern
            let price = base_price + (i as f64 * 0.1).sin() * 50.0;
            prices.push_back((ts, price));
        }

        let vol = calculate_rolling_volatility(&prices);
        assert!(vol >= MIN_VOLATILITY, "Vol should be at least MIN_VOLATILITY: {vol}");
        assert!(vol <= 5.0, "Vol should be capped at 5.0: {vol}");
    }

    #[test]
    fn test_known_bs_value() {
        // With MIN_VOLATILITY=0.40 clamp, input σ=0.20 → actual σ=0.40
        // d2 = (ln(1) - (0.40²/2) * 0.25) / (0.40 * 0.5) = -0.02 / 0.20 = -0.10
        // Φ(-0.10) ≈ 0.4602
        let t_secs = 0.25 * SECONDS_PER_YEAR;
        let p = binary_black_scholes(100.0, 100.0, t_secs, 0.20);
        assert!(
            (p - 0.4602).abs() < 0.01,
            "Known BS value (clamped vol): expected ~0.4602, got {p}"
        );
    }
}
