//! Realistic fill models for backtesting.
//!
//! This module provides various fill models with increasing realism for
//! more accurate backtesting simulations.
//!
//! # Overview
//!
//! Fill models determine how and when orders get filled during backtesting:
//!
//! - **ImmediateFillModel**: Fills at quote price when market crosses (baseline)
//! - **QueuePositionFillModel**: Simulates queue priority based on time and size
//! - **ProbabilisticFillModel**: Fill probability based on depth and time
//! - **MarketImpactFillModel**: Price impact proportional to order size
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::backtest::{
//!     ImmediateFillModel, FillModel, SimulatedOrder, MarketTick, FillResult
//! };
//! use market_maker_rs::execution::Side;
//! use market_maker_rs::dec;
//!
//! let model = ImmediateFillModel::new();
//! let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
//! let tick = MarketTick::new(1001, dec!(99.9), dec!(1.0), dec!(100.0), dec!(1.0));
//!
//! // Check if order fills
//! let result = model.simulate_fill(&order, &tick, 1);
//! match result {
//!     FillResult::FullFill { fill_price } => println!("Filled at {}", fill_price),
//!     FillResult::PartialFill { filled_quantity, fill_price } => {
//!         println!("Partial fill: {} at {}", filled_quantity, fill_price)
//!     }
//!     FillResult::NoFill => println!("No fill"),
//! }
//! ```

use crate::Decimal;
use crate::execution::Side;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::data::MarketTick;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Simulated order for fill model evaluation.
///
/// Represents an order that the fill model will evaluate for potential fills.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::SimulatedOrder;
/// use market_maker_rs::execution::Side;
/// use market_maker_rs::dec;
///
/// let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
/// assert_eq!(order.side, Side::Buy);
/// assert_eq!(order.price, dec!(100.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulatedOrder {
    /// Order side (Buy or Sell).
    pub side: Side,
    /// Limit price.
    pub price: Decimal,
    /// Order quantity.
    pub quantity: Decimal,
    /// Timestamp when order was submitted, in milliseconds.
    pub submitted_at: u64,
}

impl SimulatedOrder {
    /// Creates a new simulated order.
    #[must_use]
    pub fn new(side: Side, price: Decimal, quantity: Decimal, submitted_at: u64) -> Self {
        Self {
            side,
            price,
            quantity,
            submitted_at,
        }
    }

    /// Returns the notional value of the order.
    #[must_use]
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }

    /// Returns true if this is a buy order.
    #[must_use]
    pub fn is_buy(&self) -> bool {
        self.side == Side::Buy
    }

    /// Returns true if this is a sell order.
    #[must_use]
    pub fn is_sell(&self) -> bool {
        self.side == Side::Sell
    }
}

/// Result of a fill simulation.
///
/// Represents the outcome of evaluating an order against market conditions.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::FillResult;
/// use market_maker_rs::dec;
///
/// let result = FillResult::FullFill { fill_price: dec!(100.0) };
/// assert!(result.is_filled());
/// assert_eq!(result.fill_price(), Some(dec!(100.0)));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum FillResult {
    /// No fill occurred.
    #[default]
    NoFill,
    /// Partial fill with specified quantity and price.
    PartialFill {
        /// Quantity that was filled.
        filled_quantity: Decimal,
        /// Price at which the fill occurred.
        fill_price: Decimal,
    },
    /// Complete fill at specified price.
    FullFill {
        /// Price at which the fill occurred.
        fill_price: Decimal,
    },
}

impl FillResult {
    /// Returns true if any fill occurred.
    #[must_use]
    pub fn is_filled(&self) -> bool {
        !matches!(self, FillResult::NoFill)
    }

    /// Returns true if the order was completely filled.
    #[must_use]
    pub fn is_full_fill(&self) -> bool {
        matches!(self, FillResult::FullFill { .. })
    }

    /// Returns true if the order was partially filled.
    #[must_use]
    pub fn is_partial_fill(&self) -> bool {
        matches!(self, FillResult::PartialFill { .. })
    }

    /// Returns the fill price if any fill occurred.
    #[must_use]
    pub fn fill_price(&self) -> Option<Decimal> {
        match self {
            FillResult::NoFill => None,
            FillResult::PartialFill { fill_price, .. } => Some(*fill_price),
            FillResult::FullFill { fill_price } => Some(*fill_price),
        }
    }

    /// Returns the filled quantity.
    #[must_use]
    pub fn filled_quantity(&self, order_quantity: Decimal) -> Decimal {
        match self {
            FillResult::NoFill => Decimal::ZERO,
            FillResult::PartialFill {
                filled_quantity, ..
            } => *filled_quantity,
            FillResult::FullFill { .. } => order_quantity,
        }
    }
}

/// Trait for fill model implementations.
///
/// Fill models determine how and when orders get filled during backtesting.
/// Different models provide varying levels of realism.
pub trait FillModel: Send + Sync {
    /// Simulates whether and how an order fills given market state.
    ///
    /// # Arguments
    ///
    /// * `order` - The order to evaluate
    /// * `tick` - Current market tick data
    /// * `time_in_queue_ms` - Time the order has been in queue, in milliseconds
    ///
    /// # Returns
    ///
    /// A `FillResult` indicating no fill, partial fill, or full fill.
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> FillResult;

    /// Resets the model state (e.g., queue positions).
    fn reset(&mut self);

    /// Returns the model name for identification.
    fn name(&self) -> &'static str;
}

/// Immediate fill model - the simplest fill model.
///
/// Fills orders immediately at the quote price when the market crosses.
/// This is the baseline model that tends to overestimate performance.
///
/// # Fill Logic
///
/// - Buy orders fill when market ask <= order price
/// - Sell orders fill when market bid >= order price
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{ImmediateFillModel, FillModel, SimulatedOrder, MarketTick};
/// use market_maker_rs::execution::Side;
/// use market_maker_rs::dec;
///
/// let model = ImmediateFillModel::new();
/// let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
///
/// // Market ask at 99.9 <= order price 100.0, should fill
/// let tick = MarketTick::new(1001, dec!(99.8), dec!(1.0), dec!(99.9), dec!(1.0));
/// let result = model.simulate_fill(&order, &tick, 1);
/// assert!(result.is_full_fill());
/// ```
#[derive(Debug, Clone, Default)]
pub struct ImmediateFillModel;

impl ImmediateFillModel {
    /// Creates a new immediate fill model.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl FillModel for ImmediateFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        _time_in_queue_ms: u64,
    ) -> FillResult {
        match order.side {
            Side::Buy => {
                // Buy fills when market ask <= order price
                if tick.ask_price <= order.price {
                    FillResult::FullFill {
                        fill_price: order.price,
                    }
                } else {
                    FillResult::NoFill
                }
            }
            Side::Sell => {
                // Sell fills when market bid >= order price
                if tick.bid_price >= order.price {
                    FillResult::FullFill {
                        fill_price: order.price,
                    }
                } else {
                    FillResult::NoFill
                }
            }
        }
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "ImmediateFill"
    }
}

/// Queue position fill model.
///
/// Simulates queue priority based on time in queue and estimated queue depth.
/// More realistic than immediate fill as it accounts for orders ahead in the queue.
///
/// # Model Assumptions
///
/// - Orders join the back of the queue at their price level
/// - Queue clears at a configurable fill rate (volume per millisecond)
/// - Order fills when enough volume has cleared ahead of it
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{QueuePositionFillModel, FillModel, SimulatedOrder, MarketTick};
/// use market_maker_rs::execution::Side;
/// use market_maker_rs::dec;
///
/// let mut model = QueuePositionFillModel::new(dec!(0.1)); // 0.1 units per ms
/// let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
/// let tick = MarketTick::new(1001, dec!(99.9), dec!(10.0), dec!(100.1), dec!(10.0));
///
/// // Update queue from market data
/// model.update_queue(&tick);
///
/// // Simulate fill after some time in queue
/// let result = model.simulate_fill(&order, &tick, 100);
/// ```
#[derive(Debug, Clone)]
pub struct QueuePositionFillModel {
    /// Estimated queue depth at each price level.
    queue_depth: HashMap<String, Decimal>,
    /// Fill rate: volume per millisecond that clears the queue.
    fill_rate: Decimal,
    /// Minimum time in queue before any fill is possible, in milliseconds.
    min_queue_time_ms: u64,
}

impl QueuePositionFillModel {
    /// Creates a new queue position fill model.
    ///
    /// # Arguments
    ///
    /// * `fill_rate` - Volume per millisecond that clears the queue
    #[must_use]
    pub fn new(fill_rate: Decimal) -> Self {
        Self {
            queue_depth: HashMap::new(),
            fill_rate,
            min_queue_time_ms: 0,
        }
    }

    /// Creates a new queue position fill model with minimum queue time.
    ///
    /// # Arguments
    ///
    /// * `fill_rate` - Volume per millisecond that clears the queue
    /// * `min_queue_time_ms` - Minimum time before fills are possible
    #[must_use]
    pub fn with_min_queue_time(fill_rate: Decimal, min_queue_time_ms: u64) -> Self {
        Self {
            queue_depth: HashMap::new(),
            fill_rate,
            min_queue_time_ms,
        }
    }

    /// Updates queue depth estimates from market data.
    ///
    /// Uses the bid/ask sizes as estimates of queue depth at those levels.
    pub fn update_queue(&mut self, tick: &MarketTick) {
        // Use string keys to avoid Decimal hash issues
        let bid_key = tick.bid_price.to_string();
        let ask_key = tick.ask_price.to_string();

        self.queue_depth.insert(bid_key, tick.bid_size);
        self.queue_depth.insert(ask_key, tick.ask_size);
    }

    /// Returns the estimated queue depth at a price level.
    #[must_use]
    pub fn get_queue_depth(&self, price: Decimal) -> Decimal {
        self.queue_depth
            .get(&price.to_string())
            .copied()
            .unwrap_or(Decimal::ZERO)
    }

    /// Calculates volume cleared based on time in queue.
    fn volume_cleared(&self, time_in_queue_ms: u64) -> Decimal {
        self.fill_rate * Decimal::from(time_in_queue_ms)
    }
}

impl FillModel for QueuePositionFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> FillResult {
        // Check minimum queue time
        if time_in_queue_ms < self.min_queue_time_ms {
            return FillResult::NoFill;
        }

        // First check if market has crossed our price
        let market_crossed = match order.side {
            Side::Buy => tick.ask_price <= order.price,
            Side::Sell => tick.bid_price >= order.price,
        };

        if !market_crossed {
            return FillResult::NoFill;
        }

        // Get queue depth at our price level
        let queue_ahead = self.get_queue_depth(order.price);

        // Calculate how much volume has cleared
        let cleared = self.volume_cleared(time_in_queue_ms);

        if cleared >= queue_ahead + order.quantity {
            // Full fill - we've cleared the queue plus our order
            FillResult::FullFill {
                fill_price: order.price,
            }
        } else if cleared > queue_ahead {
            // Partial fill - some of our order filled
            let filled = cleared - queue_ahead;
            if filled > Decimal::ZERO {
                FillResult::PartialFill {
                    filled_quantity: filled.min(order.quantity),
                    fill_price: order.price,
                }
            } else {
                FillResult::NoFill
            }
        } else {
            // Still waiting in queue
            FillResult::NoFill
        }
    }

    fn reset(&mut self) {
        self.queue_depth.clear();
    }

    fn name(&self) -> &'static str {
        "QueuePosition"
    }
}

impl Default for QueuePositionFillModel {
    fn default() -> Self {
        Self::new(Decimal::from_str_exact("0.01").unwrap())
    }
}

/// Probabilistic fill model.
///
/// Fills orders based on a probability that depends on market depth and time in queue.
/// Uses a seeded RNG for reproducible backtests.
///
/// # Probability Formula
///
/// ```text
/// P(fill) = base_prob * exp(-depth_factor * relative_depth) * (1 - exp(-time_factor * time_ms))
/// ```
///
/// Where:
/// - `base_prob` is the base fill probability
/// - `depth_factor` controls sensitivity to market depth
/// - `relative_depth` is order size relative to available liquidity
/// - `time_factor` controls how probability increases over time
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{ProbabilisticFillModel, FillModel, SimulatedOrder, MarketTick};
/// use market_maker_rs::execution::Side;
/// use market_maker_rs::dec;
///
/// let model = ProbabilisticFillModel::new(
///     dec!(0.5),  // 50% base probability
///     dec!(1.0),  // depth factor
///     dec!(0.001), // time factor
///     42,         // seed for reproducibility
/// );
///
/// let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
/// let tick = MarketTick::new(1001, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));
///
/// // Probability depends on depth and time
/// let prob = model.calculate_probability(&order, &tick, 100);
/// ```
#[derive(Debug)]
pub struct ProbabilisticFillModel {
    /// Base fill probability per tick (0.0 to 1.0).
    base_probability: Decimal,
    /// Depth factor: higher = less likely to fill in deep book.
    depth_factor: Decimal,
    /// Time factor: higher = more likely to fill over time.
    time_factor: Decimal,
    /// Seed for random number generation.
    seed: u64,
    /// Current state for deterministic random generation (thread-safe).
    state: AtomicU64,
}

impl ProbabilisticFillModel {
    /// Creates a new probabilistic fill model.
    ///
    /// # Arguments
    ///
    /// * `base_probability` - Base fill probability (0.0 to 1.0)
    /// * `depth_factor` - Sensitivity to market depth
    /// * `time_factor` - How probability increases over time
    /// * `seed` - Random seed for reproducibility
    #[must_use]
    pub fn new(
        base_probability: Decimal,
        depth_factor: Decimal,
        time_factor: Decimal,
        seed: u64,
    ) -> Self {
        Self {
            base_probability,
            depth_factor,
            time_factor,
            seed,
            state: AtomicU64::new(seed),
        }
    }

    /// Calculates fill probability for an order.
    ///
    /// # Arguments
    ///
    /// * `order` - The order to evaluate
    /// * `tick` - Current market tick
    /// * `time_in_queue_ms` - Time in queue in milliseconds
    #[must_use]
    pub fn calculate_probability(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> Decimal {
        // Get available liquidity at our side
        let available_liquidity = match order.side {
            Side::Buy => tick.ask_size,
            Side::Sell => tick.bid_size,
        };

        // Calculate relative depth (order size / available liquidity)
        let relative_depth = if available_liquidity > Decimal::ZERO {
            order.quantity / available_liquidity
        } else {
            Decimal::ONE // Max depth if no liquidity
        };

        // Depth component: exp(-depth_factor * relative_depth)
        // Approximated as 1 / (1 + depth_factor * relative_depth) for simplicity
        let depth_component = Decimal::ONE / (Decimal::ONE + self.depth_factor * relative_depth);

        // Time component: 1 - exp(-time_factor * time_ms)
        // Approximated as min(1, time_factor * time_ms) for simplicity
        let time_component = (self.time_factor * Decimal::from(time_in_queue_ms)).min(Decimal::ONE);

        // Final probability
        self.base_probability * depth_component * time_component
    }

    /// Generates a deterministic pseudo-random number between 0 and 1.
    fn next_random(&self) -> Decimal {
        // Simple LCG for deterministic randomness
        let current = self.state.load(Ordering::Relaxed);
        let next = current.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state.store(next, Ordering::Relaxed);
        let value = (next >> 33) as u32;
        Decimal::from(value) / Decimal::from(u32::MAX)
    }
}

impl FillModel for ProbabilisticFillModel {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> FillResult {
        // First check if market has crossed our price
        let market_crossed = match order.side {
            Side::Buy => tick.ask_price <= order.price,
            Side::Sell => tick.bid_price >= order.price,
        };

        if !market_crossed {
            return FillResult::NoFill;
        }

        // Calculate fill probability
        let prob = self.calculate_probability(order, tick, time_in_queue_ms);

        // Generate random number and compare
        let random = self.next_random();

        if random < prob {
            FillResult::FullFill {
                fill_price: order.price,
            }
        } else {
            FillResult::NoFill
        }
    }

    fn reset(&mut self) {
        self.state.store(self.seed, Ordering::Relaxed);
    }

    fn name(&self) -> &'static str {
        "Probabilistic"
    }
}

impl Default for ProbabilisticFillModel {
    fn default() -> Self {
        Self::new(
            Decimal::from_str_exact("0.5").unwrap(),
            Decimal::ONE,
            Decimal::from_str_exact("0.001").unwrap(),
            42,
        )
    }
}

/// Market impact fill model.
///
/// Applies price impact proportional to order size using the square-root model.
/// Wraps another fill model and adjusts the fill price based on market impact.
///
/// # Impact Formula
///
/// ```text
/// impact = coefficient * sqrt(size / average_daily_volume)
/// ```
///
/// For buy orders, the fill price is increased by the impact.
/// For sell orders, the fill price is decreased by the impact.
///
/// # Example
///
/// ```rust
/// use market_maker_rs::backtest::{
///     MarketImpactFillModel, ImmediateFillModel, FillModel, SimulatedOrder, MarketTick
/// };
/// use market_maker_rs::execution::Side;
/// use market_maker_rs::dec;
///
/// let base_model = ImmediateFillModel::new();
/// let model = MarketImpactFillModel::new(
///     dec!(0.1),      // impact coefficient
///     dec!(1000000.0), // average daily volume
///     base_model,
/// );
///
/// let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(100.0), 1000);
/// let tick = MarketTick::new(1001, dec!(99.9), dec!(1000.0), dec!(100.0), dec!(1000.0));
///
/// // Calculate expected impact
/// let impact = model.calculate_impact(dec!(100.0));
/// ```
#[derive(Debug, Clone)]
pub struct MarketImpactFillModel<M: FillModel> {
    /// Impact coefficient for square-root model.
    impact_coefficient: Decimal,
    /// Average daily volume for normalization.
    average_daily_volume: Decimal,
    /// Underlying fill model.
    base_model: M,
}

impl<M: FillModel> MarketImpactFillModel<M> {
    /// Creates a new market impact fill model.
    ///
    /// # Arguments
    ///
    /// * `impact_coefficient` - Coefficient for impact calculation
    /// * `average_daily_volume` - ADV for normalization
    /// * `base_model` - Underlying fill model to wrap
    #[must_use]
    pub fn new(impact_coefficient: Decimal, average_daily_volume: Decimal, base_model: M) -> Self {
        Self {
            impact_coefficient,
            average_daily_volume,
            base_model,
        }
    }

    /// Calculates price impact for a given order size.
    ///
    /// Uses the square-root model: `impact = coeff * sqrt(size / ADV)`
    #[must_use]
    pub fn calculate_impact(&self, size: Decimal) -> Decimal {
        if self.average_daily_volume <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let size_ratio = size / self.average_daily_volume;

        // Approximate sqrt using Newton's method
        let sqrt_ratio = decimal_sqrt(size_ratio).unwrap_or(Decimal::ZERO);

        self.impact_coefficient * sqrt_ratio
    }

    /// Returns a reference to the base model.
    #[must_use]
    pub fn base_model(&self) -> &M {
        &self.base_model
    }

    /// Returns a mutable reference to the base model.
    pub fn base_model_mut(&mut self) -> &mut M {
        &mut self.base_model
    }
}

impl<M: FillModel> FillModel for MarketImpactFillModel<M> {
    fn simulate_fill(
        &self,
        order: &SimulatedOrder,
        tick: &MarketTick,
        time_in_queue_ms: u64,
    ) -> FillResult {
        // First get result from base model
        let base_result = self.base_model.simulate_fill(order, tick, time_in_queue_ms);

        // Apply market impact to fill price
        match base_result {
            FillResult::NoFill => FillResult::NoFill,
            FillResult::PartialFill {
                filled_quantity,
                fill_price,
            } => {
                let impact = self.calculate_impact(filled_quantity);
                let adjusted_price = match order.side {
                    Side::Buy => fill_price + impact,
                    Side::Sell => fill_price - impact,
                };
                FillResult::PartialFill {
                    filled_quantity,
                    fill_price: adjusted_price,
                }
            }
            FillResult::FullFill { fill_price } => {
                let impact = self.calculate_impact(order.quantity);
                let adjusted_price = match order.side {
                    Side::Buy => fill_price + impact,
                    Side::Sell => fill_price - impact,
                };
                FillResult::FullFill {
                    fill_price: adjusted_price,
                }
            }
        }
    }

    fn reset(&mut self) {
        self.base_model.reset();
    }

    fn name(&self) -> &'static str {
        "MarketImpact"
    }
}

/// Approximate square root using Newton's method.
fn decimal_sqrt(n: Decimal) -> Option<Decimal> {
    if n < Decimal::ZERO {
        return None;
    }
    if n == Decimal::ZERO {
        return Some(Decimal::ZERO);
    }

    let mut x = n;
    let two = Decimal::TWO;

    for _ in 0..20 {
        let next = (x + n / x) / two;
        if (next - x).abs() < Decimal::from_str_exact("0.0000001").unwrap() {
            return Some(next);
        }
        x = next;
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec;

    fn create_test_tick(bid: Decimal, ask: Decimal) -> MarketTick {
        MarketTick::new(1000, bid, dec!(10.0), ask, dec!(10.0))
    }

    // SimulatedOrder tests
    #[test]
    fn test_simulated_order_new() {
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);

        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.price, dec!(100.0));
        assert_eq!(order.quantity, dec!(1.0));
        assert_eq!(order.submitted_at, 1000);
    }

    #[test]
    fn test_simulated_order_notional() {
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(2.5), 1000);
        assert_eq!(order.notional(), dec!(250.0));
    }

    #[test]
    fn test_simulated_order_is_buy_sell() {
        let buy = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let sell = SimulatedOrder::new(Side::Sell, dec!(100.0), dec!(1.0), 1000);

        assert!(buy.is_buy());
        assert!(!buy.is_sell());
        assert!(!sell.is_buy());
        assert!(sell.is_sell());
    }

    // FillResult tests
    #[test]
    fn test_fill_result_no_fill() {
        let result = FillResult::NoFill;

        assert!(!result.is_filled());
        assert!(!result.is_full_fill());
        assert!(!result.is_partial_fill());
        assert!(result.fill_price().is_none());
        assert_eq!(result.filled_quantity(dec!(1.0)), Decimal::ZERO);
    }

    #[test]
    fn test_fill_result_full_fill() {
        let result = FillResult::FullFill {
            fill_price: dec!(100.0),
        };

        assert!(result.is_filled());
        assert!(result.is_full_fill());
        assert!(!result.is_partial_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
        assert_eq!(result.filled_quantity(dec!(5.0)), dec!(5.0));
    }

    #[test]
    fn test_fill_result_partial_fill() {
        let result = FillResult::PartialFill {
            filled_quantity: dec!(0.5),
            fill_price: dec!(100.0),
        };

        assert!(result.is_filled());
        assert!(!result.is_full_fill());
        assert!(result.is_partial_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
        assert_eq!(result.filled_quantity(dec!(1.0)), dec!(0.5));
    }

    // ImmediateFillModel tests
    #[test]
    fn test_immediate_fill_buy_fills() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);

        // Ask at 99.9 <= order price 100.0, should fill
        let tick = create_test_tick(dec!(99.8), dec!(99.9));
        let result = model.simulate_fill(&order, &tick, 0);

        assert!(result.is_full_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
    }

    #[test]
    fn test_immediate_fill_buy_no_fill() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);

        // Ask at 100.1 > order price 100.0, should not fill
        let tick = create_test_tick(dec!(100.0), dec!(100.1));
        let result = model.simulate_fill(&order, &tick, 0);

        assert!(!result.is_filled());
    }

    #[test]
    fn test_immediate_fill_sell_fills() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Side::Sell, dec!(100.0), dec!(1.0), 1000);

        // Bid at 100.1 >= order price 100.0, should fill
        let tick = create_test_tick(dec!(100.1), dec!(100.2));
        let result = model.simulate_fill(&order, &tick, 0);

        assert!(result.is_full_fill());
        assert_eq!(result.fill_price(), Some(dec!(100.0)));
    }

    #[test]
    fn test_immediate_fill_sell_no_fill() {
        let model = ImmediateFillModel::new();
        let order = SimulatedOrder::new(Side::Sell, dec!(100.0), dec!(1.0), 1000);

        // Bid at 99.9 < order price 100.0, should not fill
        let tick = create_test_tick(dec!(99.9), dec!(100.1));
        let result = model.simulate_fill(&order, &tick, 0);

        assert!(!result.is_filled());
    }

    #[test]
    fn test_immediate_fill_name() {
        let model = ImmediateFillModel::new();
        assert_eq!(model.name(), "ImmediateFill");
    }

    // QueuePositionFillModel tests
    #[test]
    fn test_queue_position_new() {
        let model = QueuePositionFillModel::new(dec!(0.1));
        assert_eq!(model.fill_rate, dec!(0.1));
    }

    #[test]
    fn test_queue_position_update_queue() {
        let mut model = QueuePositionFillModel::new(dec!(0.1));
        let tick = MarketTick::new(1000, dec!(100.0), dec!(50.0), dec!(100.1), dec!(30.0));

        model.update_queue(&tick);

        assert_eq!(model.get_queue_depth(dec!(100.0)), dec!(50.0));
        assert_eq!(model.get_queue_depth(dec!(100.1)), dec!(30.0));
    }

    #[test]
    fn test_queue_position_no_fill_market_not_crossed() {
        let model = QueuePositionFillModel::new(dec!(0.1));
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);

        // Ask at 100.1 > order price, market not crossed
        let tick = create_test_tick(dec!(99.9), dec!(100.1));
        let result = model.simulate_fill(&order, &tick, 1000);

        assert!(!result.is_filled());
    }

    #[test]
    fn test_queue_position_fill_after_queue_clears() {
        let mut model = QueuePositionFillModel::new(dec!(0.1)); // 0.1 per ms
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);

        // Set up queue with 5 units ahead
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(5.0));
        model.update_queue(&tick);

        // After 60ms: cleared 6 units, queue was 5, order is 1 -> full fill
        let result = model.simulate_fill(&order, &tick, 60);
        assert!(result.is_full_fill());
    }

    #[test]
    fn test_queue_position_partial_fill() {
        let mut model = QueuePositionFillModel::new(dec!(0.1)); // 0.1 per ms
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(2.0), 1000);

        // Set up queue with 5 units ahead
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(5.0));
        model.update_queue(&tick);

        // After 60ms: cleared 6 units, queue was 5, partial fill of 1 unit
        let result = model.simulate_fill(&order, &tick, 60);
        assert!(result.is_partial_fill());
        assert_eq!(result.filled_quantity(dec!(2.0)), dec!(1.0));
    }

    #[test]
    fn test_queue_position_min_queue_time() {
        let model = QueuePositionFillModel::with_min_queue_time(dec!(0.1), 100);
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9)); // Market crossed

        // Before min time
        let result = model.simulate_fill(&order, &tick, 50);
        assert!(!result.is_filled());

        // After min time
        let _result = model.simulate_fill(&order, &tick, 100);
        // May or may not fill depending on queue, but min time check passed
    }

    #[test]
    fn test_queue_position_reset() {
        let mut model = QueuePositionFillModel::new(dec!(0.1));
        let tick = create_test_tick(dec!(100.0), dec!(100.1));
        model.update_queue(&tick);

        assert!(model.get_queue_depth(dec!(100.0)) > Decimal::ZERO);

        model.reset();

        assert_eq!(model.get_queue_depth(dec!(100.0)), Decimal::ZERO);
    }

    // ProbabilisticFillModel tests
    #[test]
    fn test_probabilistic_new() {
        let model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.001), 42);

        assert_eq!(model.base_probability, dec!(0.5));
        assert_eq!(model.depth_factor, dec!(1.0));
        assert_eq!(model.seed, 42);
    }

    #[test]
    fn test_probabilistic_calculate_probability() {
        let model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));

        let prob = model.calculate_probability(&order, &tick, 100);

        // Probability should be between 0 and base_probability
        assert!(prob >= Decimal::ZERO);
        assert!(prob <= dec!(0.5));
    }

    #[test]
    fn test_probabilistic_probability_increases_with_time() {
        let model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = MarketTick::new(1000, dec!(99.9), dec!(10.0), dec!(100.0), dec!(10.0));

        let prob_early = model.calculate_probability(&order, &tick, 10);
        let prob_late = model.calculate_probability(&order, &tick, 100);

        assert!(prob_late > prob_early);
    }

    #[test]
    fn test_probabilistic_deterministic() {
        let model1 = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let model2 = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);

        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9)); // Market crossed

        // Same seed should produce same results
        let result1 = model1.simulate_fill(&order, &tick, 100);
        let result2 = model2.simulate_fill(&order, &tick, 100);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_probabilistic_reset() {
        let mut model = ProbabilisticFillModel::new(dec!(0.5), dec!(1.0), dec!(0.01), 42);
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9));

        let result1 = model.simulate_fill(&order, &tick, 100);
        model.reset();
        let result2 = model.simulate_fill(&order, &tick, 100);

        // After reset, should get same result
        assert_eq!(result1, result2);
    }

    // MarketImpactFillModel tests
    #[test]
    fn test_market_impact_calculate_impact() {
        let model =
            MarketImpactFillModel::new(dec!(0.1), dec!(1000000.0), ImmediateFillModel::new());

        // Impact = 0.1 * sqrt(100 / 1000000) = 0.1 * 0.01 = 0.001
        let impact = model.calculate_impact(dec!(100.0));
        assert!(impact > Decimal::ZERO);
        assert!(impact < dec!(0.01));
    }

    #[test]
    fn test_market_impact_buy_increases_price() {
        let model = MarketImpactFillModel::new(dec!(1.0), dec!(1000.0), ImmediateFillModel::new());

        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(100.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(99.9)); // Market crossed

        let result = model.simulate_fill(&order, &tick, 0);

        // Buy should have higher fill price due to impact
        if let FillResult::FullFill { fill_price } = result {
            assert!(fill_price > dec!(100.0));
        } else {
            panic!("Expected full fill");
        }
    }

    #[test]
    fn test_market_impact_sell_decreases_price() {
        let model = MarketImpactFillModel::new(dec!(1.0), dec!(1000.0), ImmediateFillModel::new());

        let order = SimulatedOrder::new(Side::Sell, dec!(100.0), dec!(100.0), 1000);
        let tick = create_test_tick(dec!(100.1), dec!(100.2)); // Market crossed

        let result = model.simulate_fill(&order, &tick, 0);

        // Sell should have lower fill price due to impact
        if let FillResult::FullFill { fill_price } = result {
            assert!(fill_price < dec!(100.0));
        } else {
            panic!("Expected full fill");
        }
    }

    #[test]
    fn test_market_impact_no_fill_passes_through() {
        let model =
            MarketImpactFillModel::new(dec!(0.1), dec!(1000000.0), ImmediateFillModel::new());

        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let tick = create_test_tick(dec!(99.9), dec!(100.1)); // Market not crossed

        let result = model.simulate_fill(&order, &tick, 0);
        assert!(!result.is_filled());
    }

    #[test]
    fn test_market_impact_reset() {
        let mut model =
            MarketImpactFillModel::new(dec!(0.1), dec!(1000000.0), ImmediateFillModel::new());

        model.reset(); // Should not panic
        assert_eq!(model.name(), "MarketImpact");
    }

    // decimal_sqrt tests
    #[test]
    fn test_decimal_sqrt_positive() {
        let result = decimal_sqrt(dec!(4.0)).unwrap();
        assert!((result - dec!(2.0)).abs() < dec!(0.0001));
    }

    #[test]
    fn test_decimal_sqrt_zero() {
        assert_eq!(decimal_sqrt(Decimal::ZERO), Some(Decimal::ZERO));
    }

    #[test]
    fn test_decimal_sqrt_negative() {
        assert!(decimal_sqrt(dec!(-1.0)).is_none());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_simulated_order_serialization() {
        let order = SimulatedOrder::new(Side::Buy, dec!(100.0), dec!(1.0), 1000);
        let json = serde_json::to_string(&order).unwrap();
        let deserialized: SimulatedOrder = serde_json::from_str(&json).unwrap();
        assert_eq!(order, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_fill_result_serialization() {
        let result = FillResult::FullFill {
            fill_price: dec!(100.0),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: FillResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }
}
