//! Market state snapshot representation.

#[cfg(feature = "serde")]
use pretty_simple_display::{DebugPretty, DisplaySimple};

/// Represents the observable state of the market at a point in time.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "serde"), derive(Debug))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, DebugPretty, DisplaySimple))]
pub struct MarketState {
    /// Mid-price of the asset.
    pub mid_price: f64,

    /// Volatility estimate (annualized).
    pub volatility: f64,

    /// Current timestamp in milliseconds since Unix epoch.
    pub timestamp: u64,
}

impl MarketState {
    /// Creates a new market state snapshot.
    ///
    /// # Arguments
    ///
    /// * `mid_price` - Current mid-price of the asset
    /// * `volatility` - Volatility estimate (annualized)
    /// * `timestamp` - Current timestamp in milliseconds
    #[must_use]
    pub fn new(mid_price: f64, volatility: f64, timestamp: u64) -> Self {
        Self {
            mid_price,
            volatility,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_state_new() {
        let state = MarketState::new(100.0, 0.2, 1234567890);
        assert_eq!(state.mid_price, 100.0);
        assert_eq!(state.volatility, 0.2);
        assert_eq!(state.timestamp, 1234567890);
    }

    #[test]
    fn test_market_state_creation() {
        let state = MarketState {
            mid_price: 99.5,
            volatility: 0.15,
            timestamp: 9876543210,
        };
        assert_eq!(state.mid_price, 99.5);
        assert_eq!(state.volatility, 0.15);
        assert_eq!(state.timestamp, 9876543210);
    }
}
