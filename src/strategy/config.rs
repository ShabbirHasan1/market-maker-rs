//! Strategy configuration parameters.

use crate::types::error::{MMError, MMResult};

#[cfg(feature = "serde")]
use pretty_simple_display::{DebugPretty, DisplaySimple};

/// Configuration parameters for the Avellaneda-Stoikov strategy.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "serde"), derive(Debug))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, DebugPretty, DisplaySimple))]
pub struct StrategyConfig {
    /// Risk aversion parameter (gamma).
    ///
    /// Higher values make the strategy more conservative.
    /// Must be positive.
    pub risk_aversion: f64,

    /// Order intensity parameter (k).
    ///
    /// Models how frequently market orders arrive.
    /// Must be positive.
    pub order_intensity: f64,

    /// Terminal time (end of trading session) in milliseconds since Unix epoch.
    pub terminal_time: u64,

    /// Minimum spread constraint, in price units.
    ///
    /// Ensures quotes don't cross or get too tight.
    /// Must be non-negative.
    pub min_spread: f64,
}

impl StrategyConfig {
    /// Creates a new strategy configuration with validation.
    ///
    /// # Arguments
    ///
    /// * `risk_aversion` - Risk aversion parameter (gamma), must be positive
    /// * `order_intensity` - Order intensity parameter (k), must be positive
    /// * `terminal_time` - Terminal time in milliseconds
    /// * `min_spread` - Minimum spread, must be non-negative
    ///
    /// # Errors
    ///
    /// Returns `MMError::InvalidConfiguration` if parameters are invalid.
    pub fn new(
        risk_aversion: f64,
        order_intensity: f64,
        terminal_time: u64,
        min_spread: f64,
    ) -> MMResult<Self> {
        if risk_aversion <= 0.0 {
            return Err(MMError::InvalidConfiguration(
                "risk_aversion must be positive".to_string(),
            ));
        }

        if order_intensity <= 0.0 {
            return Err(MMError::InvalidConfiguration(
                "order_intensity must be positive".to_string(),
            ));
        }

        if min_spread < 0.0 {
            return Err(MMError::InvalidConfiguration(
                "min_spread must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            risk_aversion,
            order_intensity,
            terminal_time,
            min_spread,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = StrategyConfig::new(0.5, 1.5, 1000, 0.01);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.risk_aversion, 0.5);
        assert_eq!(config.order_intensity, 1.5);
        assert_eq!(config.terminal_time, 1000);
        assert_eq!(config.min_spread, 0.01);
    }

    #[test]
    fn test_invalid_risk_aversion_zero() {
        let config = StrategyConfig::new(0.0, 1.5, 1000, 0.01);
        assert!(config.is_err());
        assert!(matches!(
            config.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn test_invalid_risk_aversion_negative() {
        let config = StrategyConfig::new(-0.5, 1.5, 1000, 0.01);
        assert!(config.is_err());
        if let Err(MMError::InvalidConfiguration(msg)) = config {
            assert!(msg.contains("risk_aversion must be positive"));
        }
    }

    #[test]
    fn test_invalid_order_intensity_zero() {
        let config = StrategyConfig::new(0.5, 0.0, 1000, 0.01);
        assert!(config.is_err());
        assert!(matches!(
            config.unwrap_err(),
            MMError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn test_invalid_order_intensity_negative() {
        let config = StrategyConfig::new(0.5, -1.5, 1000, 0.01);
        assert!(config.is_err());
        if let Err(MMError::InvalidConfiguration(msg)) = config {
            assert!(msg.contains("order_intensity must be positive"));
        }
    }

    #[test]
    fn test_invalid_min_spread_negative() {
        let config = StrategyConfig::new(0.5, 1.5, 1000, -0.01);
        assert!(config.is_err());
        if let Err(MMError::InvalidConfiguration(msg)) = config {
            assert!(msg.contains("min_spread must be non-negative"));
        }
    }

    #[test]
    fn test_valid_min_spread_zero() {
        let config = StrategyConfig::new(0.5, 1.5, 1000, 0.0);
        assert!(config.is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_config_display() {
        let config = StrategyConfig::new(0.5, 1.5, 1000, 0.01).unwrap();
        let display_str = format!("{}", config);
        assert!(display_str.contains("risk_aversion"));
        assert!(display_str.contains("0.5"));
        assert!(display_str.contains("order_intensity"));
        assert!(display_str.contains("1.5"));
    }
}
