//! PnL (Profit and Loss) calculations.

#[cfg(feature = "serde")]
use pretty_simple_display::{DebugPretty, DisplaySimple};

/// Represents profit and loss information.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "serde"), derive(Debug))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, DebugPretty, DisplaySimple))]
pub struct PnL {
    /// Realized PnL from closed positions.
    pub realized: f64,

    /// Unrealized PnL from current open position.
    pub unrealized: f64,

    /// Total PnL (realized + unrealized).
    pub total: f64,
}

impl PnL {
    /// Creates a new PnL with zero values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            realized: 0.0,
            unrealized: 0.0,
            total: 0.0,
        }
    }
}

impl Default for PnL {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_pnl() {
        let pnl = PnL::new();
        assert_eq!(pnl.realized, 0.0);
        assert_eq!(pnl.unrealized, 0.0);
        assert_eq!(pnl.total, 0.0);
    }

    #[test]
    fn test_default_pnl() {
        let pnl = PnL::default();
        assert_eq!(pnl.realized, 0.0);
        assert_eq!(pnl.unrealized, 0.0);
        assert_eq!(pnl.total, 0.0);
    }

    #[test]
    fn test_pnl_with_values() {
        let pnl = PnL {
            realized: 100.0,
            unrealized: 50.0,
            total: 150.0,
        };
        assert_eq!(pnl.realized, 100.0);
        assert_eq!(pnl.unrealized, 50.0);
        assert_eq!(pnl.total, 150.0);
    }
}
