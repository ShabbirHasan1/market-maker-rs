//! Inventory position tracking.

#[cfg(feature = "serde")]
use pretty_simple_display::{DebugPretty, DisplaySimple};

/// Represents the market maker's current inventory position.
#[derive(Clone, PartialEq)]
#[cfg_attr(not(feature = "serde"), derive(Debug))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, DebugPretty, DisplaySimple))]
pub struct InventoryPosition {
    /// Current quantity held.
    ///
    /// Positive = long position, Negative = short position, Zero = flat.
    pub quantity: f64,

    /// Average entry price for the current position.
    pub avg_entry_price: f64,

    /// Timestamp of last position update, in milliseconds since Unix epoch.
    pub last_update: u64,
}

impl InventoryPosition {
    /// Creates a new flat (zero) position.
    #[must_use]
    pub fn new() -> Self {
        Self {
            quantity: 0.0,
            avg_entry_price: 0.0,
            last_update: 0,
        }
    }

    /// Returns true if the position is flat (zero).
    #[must_use]
    pub fn is_flat(&self) -> bool {
        self.quantity.abs() < f64::EPSILON
    }

    /// Returns true if the position is long (positive).
    #[must_use]
    pub fn is_long(&self) -> bool {
        self.quantity > f64::EPSILON
    }

    /// Returns true if the position is short (negative).
    #[must_use]
    pub fn is_short(&self) -> bool {
        self.quantity < -f64::EPSILON
    }
}

impl Default for InventoryPosition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position_is_flat() {
        let position = InventoryPosition::new();
        assert_eq!(position.quantity, 0.0);
        assert_eq!(position.avg_entry_price, 0.0);
        assert_eq!(position.last_update, 0);
        assert!(position.is_flat());
    }

    #[test]
    fn test_default_position() {
        let position = InventoryPosition::default();
        assert_eq!(position.quantity, 0.0);
        assert!(position.is_flat());
    }

    #[test]
    fn test_is_long() {
        let position = InventoryPosition {
            quantity: 10.0,
            avg_entry_price: 100.0,
            last_update: 1000,
        };
        assert!(position.is_long());
        assert!(!position.is_flat());
        assert!(!position.is_short());
    }

    #[test]
    fn test_is_short() {
        let position = InventoryPosition {
            quantity: -10.0,
            avg_entry_price: 100.0,
            last_update: 1000,
        };
        assert!(position.is_short());
        assert!(!position.is_flat());
        assert!(!position.is_long());
    }

    #[test]
    fn test_is_flat() {
        let position = InventoryPosition {
            quantity: 0.0,
            avg_entry_price: 100.0,
            last_update: 1000,
        };
        assert!(position.is_flat());
        assert!(!position.is_long());
        assert!(!position.is_short());
    }

    #[test]
    fn test_very_small_position_is_flat() {
        let position = InventoryPosition {
            quantity: 1e-16,
            avg_entry_price: 100.0,
            last_update: 1000,
        };
        assert!(position.is_flat());
    }
}
