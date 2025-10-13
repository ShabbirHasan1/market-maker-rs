//! Primitive type aliases for market making domain concepts.

/// Price value in the market, represented as f64.
pub type Price = f64;

/// Quantity or size of an order/position, represented as f64.
///
/// Positive values indicate long positions, negative values indicate short positions.
pub type Quantity = f64;

/// Timestamp in milliseconds since Unix epoch.
pub type Timestamp = u64;

/// Volatility value (annualized), represented as f64.
pub type Volatility = f64;

/// Risk aversion parameter (gamma), represented as f64.
pub type RiskAversion = f64;

/// Order intensity parameter (k), represented as f64.
pub type OrderIntensity = f64;
