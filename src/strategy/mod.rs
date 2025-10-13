//! Strategy module containing pure mathematical calculations for market making.
//!
//! This module implements the Avellaneda-Stoikov model, which solves the optimal
//! market making problem using stochastic control theory.
//!
//! # Key Formulas
//!
//! ## Reservation Price
//! ```text
//! r = s - q * γ * σ² * (T - t)
//! ```
//!
//! ## Optimal Spread
//! ```text
//! spread = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
//! ```
//!
//! ## Optimal Quotes
//! ```text
//! bid = reservation_price - spread/2
//! ask = reservation_price + spread/2
//! ```

/// Core Avellaneda-Stoikov model calculations.
pub mod avellaneda_stoikov;

/// Quote generation logic.
pub mod quote;

/// Strategy configuration.
pub mod config;
