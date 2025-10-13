//! Position management module for tracking inventory and PnL.
//!
//! This module handles:
//! - Position updates when orders fill
//! - Average entry price calculation
//! - Realized and unrealized PnL tracking
//! - Position flattening (closing to zero)

/// Inventory position tracking.
pub mod inventory;

/// PnL (Profit and Loss) calculations.
pub mod pnl;
