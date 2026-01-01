# Market Maker RS - Integration Roadmap

This document describes the features and improvements needed to integrate `market-maker-rs` into the global options trading project ecosystem.

## Current State

`market-maker-rs` v0.2.0 is a comprehensive library implementing quantitative market making strategies based on the Avellaneda-Stoikov model. It provides:

### Existing Features

| Module | Description |
|--------|-------------|
| `strategy` | Avellaneda-Stoikov, GLFT, Grid, Adaptive Spread, Depth-Based |
| `position` | Inventory tracking and PnL management |
| `market_state` | Market snapshots and volatility estimation |
| `risk` | Limits, circuit breakers, alerts, portfolio risk |
| `analytics` | Order flow, VPIN, intensity estimation, live metrics |
| `execution` | Exchange connector trait, order manager, latency tracking |
| `backtest` | Event-driven engine, fill models, performance metrics |

---

## Integration Requirements

The global project consists of:

| Repository | Purpose |
|------------|---------|
| `OptionStratLib` | Options pricing, Greeks, strategies |
| `OrderBook-rs` | Lock-free order book engine |
| `PriceLevel` | Price level data structures |
| `Option-Chain-OrderBook` | Hierarchical option chain order books |
| `Option-Chain-OrderBook-Backend` | REST API backend with market maker engine |
| `Option-Chain-OrderBook-FrontEnd` | SvelteKit frontend console |

---

## Missing Features

### 1. Options-Specific Market Making

**Priority: HIGH**

The current implementation is designed for single-asset market making. Options require:

- [ ] **Greeks-Aware Quoting**: Adjust spreads based on delta, gamma, vega exposure
- [ ] **Volatility Surface Integration**: Use IV surface from `OptionStratLib` for pricing
- [ ] **Cross-Strike Hedging**: Delta-neutral portfolio management across strikes
- [ ] **Expiration-Aware Risk**: Time decay (theta) impact on inventory risk
- [ ] **Skew-Based Spread Adjustment**: Widen spreads for high-gamma strikes (ATM)
- [ ] **Term Structure Arbitrage**: Detect and exploit calendar spread opportunities

```rust
// Example: Greeks-aware quote adjustment
pub trait OptionsMarketMaker {
    fn calculate_greeks_adjusted_quotes(
        &self,
        option: &Options,
        portfolio_greeks: &PortfolioGreeks,
        risk_limits: &GreeksLimits,
    ) -> MMResult<(Decimal, Decimal)>;
}
```

### 2. OptionStratLib Integration

**Priority: HIGH**

Direct integration with `OptionStratLib` for:

- [ ] **Options Struct Compatibility**: Accept `optionstratlib::Options` directly
- [ ] **Greeks Calculation**: Use `OptionStratLib` Greeks (delta, gamma, theta, vega, rho)
- [ ] **Pricing Models**: Support Black-Scholes, Binomial, Monte Carlo from `OptionStratLib`
- [ ] **ExpirationDate Type**: Use `optionstratlib::ExpirationDate` for time handling
- [ ] **Volatility Models**: Integrate EWMA, GARCH, Heston from `OptionStratLib`

```toml
# Cargo.toml addition
[dependencies]
optionstratlib = { version = "0.14", optional = true }

[features]
options = ["dep:optionstratlib"]
```

### 3. OrderBook-rs Integration

**Priority: HIGH**

Connect market making strategies to the order book infrastructure:

- [ ] **OrderBook Connector**: Implement `ExchangeConnector` for `OrderBook-rs`
- [ ] **Order Type Support**: Map to `OrderBook-rs` order types (Limit, Market, IOC, FOK)
- [ ] **Real-Time Updates**: Subscribe to order book changes via `OrderBook-rs` channels
- [ ] **Depth Analysis**: Use order book depth for spread/size decisions
- [ ] **Quote Management**: Automatic quote refresh on book changes

```rust
// Example: OrderBook-rs connector
pub struct OrderBookConnector {
    orderbook: Arc<OrderBook>,
}

#[async_trait]
impl ExchangeConnector for OrderBookConnector {
    async fn submit_order(&self, request: OrderRequest) -> MMResult<OrderId>;
    async fn cancel_order(&self, order_id: &OrderId) -> MMResult<()>;
    async fn get_order_book(&self, symbol: &str) -> MMResult<OrderBookSnapshot>;
}
```

### 4. Option-Chain-OrderBook Integration

**Priority: HIGH**

Integration with the hierarchical option chain structure:

- [ ] **Multi-Strike Quoting**: Quote across all strikes in an expiration
- [ ] **Chain-Level Risk**: Aggregate Greeks across the entire chain
- [ ] **ATM Detection**: Automatic ATM strike identification for spread adjustment
- [ ] **Expiration Manager**: Handle multiple expirations simultaneously
- [ ] **Underlying Price Feed**: React to underlying price changes

```rust
// Example: Chain-level market maker
pub struct OptionChainMarketMaker {
    chain: Arc<OptionChainOrderBook>,
    quoter: Quoter,
    risk_manager: ChainRiskManager,
}

impl OptionChainMarketMaker {
    pub fn refresh_all_quotes(&self, underlying_price: Decimal) -> MMResult<Vec<QuoteUpdate>>;
    pub fn get_chain_greeks(&self) -> PortfolioGreeks;
}
```

### 5. Real-Time Data Feeds

**Priority: MEDIUM**

Support for live market data:

- [ ] **WebSocket Support**: Real-time price updates via WebSocket
- [ ] **Market Data Trait**: Abstract interface for different data providers
- [ ] **Underlying Price Stream**: Continuous underlying price updates
- [ ] **IV Surface Updates**: Real-time implied volatility updates
- [ ] **Trade Feed**: Process incoming trades for VPIN/toxicity

```rust
#[async_trait]
pub trait MarketDataFeed {
    async fn subscribe_underlying(&self, symbol: &str) -> mpsc::Receiver<PriceUpdate>;
    async fn subscribe_trades(&self, symbol: &str) -> mpsc::Receiver<Trade>;
    async fn get_iv_surface(&self, symbol: &str) -> MMResult<IVSurface>;
}
```

### 6. REST/WebSocket API Layer

**Priority: MEDIUM**

HTTP and WebSocket endpoints for the backend:

- [ ] **Axum Integration**: Route handlers for market maker control
- [ ] **Configuration API**: Enable/disable, adjust parameters via REST
- [ ] **Status Endpoints**: Current quotes, positions, Greeks, PnL
- [ ] **WebSocket Events**: Real-time quote updates, fills, alerts
- [ ] **OpenAPI Documentation**: Utoipa schemas for all endpoints

```rust
// Example: API routes
pub fn market_maker_routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/mm/config", get(get_config).put(update_config))
        .route("/api/v1/mm/quotes", get(get_all_quotes))
        .route("/api/v1/mm/positions", get(get_positions))
        .route("/api/v1/mm/greeks", get(get_portfolio_greeks))
        .route("/api/v1/mm/ws", get(websocket_handler))
}
```

### 7. Persistence Layer

**Priority: MEDIUM**

Database integration for state persistence:

- [ ] **Trade History**: Store all fills with timestamps
- [ ] **Position Snapshots**: Periodic position state saves
- [ ] **Configuration Storage**: Persist MM configuration
- [ ] **Performance Metrics**: Historical PnL, Sharpe, drawdown
- [ ] **SQLx Integration**: PostgreSQL support matching backend

```rust
pub trait MarketMakerRepository {
    async fn save_fill(&self, fill: &Fill) -> Result<()>;
    async fn get_fills(&self, from: DateTime, to: DateTime) -> Result<Vec<Fill>>;
    async fn save_position_snapshot(&self, snapshot: &PositionSnapshot) -> Result<()>;
    async fn get_daily_pnl(&self, date: NaiveDate) -> Result<DailyPnL>;
}
```

### 8. Greeks-Based Risk Management

**Priority: HIGH**

Options-specific risk controls:

- [ ] **Delta Limits**: Maximum portfolio delta exposure
- [ ] **Gamma Limits**: Maximum gamma (convexity risk)
- [ ] **Vega Limits**: Maximum volatility exposure
- [ ] **Theta Monitoring**: Track time decay impact
- [ ] **Greeks Circuit Breakers**: Halt quoting on limit breach
- [ ] **Auto-Hedging**: Automatic delta hedging triggers

```rust
pub struct GreeksLimits {
    pub max_delta: Decimal,
    pub max_gamma: Decimal,
    pub max_vega: Decimal,
    pub max_theta: Decimal,
}

pub struct GreeksRiskManager {
    limits: GreeksLimits,
    current_greeks: PortfolioGreeks,
}

impl GreeksRiskManager {
    pub fn check_order(&self, order: &Order, option: &Options) -> MMResult<bool>;
    pub fn calculate_hedge_order(&self) -> Option<HedgeOrder>;
}
```

### 9. Multi-Underlying Support

**Priority: MEDIUM**

Handle multiple underlying assets:

- [ ] **Underlying Manager**: Coordinate MM across BTC, ETH, SPX, etc.
- [ ] **Cross-Asset Correlation**: Use correlation matrix for portfolio risk
- [ ] **Capital Allocation**: Distribute capital across underlyings
- [ ] **Unified Risk View**: Aggregate risk across all underlyings

### 10. Event System

**Priority: MEDIUM**

Structured event handling for the frontend:

- [ ] **Event Types**: Quote updates, fills, alerts, config changes
- [ ] **Broadcast Channels**: Tokio broadcast for event distribution
- [ ] **Event Serialization**: Serde-compatible event structs
- [ ] **Event History**: Recent events buffer for reconnection

```rust
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum MarketMakerEvent {
    QuoteUpdated { symbol: String, strike: u64, bid: u64, ask: u64 },
    OrderFilled { order_id: String, side: Side, price: u64, qty: u64 },
    GreeksUpdated { delta: f64, gamma: f64, vega: f64, theta: f64 },
    AlertTriggered { level: AlertLevel, message: String },
    CircuitBreakerTripped { reason: String },
}
```

---

## Implementation Phases

### Phase 1: Core Integration (Weeks 1-2)

1. Add `optionstratlib` as optional dependency
2. Create `Options` adapter for existing strategies
3. Implement Greeks calculation integration
4. Add `ExpirationDate` support

### Phase 2: OrderBook Integration (Weeks 3-4)

1. Implement `OrderBookConnector` for `orderbook-rs`
2. Create order type mappings
3. Add real-time book subscription
4. Implement depth-based quoting

### Phase 3: Option Chain Support (Weeks 5-6)

1. Multi-strike quoting engine
2. Chain-level risk aggregation
3. ATM detection and spread adjustment
4. Expiration management

### Phase 4: Risk & Greeks (Weeks 7-8)

1. Greeks-based risk limits
2. Portfolio Greeks tracking
3. Auto-hedging logic
4. Greeks circuit breakers

### Phase 5: API & Persistence (Weeks 9-10)

1. REST API endpoints
2. WebSocket event streaming
3. Database integration
4. Configuration persistence

---

## Dependency Changes

```toml
[dependencies]
# Existing
orderbook-rs = "0.4"
rust_decimal = "1.39"
thiserror = "2.0"
async-trait = "0.1"

# New required
optionstratlib = { version = "0.14", optional = true }
option-chain-orderbook = { version = "0.1", optional = true }

# New optional (for backend integration)
axum = { version = "0.8", optional = true }
tokio = { version = "1.48", features = ["full"], optional = true }
sqlx = { version = "0.8", features = ["postgres"], optional = true }
utoipa = { version = "5.4", optional = true }

[features]
default = []
options = ["dep:optionstratlib"]
chain = ["options", "dep:option-chain-orderbook"]
api = ["dep:axum", "dep:tokio", "dep:utoipa"]
persistence = ["dep:sqlx"]
full = ["options", "chain", "api", "persistence", "prometheus"]
```

---

## API Compatibility

### Current Backend Integration

The `Option-Chain-OrderBook-Backend` already has a basic market maker implementation in `src/market_maker/`. The goal is to replace this with `market-maker-rs` components:

| Backend Component | market-maker-rs Replacement |
|-------------------|----------------------------|
| `MarketMakerConfig` | `strategy::StrategyConfig` |
| `Quoter` | `strategy::adaptive_spread` + Greeks |
| `OptionPricer` | `OptionStratLib` integration |
| `MarketMakerEngine` | New `OptionsMarketMaker` |

---

## Testing Requirements

- [ ] Unit tests for all new traits and structs
- [ ] Integration tests with `OptionStratLib`
- [ ] Integration tests with `OrderBook-rs`
- [ ] Benchmark tests for quote generation latency
- [ ] Property-based tests for risk limit enforcement

---

## Documentation Requirements

- [ ] Module-level documentation for all new modules
- [ ] Example code for each integration point
- [ ] User guide update for options market making
- [ ] API reference for new public types
- [ ] Architecture diagram showing component relationships

---

## Success Criteria

1. **Quote Latency**: < 100μs for single option quote generation
2. **Chain Refresh**: < 10ms to refresh all quotes in a 50-strike chain
3. **Greeks Accuracy**: Match `OptionStratLib` Greeks within 0.01%
4. **Risk Enforcement**: Zero tolerance for limit breaches
5. **API Response**: < 5ms for REST endpoints
6. **WebSocket Latency**: < 1ms event broadcast

---

## Contact

- **Author**: Joaquín Béjar García
- **Email**: jb@taunais.com
- **Repository**: https://github.com/joaquinbejar/market-maker-rs
