# Declare parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import torch


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seeds(42)

# 4.1 Parameter Settings
MM_PARAMS = {
    's0': 0.001,  # Half-spread parameter
    'q0': 0.1,  # Base order size (BTC)
    'ks': 0.0005,  # Inventory skew strength
    'I_max': 5.0,  # Max inventory (BTC)
    'Pi_min': 45000,  # Minimum equity
    'Pi_init': 50000,  # Initial equity
    'T_days': 14,  # Simulation days
    'M_paths': 50  # Monte Carlo paths
}

# Time grid settings (5-minute frequency)
N_steps = MM_PARAMS['T_days'] * 24 * 12  # 14 days * 24 hours * 12 five-minutes
print(f"Simulation setup: {MM_PARAMS['T_days']} days, {N_steps} time steps, {MM_PARAMS['M_paths']} paths")


# 4.2 State Variables and Market Maker Class
class MarketMaker:
    def __init__(self, params):
        self.params = params
        self.reset()

    def reset(self):
        """Reset state"""
        self.S = 0.0  # Spot price
        self.I = 0.0  # Inventory
        self.cash = self.params['Pi_init']  # Cash
        self.equity = self.params['Pi_init']  # Equity

    @property
    def phi(self):
        """Normalized inventory"""
        return self.I / self.params['I_max']

    def compute_quotes(self):
        """Compute bid/ask quotes"""
        s0, ks = self.params['s0'], self.params['ks']

        # Base quotes
        P_bid = self.S * (1 - s0 - ks * self.phi)
        P_ask = self.S * (1 + s0 + ks * self.phi)

        # Size control
        q = self.params['q0'] * max(0, 1 - abs(self.phi))

        # Hard kill-switch
        if abs(self.I) >= self.params['I_max']:
            if self.I > 0:  # Net long, only sell
                return P_bid, 0, P_ask, q
            else:  # Net short, only buy
                return P_bid, q, P_ask, 0
        else:
            return P_bid, q, P_ask, q

    def update_fills(self, P_bid, q_bid, P_ask, q_ask, H, L, S_next):
        """Update fill logic"""
        # Check fill conditions
        bid_filled = L <= P_bid <= H and q_bid > 0
        ask_filled = L <= P_ask <= H and q_ask > 0

        # Update inventory and cash
        if bid_filled:
            self.I += q_bid
            self.cash -= q_bid * P_bid

        if ask_filled:
            self.I -= q_ask
            self.cash += q_ask * P_ask

        # Update price and equity
        self.S = S_next
        self.equity = self.cash + self.I * self.S

        return bid_filled, ask_filled

    def risk_breach(self):
        """Check risk breach"""
        return (abs(self.I) > self.params['I_max'] or
                self.equity < self.params['Pi_min'])


# 4.3 BTC Price Simulator (Simplified)
def simulate_btc_paths(n_paths, n_steps, initial_price=50000):
    """Simulate BTC price paths"""
    paths = []
    for _ in range(n_paths):
        prices = [initial_price]
        for i in range(1, n_steps):
            # Geometric Brownian Motion
            dt = 1 / (365 * 24 * 12)  # 5-minute time step
            mu = 0.0001  # Daily drift
            sigma = 0.02  # Daily volatility

            drift = (mu - 0.5 * sigma ** 2) * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            price = prices[-1] * np.exp(drift + shock)
            prices.append(price)

        # Generate OHLC data (simplified: based on close prices)
        ohlc = generate_ohlc_from_close(prices)
        paths.append(ohlc)

    return paths


def generate_ohlc_from_close(close_prices):
    """Generate OHLC data from close prices"""
    n = len(close_prices)
    open_prices = [close_prices[0]] + close_prices[:-1]
    high_prices = [max(o, c) + abs(o - c) * 0.1 for o, c in zip(open_prices, close_prices)]
    low_prices = [min(o, c) - abs(o - c) * 0.1 for o, c in zip(open_prices, close_prices)]

    return {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }


# 4.5 Monte Carlo Simulation
def run_mm_simulation():
    """Run market maker Monte Carlo simulation"""
    print("Starting market maker simulation...")

    # Generate BTC price paths
    btc_paths = simulate_btc_paths(
        MM_PARAMS['M_paths'],
        N_steps + 1,  # +1 to include initial point
        initial_price=50000
    )

    # Store results
    results = {
        'equity_paths': [],
        'inventory_paths': [],
        'risk_breaches': [],
        'final_equity': [],
        'max_inventory': []
    }

    for m in range(MM_PARAMS['M_paths']):
        mm = MarketMaker(MM_PARAMS)
        ohlc = btc_paths[m]

        # Initialize
        mm.S = ohlc['close'][0]
        equity_path = [mm.equity]
        inventory_path = [mm.I]
        breach_time = None

        # Time step loop
        for n in range(N_steps):
            # Compute quotes
            P_bid, q_bid, P_ask, q_ask = mm.compute_quotes()

            # Get next candle OHLC
            H, L = ohlc['high'][n + 1], ohlc['low'][n + 1]
            S_next = ohlc['close'][n + 1]

            # Update fills
            mm.update_fills(P_bid, q_bid, P_ask, q_ask, H, L, S_next)

            # Check risk breach
            if breach_time is None and mm.risk_breach():
                breach_time = n

            # Store state
            equity_path.append(mm.equity)
            inventory_path.append(mm.I)

        # Store path results
        results['equity_paths'].append(equity_path)
        results['inventory_paths'].append(inventory_path)
        results['risk_breaches'].append(breach_time)
        results['final_equity'].append(mm.equity)
        results['max_inventory'].append(max(abs(i) for i in inventory_path))

    print(f"Simulation completed! Risk breach paths: {sum(1 for x in results['risk_breaches'] if x is not None)}")
    return results


# Run simulation
results = run_mm_simulation()


# 4.6 Diagnostic Analysis
def plot_diagnostics(results):
    """Plot diagnostic charts"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Sample Equity Paths
    sample_paths = min(5, MM_PARAMS['M_paths'])
    for i in range(sample_paths):
        equity = results['equity_paths'][i]
        days = np.arange(len(equity)) / (24 * 12)  # Convert to days
        axes[0, 0].plot(days, equity, alpha=0.7, label=f'Path {i + 1}')

    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Equity')
    axes[0, 0].set_title('Sample Equity Paths')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Sample Inventory Paths
    for i in range(sample_paths):
        inventory = results['inventory_paths'][i]
        days = np.arange(len(inventory)) / (24 * 12)
        axes[0, 1].plot(days, inventory, alpha=0.7, label=f'Path {i + 1}')

    axes[0, 1].axhline(y=MM_PARAMS['I_max'], color='r', linestyle='--', label='Max Inventory')
    axes[0, 1].axhline(y=-MM_PARAMS['I_max'], color='r', linestyle='--')
    axes[0, 1].set_xlabel('Days')
    axes[0, 1].set_ylabel('Inventory (BTC)')
    axes[0, 1].set_title('Sample Inventory Paths')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Final Equity Distribution
    axes[0, 2].hist(results['final_equity'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(x=MM_PARAMS['Pi_init'], color='r', linestyle='--', label='Initial Equity')
    axes[0, 2].set_xlabel('Final Equity')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Final Equity Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # 4. Max Drawdown Distribution
    drawdowns = []
    for equity_path in results['equity_paths']:
        peak = np.maximum.accumulate(equity_path)
        drawdown = (peak - equity_path) / peak
        drawdowns.append(np.max(drawdown))

    axes[1, 0].hist(drawdowns, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Max Drawdown')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Max Drawdown Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Max Inventory Distribution
    axes[1, 1].hist(results['max_inventory'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=MM_PARAMS['I_max'], color='r', linestyle='--', label='Max Inventory Limit')
    axes[1, 1].set_xlabel('Max Inventory (BTC)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Max Inventory Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # 6. Risk Breach Times
    breach_times = [t for t in results['risk_breaches'] if t is not None]
    if breach_times:
        axes[1, 2].hist(breach_times, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Breach Time Step')
        axes[1, 2].set_ylabel('Frequency')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Risk Breaches', ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Risk Breach Time Distribution')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Output statistical summary
    print("\n=== Market Maker Strategy Statistical Summary ===")
    print(f"Final Equity Statistics:")
    print(f"  Mean: {np.mean(results['final_equity']):.2f}")
    print(f"  Std: {np.std(results['final_equity']):.2f}")
    print(f"  Min: {np.min(results['final_equity']):.2f}")
    print(f"  Max: {np.max(results['final_equity']):.2f}")

    print(f"\nInventory Statistics:")
    print(f"  Average Max Inventory: {np.mean(results['max_inventory']):.2f} BTC")
    print(f"  Risk Breach Paths: {len(breach_times)}/{MM_PARAMS['M_paths']}")

    print(f"\nDrawdown Statistics:")
    print(f"  Average Max Drawdown: {np.mean(drawdowns):.2%}")
    print(f"  Max Drawdown: {np.max(drawdowns):.2%}")


# Plot diagnostic charts
plot_diagnostics(results)

print("\nMarket Maker Baseline Strategy Implementation Completed!")

