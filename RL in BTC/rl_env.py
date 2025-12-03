import numpy as np
import pandas as pd
from enum import Enum
import random
import torch


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seeds(42)


# 5.1 MDP Structure
class HedgingEnvironment:
    def __init__(self):
        self.strikes = [0.9, 1.0, 1.1]  # Moneyness ratios
        self.maturities = [1, 7]  # Days
        self.option_types = ['call', 'put']
        self.lot_size = 0.1  # BTC per option contract
        self.transaction_cost = 0.0005  # 0.05%

        # Initialize state
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        # Market state
        self.S = 50000  # BTC price
        self.local_vol = 0.6  # Local volatility
        self.time_step = 0
        self.max_steps = 100

        # MM state
        self.I = 0.0  # Inventory
        self.cash = 100000  # Cash
        self.mm_equity = self.cash + self.I * self.S

        # Option portfolio
        self.option_positions = self._initialize_option_positions()
        self.option_prices = self._compute_option_prices()

        # Greeks
        self.portfolio_delta = self.I  # MM delta = inventory
        self.portfolio_vega = 0.0

        # History
        self.equity_history = [self.total_equity]
        self.delta_history = [self.portfolio_delta]
        self.vega_history = [self.portfolio_vega]

        return self.state

    @property
    def total_equity(self):
        """Total equity: MM + options"""
        option_value = sum(pos * price for pos, price in
                           zip(self.option_positions.values(), self.option_prices.values()))
        return self.mm_equity + option_value

    @property
    def state(self):
        """State representation for RL agent"""
        return np.array([
            self.S / 50000,  # Normalized price
            self.I / 10,  # Normalized inventory
            self.total_equity / 100000,  # Normalized equity
            self.local_vol,
            self.portfolio_delta / 10,  # Normalized delta
            self.portfolio_vega / 1000,  # Normalized vega
            self.time_step / self.max_steps,  # Progress
            np.random.normal(0, 0.01),  # Price change (simplified)
            np.random.normal(0, 0.001)  # Vol change (simplified)
        ])

    def _initialize_option_positions(self):
        """Initialize option positions to zero"""
        positions = {}
        for strike in self.strikes:
            for maturity in self.maturities:
                for opt_type in self.option_types:
                    key = f"{strike}_{maturity}_{opt_type}"
                    positions[key] = 0
        return positions

    def _compute_option_prices(self):
        """Compute option prices using Black-Scholes"""
        prices = {}
        for strike_ratio in self.strikes:
            for maturity in self.maturities:
                for opt_type in self.option_types:
                    K = strike_ratio * 50000  # Strike price
                    tau = maturity / 365  # Time to maturity

                    # Simplified Black-Scholes
                    price = self._black_scholes(self.S, K, tau, self.local_vol, opt_type)
                    key = f"{strike_ratio}_{maturity}_{opt_type}"
                    prices[key] = price
        return prices

    def _black_scholes(self, S, K, tau, sigma, option_type):
        """Simplified Black-Scholes pricing"""
        if tau <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if option_type == 'call':
            price = S * self._norm_cdf(d1) - K * self._norm_cdf(d2)
        else:
            price = K * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

        return max(price, 0)

    def _norm_cdf(self, x):
        """Normal CDF approximation"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def _compute_greeks(self):
        """Compute portfolio Greeks"""
        delta_total = self.I  # Start with MM delta
        vega_total = 0.0

        for key, position in self.option_positions.items():
            # Parse key to get strike, maturity, type
            parts = key.split('_')
            strike_ratio = float(parts[0])
            maturity = int(parts[1])
            opt_type = parts[2]

            K = strike_ratio * 50000
            tau = maturity / 365

            # Simplified delta and vega
            if opt_type == 'call':
                delta = 0.6 if self.S > K else 0.4  # Simplified
            else:
                delta = -0.4 if self.S > K else -0.6  # Simplified

            vega = 0.1 * self.S * np.sqrt(tau)  # Simplified vega

            delta_total += position * delta
            vega_total += position * vega

        self.portfolio_delta = delta_total
        self.portfolio_vega = vega_total

    # 5.5 Action Space
    @property
    def action_space(self):
        """Discrete action space: no trade + buy/sell for each option"""
        actions = ['no_trade']

        # Add buy/sell actions for each option
        for strike in self.strikes:
            for maturity in self.maturities:
                for opt_type in self.option_types:
                    actions.append(f"buy_{strike}_{maturity}_{opt_type}")
                    actions.append(f"sell_{strike}_{maturity}_{opt_type}")

        return actions

    def step(self, action):
        """Execute one environment step"""
        # Store previous equity for reward calculation
        prev_equity = self.total_equity

        # Update market state (simplified)
        self._update_market_state()

        # Execute action
        transaction_cost = self._execute_action(action)

        # Update Greeks
        self._compute_greeks()

        # Update MM equity
        self.mm_equity = self.cash + self.I * self.S

        # Compute reward
        reward = self._compute_reward(prev_equity, transaction_cost)

        # Update history
        self.time_step += 1
        self.equity_history.append(self.total_equity)
        self.delta_history.append(self.portfolio_delta)
        self.vega_history.append(self.portfolio_vega)

        # Check termination
        done = self.time_step >= self.max_steps

        return self.state, reward, done, {}

    def _update_market_state(self):
        """Update market state (simplified)"""
        # Random price movement
        price_change = np.random.normal(0, 0.01) * self.S
        self.S += price_change

        # Random volatility change
        vol_change = np.random.normal(0, 0.05)
        self.local_vol = max(0.3, min(1.0, self.local_vol + vol_change))

        # Update option prices
        self.option_prices = self._compute_option_prices()

    def _execute_action(self, action):
        """Execute trading action and return transaction cost"""
        if action == 'no_trade':
            return 0

        # Parse action
        parts = action.split('_')
        action_type, strike_ratio, maturity, opt_type = parts[0], float(parts[1]), int(parts[2]), parts[3]
        key = f"{strike_ratio}_{maturity}_{opt_type}"

        # Get option price
        option_price = self.option_prices[key]

        # Determine trade direction and size
        if action_type == 'buy':
            trade_size = self.lot_size
            self.option_positions[key] += trade_size
            self.cash -= trade_size * option_price
        else:  # sell
            trade_size = self.lot_size
            # Check if we have enough to sell
            if self.option_positions[key] >= trade_size:
                self.option_positions[key] -= trade_size
                self.cash += trade_size * option_price
            else:
                # If not enough, sell what we have
                trade_size = self.option_positions[key]
                self.option_positions[key] = 0
                self.cash += trade_size * option_price

        # Compute transaction cost
        notional = trade_size * option_price
        transaction_cost = self.transaction_cost * notional
        self.cash -= transaction_cost

        return transaction_cost

    # 5.7 Reward Design
    def _compute_reward(self, prev_equity, transaction_cost):
        """
        Reward function design:
        r = PnL - cost_penalty - risk_penalty - tail_risk_penalty
        """
        # 1. Profit focus: Realized PnL
        pnl = self.total_equity - prev_equity

        # 2. Cost awareness: Transaction cost penalty
        cost_penalty = transaction_cost * 10  # Scale for balance

        # 3. Risk exposure control: Soft penalties for large exposures
        delta_penalty = 0.01 * max(0, abs(self.portfolio_delta) - 5) ** 2
        vega_penalty = 0.001 * max(0, abs(self.portfolio_vega) - 500) ** 2

        # 4. Tail-risk awareness: Drawdown penalty
        current_drawdown = 0
        if len(self.equity_history) > 0:
            peak_equity = max(self.equity_history)
            if peak_equity > 0:
                current_drawdown = (peak_equity - self.total_equity) / peak_equity
        drawdown_penalty = 100 * max(0, current_drawdown - 0.1) ** 2

        # 5. Final outcome penalty for large losses
        final_penalty = 0
        if self.time_step == self.max_steps - 1:  # Last step
            if self.total_equity < 90000:  # Large loss threshold
                final_penalty = 50 * (100000 - self.total_equity) / 10000

        reward = pnl - cost_penalty - delta_penalty - vega_penalty - drawdown_penalty - final_penalty

        return reward


# Test the environment
if __name__ == "__main__":
    env = HedgingEnvironment()

    print("Hedging Environment Test")
    print(f"Initial state shape: {env.state.shape}")
    print(f"Action space size: {len(env.action_space)}")
    print(f"Initial total equity: {env.total_equity:.2f}")
    print(f"Initial portfolio delta: {env.portfolio_delta:.2f}")
    print(f"Initial portfolio vega: {env.portfolio_vega:.2f}")

    # Test a few steps
    print("\nTesting environment steps:")
    for i in range(3):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        print(f"Step {i + 1}: Action={action}, Reward={reward:.2f}, Equity={env.total_equity:.2f}")

    print(f"\nFinal portfolio delta: {env.portfolio_delta:.2f}")
    print(f"Final portfolio vega: {env.portfolio_vega:.2f}")
    print(f"Final total equity: {env.total_equity:.2f}")

