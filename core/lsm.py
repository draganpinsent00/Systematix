"""
Longstaff-Schwartz method for American options.

Implements backward induction with polynomial regression to estimate continuation values.
"""

import numpy as np
from typing import Callable, Tuple


class LongstaffSchwartz:
    """Longstaff-Schwartz least-squares Monte Carlo for American options."""

    def __init__(self, degree: int = 2, risk_free_rate: float = 0.05):
        """
        Initialize LSM.

        Args:
            degree: Polynomial degree for regression (2 or 3)
            risk_free_rate: Risk-free rate for discounting
        """
        self.degree = degree
        self.risk_free_rate = risk_free_rate

    def price(
        self,
        paths: np.ndarray,
        intrinsic_payoff: Callable[[np.ndarray], np.ndarray],
        dt: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Price American option using Longstaff-Schwartz LSM.

        Algorithm:
        1. Compute intrinsic payoff at each time step
        2. Backward induction from T-1 down to 1:
           - For each timestep t:
             - Identify in-the-money paths
             - Regress discounted future cashflows against spot prices (polynomial basis)
             - Compute estimated continuation value
             - Compare: exercise if intrinsic > continuation, else hold
        3. Return mean optimal cashflow discounted to present

        Args:
            paths: Simulated paths (num_paths, num_timesteps + 1), S at each time
            intrinsic_payoff: Function intrinsic_payoff(S) -> payoff (works on 1D spot array)
            dt: Time step size

        Returns:
            (option_price, discounted_cashflows)
        """
        num_paths = paths.shape[0]
        num_steps = paths.shape[1] - 1

        # Discount factor per single step
        discount = np.exp(-self.risk_free_rate * dt)

        # Compute intrinsic payoff at each timestep for all paths
        # intrinsic[t, :] = intrinsic value at time t for each path
        intrinsic = np.zeros((num_steps + 1, num_paths))
        for t in range(num_steps + 1):
            intrinsic[t, :] = intrinsic_payoff(paths[:, t])

        # Initialize: if we hold to maturity, we get maturity payoff
        # cashflow[i] = value for path i at exercise time
        cashflow = np.copy(intrinsic[-1, :])  # Payoff at maturity T

        # Backward induction: from T-1 down to 1
        # At each step, decide to exercise or continue
        for t in range(num_steps - 1, 0, -1):
            # Spot price at current time step
            S_t = paths[:, t]

            # Intrinsic value (payoff if we exercise now)
            payoff_t = intrinsic[t, :]

            # Value of continuing: discount future cashflow back one step
            continuation_value_undiscounted = cashflow  # Cashflow from future exercise
            continuation_value_discounted = discount * continuation_value_undiscounted

            # In-the-money (ITM) paths: where intrinsic > 0
            itm_mask = payoff_t > 0

            # Need at least degree+1 points to fit polynomial
            num_itm = np.sum(itm_mask)
            if num_itm < self.degree + 1:
                # Not enough ITM paths - can't fit regression
                # Just use max of exercise vs continuation
                cashflow = np.where(
                    itm_mask,
                    np.maximum(payoff_t, continuation_value_discounted),
                    continuation_value_discounted
                )
                continue

            # Build polynomial basis for ITM paths only
            # X[i, j] = (S_t[i])^j for ITM paths
            X_itm = self._build_basis(S_t[itm_mask])

            # Right-hand side: discounted continuation values for ITM paths
            y_itm = continuation_value_discounted[itm_mask]

            # Regression: fit polynomial to continuation values
            # Solves: X @ coeff = y  (in least squares sense)
            try:
                coeffs = np.linalg.lstsq(X_itm, y_itm, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Singular matrix - use fallback (constant: just the mean)
                coeffs = np.zeros(X_itm.shape[1])
                coeffs[0] = np.mean(y_itm)

            # Evaluate fitted continuation value at ALL paths (ITM and OTM)
            X_all = self._build_basis(S_t)
            continuation_fit = X_all @ coeffs
            continuation_fit = np.maximum(continuation_fit, 0)  # Ensure non-negative

            # Early exercise decision:
            # Exercise if immediate payoff > estimated continuation value
            exercise_now = payoff_t > continuation_fit

            # Update cashflow: either exercise now or continue holding
            cashflow = np.where(exercise_now, payoff_t, continuation_value_discounted)

        # At this point, cashflow contains the value at time t=1
        # We need to discount back to t=0 one more step
        # The option price at t=0 is the discounted value of the cashflow at t=1
        option_values = discount * cashflow
        option_price = np.mean(option_values)

        return option_price, option_values

    def _build_basis(self, S: np.ndarray) -> np.ndarray:
        """
        Build polynomial basis functions for regression.

        Args:
            S: Spot prices (1D array)

        Returns:
            X: Design matrix of shape (len(S), degree+1) with columns [1, S, S^2, ...]
        """
        X = np.ones((len(S), self.degree + 1))
        for d in range(1, self.degree + 1):
            X[:, d] = S ** d
        return X

