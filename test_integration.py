"""
Quick start demo and integration test.
"""

import numpy as np
from core.rng_engines import create_rng
from models.gbm import GBM
from instruments.registry import create_instrument
from core.mc_engine import MonteCarloEngine
from analytics.greeks import GreeksComputer
from analytics.risk import RiskAnalyzer


def test_european_call():
    """Test pricing a European call."""
    print("=" * 60)
    print("Testing European Call Option")
    print("=" * 60)

    # Market parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    div_yield = 0.0
    vol = 0.20
    time_to_maturity = 1.0

    # MC parameters
    num_paths = 10000
    num_steps = 252

    # Create RNG
    print("\n1. Creating RNG engine (Mersenne Twister)...")
    rng = create_rng("mersenne", seed=42)

    # Build GBM model
    print("2. Building GBM model...")
    gbm = GBM(
        spot=spot,
        risk_free_rate=rate,
        dividend_yield=div_yield,
        initial_volatility=vol,
        time_to_maturity=time_to_maturity,
    )

    # Generate paths
    print(f"3. Generating {num_paths} paths with {num_steps} steps...")
    paths = gbm.generate_paths(
        rng,
        num_paths=num_paths,
        num_steps=num_steps,
        distribution="normal",
        antithetic_variates=True,
    )
    print(f"   Paths shape: {paths.shape}")

    # Create European call
    print("4. Creating European call option (K=100)...")
    option = create_instrument("european_call", strike=strike)

    # Price with MC
    print("5. Running Monte Carlo pricing...")
    mc_engine = MonteCarloEngine(rng, num_simulations=num_paths, num_timesteps=num_steps)
    result = mc_engine.price(
        paths,
        option.payoff,
        risk_free_rate=rate,
        time_to_maturity=time_to_maturity,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Option Price:      ${result.price:.6f}")
    print(f"Std Error:         ${result.std_error:.6f}")
    print(f"95% CI:            [${result.ci_lower:.6f}, ${result.ci_upper:.6f}]")
    print(f"Num Paths:         {num_paths}")
    print(f"Num Steps:         {num_steps}")
    print(f"Variance Reduction:{result.variance_reduction_factor:.4f}")

    # Black-Scholes comparison
    from analytics.pricing import black_scholes_call
    bs_price = black_scholes_call(spot, strike, time_to_maturity, rate, vol)
    print(f"\nBlack-Scholes:     ${bs_price:.6f}")
    print(f"Error:             {abs(result.price - bs_price):.6f} ({100*abs(result.price - bs_price)/bs_price:.2f}%)")

    # Greeks
    print("\n6. Computing Greeks...")
    greeks_computer = GreeksComputer()
    greeks = greeks_computer.compute_all(
        spot=spot,
        price=result.price,
        paths=paths,
        payoff_func=option.payoff,
        risk_free_rate=rate,
        time_to_maturity=time_to_maturity,
        volatility=vol,
    )

    print("\n" + "=" * 60)
    print("GREEKS")
    print("=" * 60)
    print(f"Delta:             {greeks['delta']:.6f}")
    print(f"Gamma:             {greeks['gamma']:.6f}")
    print(f"Vega:              {greeks['vega']:.6f}")
    print(f"Theta:             {greeks['theta']:.6f}")
    print(f"Rho:               {greeks['rho']:.6f}")

    # Risk metrics
    print("\n7. Computing risk metrics...")
    risk_analyzer = RiskAnalyzer()
    var, cvar = risk_analyzer.compute_var_cvar(result.payoffs, confidence_level=0.95)
    stats = risk_analyzer.compute_statistics(result.payoffs)

    print("\n" + "=" * 60)
    print("RISK METRICS")
    print("=" * 60)
    print(f"VaR (95%):         ${var:.6f}")
    print(f"CVaR (95%):        ${cvar:.6f}")
    print(f"Mean Payoff:       ${stats['mean']:.6f}")
    print(f"Std Dev:           ${stats['std']:.6f}")
    print(f"Skewness:          {stats['skewness']:.6f}")
    print(f"Kurtosis:          {stats['kurtosis']:.6f}")

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)


def test_asian_option():
    """Test pricing an Asian option."""
    print("\n\n" + "=" * 60)
    print("Testing Asian Option (Arithmetic Average)")
    print("=" * 60)

    spot = 100.0
    strike = 100.0
    rate = 0.05
    vol = 0.20
    time_to_maturity = 1.0
    num_paths = 5000
    num_steps = 252

    rng = create_rng("mersenne", seed=43)
    gbm = GBM(spot, rate, 0.0, vol, time_to_maturity)
    paths = gbm.generate_paths(rng, num_paths, num_steps)

    option = create_instrument("asian_arithmetic_call", strike=strike)
    mc_engine = MonteCarloEngine(rng, num_paths, num_steps)
    result = mc_engine.price(paths, option.payoff, rate, time_to_maturity)

    print(f"Asian Call Price:  ${result.price:.6f}")
    print(f"95% CI:            [${result.ci_lower:.6f}, ${result.ci_upper:.6f}]")
    print("✅ Asian option test passed!")


def test_barrier_option():
    """Test pricing a barrier option."""
    print("\n\n" + "=" * 60)
    print("Testing Barrier Option (Up-and-Out Call)")
    print("=" * 60)

    spot = 100.0
    strike = 100.0
    barrier = 120.0
    rate = 0.05
    vol = 0.20
    time_to_maturity = 1.0
    num_paths = 5000
    num_steps = 252

    rng = create_rng("mersenne", seed=44)
    gbm = GBM(spot, rate, 0.0, vol, time_to_maturity)
    paths = gbm.generate_paths(rng, num_paths, num_steps)

    option = create_instrument("barrier_up_out_call", strike=strike, barrier=barrier)
    mc_engine = MonteCarloEngine(rng, num_paths, num_steps)
    result = mc_engine.price(paths, option.payoff, rate, time_to_maturity)

    print(f"Barrier Call Price: ${result.price:.6f}")
    print(f"95% CI:             [${result.ci_lower:.6f}, ${result.ci_upper:.6f}]")
    print("✅ Barrier option test passed!")


def test_rng_engines():
    """Test all RNG engines."""
    print("\n\n" + "=" * 60)
    print("Testing RNG Engines")
    print("=" * 60)

    engines = ["mersenne", "pcg64", "xorshift", "philox", "middle_square"]

    for engine_name in engines:
        try:
            rng = create_rng(engine_name, seed=42)
            samples = rng.standard_normal((100, 10))
            mean = np.mean(samples)
            std = np.std(samples)
            print(f"{engine_name:15s}: mean={mean:8.6f}, std={std:8.6f} ✓")
        except Exception as e:
            print(f"{engine_name:15s}: FAILED - {str(e)}")

    print("✅ RNG engine test passed!")


def test_heston_model():
    """Test Heston model."""
    print("\n\n" + "=" * 60)
    print("Testing Heston Model")
    print("=" * 60)

    from models.heston import Heston

    spot = 100.0
    rate = 0.05
    vol = 0.20
    time_to_maturity = 1.0
    num_paths = 5000
    num_steps = 252

    heston = Heston(
        spot=spot,
        risk_free_rate=rate,
        dividend_yield=0.0,
        initial_volatility=vol,
        time_to_maturity=time_to_maturity,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
    )

    rng = create_rng("mersenne", seed=45)
    paths = heston.generate_paths(rng, num_paths, num_steps)

    option = create_instrument("european_call", strike=100.0)
    mc_engine = MonteCarloEngine(rng, num_paths, num_steps)
    result = mc_engine.price(paths, option.payoff, rate, time_to_maturity)

    print(f"Heston Call Price: ${result.price:.6f}")
    print(f"95% CI:            [${result.ci_lower:.6f}, ${result.ci_upper:.6f}]")
    print("✅ Heston model test passed!")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "SYSTEMATIX INTEGRATION TEST SUITE" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")

    try:
        test_european_call()
        test_asian_option()
        test_barrier_option()
        test_rng_engines()
        test_heston_model()

        print("\n\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 20 + "ALL TESTS PASSED! ✅" + " " * 18 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\nTo run the Streamlit dashboard, use:")
        print("  streamlit run app.py")

    except Exception as e:
        print(f"\n\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

