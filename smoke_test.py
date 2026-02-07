"""
Systematic validation and smoke test for Systematix platform.

Run this to verify the complete system is operational:
    python smoke_test.py
"""

import sys
import traceback
from pathlib import Path

# Test results
TESTS = {}


def test_module(module_path: str, module_name: str) -> bool:
    """Test if a module can be imported."""
    try:
        __import__(module_path)
        TESTS[module_name] = ("PASS", None)
        return True
    except Exception as e:
        TESTS[module_name] = ("FAIL", str(e))
        return False


def main():
    print("\n" + "=" * 70)
    print("SYSTEMATIX - SYSTEM VALIDATION & SMOKE TEST")
    print("=" * 70 + "\n")

    # Test core modules
    print("Testing Core Modules...")
    core_modules = [
        ("config.settings", "Configuration"),
        ("config.schemas", "Schemas & Registry"),
        ("core.rng_engines", "RNG Engines"),
        ("core.rng_distributions", "Distribution Transforms"),
        ("core.brownian", "Brownian Motion"),
        ("core.mc_engine", "MC Engine"),
        ("core.variance_reduction", "Variance Reduction"),
        ("core.lsm", "Longstaff-Schwartz"),
        ("core.numerics", "Numerical Utilities"),
    ]

    for module_path, module_name in core_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Test models
    print("\nTesting Stochastic Models...")
    model_modules = [
        ("models.base", "Base Model"),
        ("models.gbm", "GBM"),
        ("models.heston", "Heston"),
        ("models.heston_3_2", "3/2 Heston"),
        ("models.merton_jump", "Merton Jump"),
        ("models.kou_jump", "Kou Jump"),
        ("models.sabr", "SABR"),
        ("models.multi_asset", "Multi-Asset"),
        ("models.sobol_wrapper", "Sobol Wrapper"),
    ]

    for module_path, module_name in model_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Test instruments
    print("\nTesting Instruments (50+ Option Types)...")
    instrument_modules = [
        ("instruments.base", "Base Instrument"),
        ("instruments.registry", "Instrument Registry"),
        ("instruments.payoffs_vanilla", "Vanilla Payoffs"),
        ("instruments.payoffs_exotic", "Exotic Payoffs"),
        ("instruments.payoffs_rates_fx", "Multi-Asset Payoffs"),
        ("instruments.custom_payoff", "Custom Payoff"),
    ]

    for module_path, module_name in instrument_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Test analytics
    print("\nTesting Analytics...")
    analytics_modules = [
        ("analytics.pricing", "Pricing"),
        ("analytics.greeks", "Greeks"),
        ("analytics.risk", "Risk Analysis"),
        ("analytics.diagnostics", "Diagnostics"),
        ("analytics.calibration", "Calibration"),
    ]

    for module_path, module_name in analytics_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Test visualization
    print("\nTesting Visualization...")
    viz_modules = [
        ("visualization.plotly_theme", "Plotly Theme"),
        ("visualization.charts_paths", "Path Charts"),
        ("visualization.charts_payoffs", "Payoff Charts"),
        ("visualization.charts_diagnostics", "Diagnostic Charts"),
    ]

    for module_path, module_name in viz_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Test UI
    print("\nTesting UI Components...")
    ui_modules = [
        ("ui.layout", "Layout"),
        ("ui.components", "Components"),
        ("ui.dynamic_forms", "Dynamic Forms"),
        ("ui.state", "State Management"),
    ]

    for module_path, module_name in ui_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Test utilities
    print("\nTesting Utilities...")
    util_modules = [
        ("utils.validation", "Validation"),
        ("utils.io", "I/O"),
        ("utils.logging", "Logging"),
    ]

    for module_path, module_name in util_modules:
        status = "✓" if test_module(module_path, module_name) else "✗"
        print(f"  {status} {module_name}")

    # Summary
    passed = sum(1 for status, _ in TESTS.values() if status == "PASS")
    failed = sum(1 for status, _ in TESTS.values() if status == "FAIL")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TESTS)} modules")
    print("=" * 70)

    if failed == 0:
        print("\n✅ ALL MODULES VALIDATED SUCCESSFULLY!")
        print("\nTo run the Streamlit dashboard:")
        print("  streamlit run app.py")
        print("\nTo run integration tests:")
        print("  python test_integration.py")
        return 0
    else:
        print("\n❌ SOME MODULES FAILED:")
        for name, (status, error) in TESTS.items():
            if status == "FAIL":
                print(f"\n  {name}:")
                print(f"    {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

