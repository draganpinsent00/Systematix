"""processes.py
Wrapper around existing `simulator.py` to centralize stochastic processes.
"""
from simulator import simulate_paths

# export simulate_paths directly for compatibility

__all__ = ['simulate_paths']

