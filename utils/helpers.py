def safe_mean(a):
    """Small helper to compute mean with empty-array guard."""
    import numpy as np
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    return float(a.mean())

