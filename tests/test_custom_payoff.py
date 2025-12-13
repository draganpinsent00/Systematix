import numpy as np
from payoff_utils import safe_compile_payoff


def test_safe_compile_and_run_simple():
    code = '''
def custom_payoff(paths):
    import numpy as np
    ST = paths[:, -1]
    return np.maximum(ST - 100, 0.0)
'''
    fn = safe_compile_payoff(code)
    paths = np.array([[100, 110], [100, 90]])
    res = fn(paths)
    assert res.shape == (2,)
    assert np.allclose(res, np.array([10.0, 0.0]))


def test_reject_imports():
    bad = 'import os\ndef custom_payoff(paths):\n    return paths[:, -1]'
    try:
        safe_compile_payoff(bad)
        assert False, 'Should have raised'
    except ValueError:
        pass


def test_reject_multiple_funcs():
    bad = 'def foo(x):\n    return x\ndef custom_payoff(paths):\n    return paths[:, -1]'
    try:
        safe_compile_payoff(bad)
        assert False
    except ValueError:
        pass

