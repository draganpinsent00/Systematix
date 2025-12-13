import numpy as np
from history_utils import make_history_entry


def test_history_entry_shape_and_types():
    paths = np.zeros((10, 5))
    entry = make_history_entry('gbm', {'S0': 100}, 1.23, 0.01, paths)
    assert 'timestamp' in entry
    assert entry['model'] == 'gbm'
    assert isinstance(entry['params'], dict)
    assert entry['price_disc'] == 1.23
    assert entry['stderr_disc'] == 0.01
    assert entry['paths'].shape == (10, 5)

