from typing import Any, Dict
from datetime import datetime, timezone


def make_history_entry(model: str, params: Dict[str, Any], price_disc: float, stderr_disc: float, paths) -> Dict[str, Any]:
    """Create a history entry with a timezone-aware UTC timestamp."""
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': model,
        'params': params,
        'price_disc': float(price_disc),
        'stderr_disc': float(stderr_disc),
        'paths': paths
    }
