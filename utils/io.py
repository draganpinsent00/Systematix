"""
I/O utilities for data import/export.
"""

import json
import csv
from typing import Dict, Any, List
import numpy as np


def export_config_json(config: Dict[str, Any], filename: str = "pricing_config.json"):
    """Export configuration to JSON."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def import_config_json(filename: str) -> Dict[str, Any]:
    """Import configuration from JSON."""
    with open(filename, 'r') as f:
        return json.load(f)


def export_results_csv(results: Dict[str, Any], filename: str = "pricing_results.csv"):
    """Export results to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in results.items():
            if isinstance(value, (int, float)):
                writer.writerow([key, value])


def export_paths_csv(paths: np.ndarray, filename: str = "simulation_paths.csv"):
    """Export simulated paths to CSV."""
    np.savetxt(filename, paths, delimiter=',')


def import_paths_csv(filename: str) -> np.ndarray:
    """Import paths from CSV."""
    return np.loadtxt(filename, delimiter=',')

