"""Quick validation of fixed imports."""
import sys

try:
    print("Testing GBM import...")
    from models.gbm import GBM
    print("✓ GBM imports successfully")
except Exception as e:
    print(f"✗ GBM failed: {e}")
    sys.exit(1)

try:
    print("Testing Heston import...")
    from models.heston import Heston
    print("✓ Heston imports successfully")
except Exception as e:
    print(f"✗ Heston failed: {e}")
    sys.exit(1)

try:
    print("Testing Heston 3/2 import...")
    from models.heston_3_2 import Heston32
    print("✓ Heston 3/2 imports successfully")
except Exception as e:
    print(f"✗ Heston 3/2 failed: {e}")
    sys.exit(1)

try:
    print("Testing Sobol import...")
    from models.sobol_wrapper import generate_sobol_normals
    print("✓ Sobol wrapper imports successfully")
except Exception as e:
    print(f"✗ Sobol wrapper failed: {e}")
    sys.exit(1)

try:
    print("Testing Merton Jump import...")
    from models.merton_jump import MertonJump
    print("✓ Merton Jump imports successfully")
except Exception as e:
    print(f"✗ Merton Jump failed: {e}")
    sys.exit(1)

try:
    print("Testing Kou Jump import...")
    from models.kou_jump import KouJump
    print("✓ Kou Jump imports successfully")
except Exception as e:
    print(f"✗ Kou Jump failed: {e}")
    sys.exit(1)

try:
    print("Testing SABR import...")
    from models.sabr import SABR
    print("✓ SABR imports successfully")
except Exception as e:
    print(f"✗ SABR failed: {e}")
    sys.exit(1)

try:
    print("Testing Multi-Asset import...")
    from models.multi_asset import MultiAssetGBM
    print("✓ Multi-Asset imports successfully")
except Exception as e:
    print(f"✗ Multi-Asset failed: {e}")
    sys.exit(1)

print("\n✓ ALL MODEL IMPORTS SUCCESSFUL!")

