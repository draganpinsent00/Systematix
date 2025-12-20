#!/usr/bin/env python
"""
Systematix Installation & Quick Start Checklist
Run this for a guided setup and validation.
"""

import os
import subprocess
import sys
from pathlib import Path


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor} (OK)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requires 3.9+)")
        return False


def check_pip():
    """Check pip installation."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
        print("✅ pip is installed")
        return True
    except:
        print("❌ pip not found")
        return False


def install_requirements():
    """Install requirements.txt."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False

    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        print("✅ Dependencies installed")
        return True
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False


def run_validation():
    """Run smoke_test.py."""
    if not Path("smoke_test.py").exists():
        print("❌ smoke_test.py not found")
        return False

    print("Running module validation...")
    try:
        result = subprocess.run([sys.executable, "smoke_test.py"],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ All modules validated")
            return True
        else:
            print(f"❌ Validation failed:\n{result.stdout}\n{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def check_project_structure():
    """Verify project structure."""
    required_dirs = [
        "config", "core", "models", "instruments",
        "analytics", "visualization", "ui", "utils"
    ]

    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ missing")
            all_exist = False

    return all_exist


def check_documentation():
    """Check for documentation files."""
    docs = [
        "README.md",
        "QUICKSTART.md",
        "ARCHITECTURE.md",
        "INDEX.md",
        "PROJECT_SUMMARY.md",
    ]

    all_exist = True
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ {doc}")
        else:
            print(f"❌ {doc} missing")
            all_exist = False

    return all_exist


def main():
    """Run checklist."""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SYSTEMATIX INSTALLATION CHECKLIST" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    results = {}

    # Step 1: Check Python
    print_section("Step 1: Python Environment")
    results["python"] = check_python()

    # Step 2: Check pip
    print_section("Step 2: Package Manager")
    results["pip"] = check_pip()

    # Step 3: Project structure
    print_section("Step 3: Project Structure")
    results["structure"] = check_project_structure()

    # Step 4: Documentation
    print_section("Step 4: Documentation")
    results["docs"] = check_documentation()

    # Step 5: Install dependencies
    print_section("Step 5: Install Dependencies")
    if results["pip"]:
        results["install"] = install_requirements()
    else:
        results["install"] = False

    # Step 6: Validate modules
    print_section("Step 6: Module Validation")
    if results["install"]:
        results["validate"] = run_validation()
    else:
        results["validate"] = False

    # Summary
    print_section("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for step, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {step}")

    print(f"\n  Score: {passed}/{total}")

    if passed == total:
        print("\n" + "=" * 70)
        print("✅ READY TO USE!")
        print("\nNext steps:")
        print("  1. Read: INDEX.md")
        print("  2. Setup: See QUICKSTART.md")
        print("  3. Run: streamlit run app.py")
        print("  4. Open: http://localhost:8501")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ SETUP INCOMPLETE")
        print("\nTroubleshooting:")
        print("  • Python 3.9+ required: python --version")
        print("  • Install deps: pip install -r requirements.txt")
        print("  • Check docs: INDEX.md")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏸️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

