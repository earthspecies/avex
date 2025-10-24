#!/usr/bin/env python3
"""
Setup script for representation-learning package.

This script provides an easy way to install the package with all dependencies,
including the private esp-data package.
"""

import subprocess
import sys
from pathlib import Path


def install_with_private_index() -> None:
    """Install the package with the private esp-data index."""
    print("Installing representation-learning with private dependencies...")

    # Install with the private index
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".",  # Install in editable mode
        "--extra-index-url",
        "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Successfully installed representation-learning!")
        print("\nYou can now use:")
        print("  from representation_learning import load_model, list_models")
        print("  list-models  # CLI command")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)


def install_from_wheel() -> None:
    """Install from a built wheel."""
    wheel_path = Path("dist") / "representation_learning-1.0.0-py3-none-any.whl"

    if not wheel_path.exists():
        print("❌ Wheel not found. Please run 'python -m build' first.")
        sys.exit(1)

    print(f"Installing from wheel: {wheel_path}")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        str(wheel_path),
        "--extra-index-url",
        "https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Successfully installed representation-learning from wheel!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--wheel":
        install_from_wheel()
    else:
        install_with_private_index()
