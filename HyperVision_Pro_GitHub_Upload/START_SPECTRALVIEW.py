"""Dependency-aware SpectralView launcher."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "spectral": "spectral",
    "PyQt5": "PyQt5",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
}


def main() -> int:
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    if missing:
        print("SpectralView cannot start because these packages are missing:")
        for package in missing:
            print(f"  - {package}")
        print("\nRun INSTALL_DEPENDENCIES.bat, then start again.")
        return 2
    script = Path(__file__).with_name("spectral_gui.py")
    return subprocess.call([sys.executable, str(script), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
