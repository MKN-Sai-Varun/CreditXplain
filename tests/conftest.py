"""
Pytest configuration to add src/ to Python path.
"""
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs(*paths): skip unless these local-only files exist (data and trained models are not in git)",
    )


def pytest_runtest_setup(item):
    for mark in item.iter_markers("needs"):
        missing = [p for p in mark.args if not (ROOT / p).exists()]
        if missing:
            pytest.skip(f"needs {', '.join(missing)} (run scripts/run_pipeline.sh)")

