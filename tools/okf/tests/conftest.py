"""pytest 共享 fixtures。

提供示例 OKF Bundle 路径与内存中构建 Bundle 的辅助函数。
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_BUNDLE = FIXTURES_DIR / "sample_bundle"


@pytest.fixture
def sample_bundle_path() -> Path:
    """返回磁盘上的示例 OKF Bundle 根目录路径。"""
    return SAMPLE_BUNDLE
