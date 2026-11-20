"""包级契约：版本号与公开 API。"""

import xuan_cmd


def test_version_format():
    parts = xuan_cmd.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_public_api_exported():
    for name in xuan_cmd.__all__:
        assert hasattr(xuan_cmd, name)
