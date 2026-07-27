"""DEMO FFI API module."""
from __future__ import annotations

from . import math
from ._ffi_api import *  # noqa: F401,F403

__all__ = ["math"]
