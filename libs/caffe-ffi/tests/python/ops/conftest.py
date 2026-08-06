# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import logging
import os

import pytest

logger = logging.getLogger(__name__)

# Use relative import to avoid sys.path conflicts with networks/utils.py
from . import utils as _utils_module  # noqa: F401

# Verify that the L class (used by all ops tests) is available
assert hasattr(_utils_module, "L"), "ops/utils.py does not define class L"
logger.info(f"ops conftest: utils loaded from {_utils_module.__file__} (has L={hasattr(_utils_module, 'L')})")


@pytest.fixture(scope="session")
def caffe_test_dir(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp("caffe_test_data")
    os.makedirs(test_dir, exist_ok=True)
    logger.info(f"Created caffe test directory: {test_dir}")
    return str(test_dir)


def pytest_collection_modifyitems(config, items):
    """Warn about test functions without any pytest marker (prevents silent omission)."""
    registered_markers = set()
    if config.getini("markers"):
        for line in config.getini("markers"):
            marker_name = line.split(":")[0].strip()
            registered_markers.add(marker_name)

    for item in items:
        if not hasattr(item, "own_markers") or not item.own_markers:
            logger.warning(
                f"Test function '{item.name}' in {item.fspath.basename} "
                f"has no pytest markers — it will be skipped by -m correctness/edge/slow filters. "
                f"Add @pytest.mark.correctness or appropriate marker."
            )
