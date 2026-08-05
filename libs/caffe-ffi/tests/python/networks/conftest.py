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

import os
import sys
import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# The migrated networks tests use flat imports (``from utils import ...``) while
# living inside a package directory (``networks/__init__.py``). pytest treats the
# directory as the ``networks`` package and does not put it on the flat import
# path, so explicitly add it to sys.path to make ``utils`` resolvable.
_networks_dir = str(Path(__file__).resolve().parent)
if _networks_dir not in sys.path:
    sys.path.insert(0, _networks_dir)


@pytest.fixture(scope="session")
def caffe_model_dir():
    model_dir = os.path.expanduser("~/.caffe_test_data/models")
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Using caffe model directory: {model_dir}")
    return str(model_dir)