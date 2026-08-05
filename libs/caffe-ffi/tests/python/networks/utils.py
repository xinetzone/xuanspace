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
import logging
import urllib.request
import sys
from pathlib import Path

import numpy as np

# Ensure the caffe_ffi package is importable even when the networks/ directory is
# run directly (without the parent tests/python/conftest.py that normally inserts
# the package dir into sys.path).
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

import pytest  # noqa: E402
import caffe_ffi  # noqa: E402

os.environ["GLOG_minloglevel"] = "2"
logging.basicConfig(level=logging.ERROR)

# Whether the native caffe_ffi C++ extension is available. When it is not,
# network tests are skipped cleanly instead of failing on empty/wrong results.
_CAFFE_FFI_AVAILABLE = bool(caffe_ffi.is_available())


def _download_model(url, filename, cache_dir):
    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        return local_path
    urllib.request.urlretrieve(url, local_path)
    return local_path


def _preprocess_imagenet(data, mean_val=None, scale=1.0):
    if mean_val is None:
        mean_val = [103.939, 116.779, 123.68]
    mean = np.array(mean_val, dtype=np.float32)
    mean = mean.reshape((1, 3, 1, 1))
    mean = np.tile(mean, (1, 1, data.shape[2], data.shape[3]))
    data_process = data - mean
    if scale != 1.0:
        data_process = data_process / scale
    return data_process.astype(np.float32)


def _test_network(data, proto_file, blob_file):
    if not _CAFFE_FFI_AVAILABLE:
        pytest.skip("caffe_ffi C++ extension is not available; network tests skipped")
    net = caffe_ffi.read_net(proto_file, blob_file)
    net.blob_by_name("data").data = data
    out = net.forward()
    return list(out.values())