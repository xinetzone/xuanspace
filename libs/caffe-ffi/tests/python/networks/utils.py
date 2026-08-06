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


def _download_model(url, filename, cache_dir, expected_size=None):
    """Download a model file with integrity validation.

    If a cached file exists but is smaller than the server-reported
    Content-Length (or an explicit expected_size), it is considered
    a truncated download from a previous run and is re-downloaded.
    """
    import time
    local_path = os.path.join(cache_dir, filename)

    # Probe server for expected size to detect truncated downloads.
    if expected_size is None:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as resp:
                cl = resp.headers.get("Content-Length")
                if cl:
                    expected_size = int(cl)
        except Exception:
            pass

    # Check if existing file is valid (exists and size matches expected).
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if expected_size is None or file_size == expected_size:
            return local_path
        # File exists but is truncated — remove and re-download.
        logging.warning(
            "Cached model %s is truncated (%d bytes < expected %d); re-downloading",
            filename, file_size, expected_size,
        )
        os.remove(local_path)

    # Download with retries.
    max_retries = 3
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, local_path)
            file_size = os.path.getsize(local_path)
            if expected_size is not None and file_size < expected_size:
                if attempt < max_retries - 1:
                    logging.warning(
                        "Download attempt %d for %s truncated (%d/%d); retrying...",
                        attempt + 1, filename, file_size, expected_size,
                    )
                    os.remove(local_path)
                    time.sleep(2 ** attempt)
                    continue
                raise IOError(
                    f"Failed to download {filename} after {max_retries} attempts: "
                    f"got {file_size} bytes, expected {expected_size}"
                )
            return local_path
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning("Download attempt %d for %s failed: %s; retrying...",
                                attempt + 1, filename, e)
                time.sleep(2 ** attempt)
            else:
                raise


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