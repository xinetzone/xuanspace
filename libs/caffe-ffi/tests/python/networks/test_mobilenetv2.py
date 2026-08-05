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

import numpy as np
import pytest
from utils import (
    _CAFFE_FFI_AVAILABLE,
    _download_model,
    _preprocess_imagenet,
    _test_network,
)


def _test_mobilenetv2(data, model_dir):
    if not _CAFFE_FFI_AVAILABLE:
        pytest.skip("caffe_ffi C++ extension is not available; network tests skipped")
    data_process = _preprocess_imagenet(data, scale=58.8)

    proto_file_url = (
        "https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2_deploy.prototxt"
    )
    blob_file_url = (
        "https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel?raw=true"
    )
    proto_file = _download_model(proto_file_url, "mobilenetv2.prototxt", model_dir)
    blob_file = _download_model(blob_file_url, "mobilenetv2.caffemodel", model_dir)

    caffe_out = _test_network(data_process, proto_file, blob_file)

    assert len(caffe_out) > 0, "Caffe output should not be empty"
    for out in caffe_out:
        assert not np.any(np.isnan(out)), "Output contains NaN"
        assert not np.any(np.isinf(out)), "Output contains Inf"
    return caffe_out


@pytest.mark.slow
def test_forward_Mobilenetv2(caffe_model_dir):
    """Mobilenetv2"""
    data = np.random.randint(0, 256, size=(1, 3, 224, 224)).astype(np.float32)
    _test_mobilenetv2(data, caffe_model_dir)