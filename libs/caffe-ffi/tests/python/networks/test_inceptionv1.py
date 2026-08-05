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


def _test_inceptionv1(data, model_dir):
    """One iteration of InceptionV1 (GoogLeNet)

    Note: Source file comment mistakenly says Inceptionv4,
    but the actual model used is BVLC GoogleNet (InceptionV1).
    """
    if not _CAFFE_FFI_AVAILABLE:
        pytest.skip("caffe_ffi C++ extension is not available; network tests skipped")
    data_process = _preprocess_imagenet(data, scale=58.8)

    proto_file_url = (
        "https://github.com/BVLC/caffe/raw/master/models/bvlc_googlenet/deploy.prototxt"
    )
    blob_file_url = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel"
    proto_file = _download_model(proto_file_url, "inceptionv1.prototxt", model_dir)
    blob_file = _download_model(blob_file_url, "inceptionv1.caffemodel", model_dir)

    caffe_out = _test_network(data_process, proto_file, blob_file)

    assert len(caffe_out) > 0, "Caffe output should not be empty"
    for out in caffe_out:
        assert not np.any(np.isnan(out)), "Output contains NaN"
        assert not np.any(np.isinf(out)), "Output contains Inf"
    return caffe_out


@pytest.mark.slow
def test_forward_Inceptionv1(caffe_model_dir):
    """InceptionV1 (GoogLeNet)"""
    data = np.random.randint(0, 256, size=(1, 3, 224, 224)).astype(np.float32)
    _test_inceptionv1(data, caffe_model_dir)