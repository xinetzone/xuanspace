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

"""
Permute layer tests.

IMPORTANT (Pattern C5: framework parameter semantics verification):
The Permute layer is NOT available in this Caffe build (caffex/BVLC fork).
- caffe.proto has no `permute_param` message definition
- No PermuteLayer class is registered in the layer factory
- L.Permute will raise AttributeError

This is a known gap. The reference numpy implementation is preserved below
as documentation for when/if Permute support is added. When enabling, follow
the 5 patterns from retrospective-caffe-ops-correctness-test-20260727:
- C1: dockerignore whitelist (build scripts included)
- C2: assert_op_correct handles list/tuple outputs
- C3: use @pytest.mark.correctness (positive marker selection)
- C4: dict params serialized in filename generation
- C5: verify all parameter semantics before writing tests
"""

import logging
import numpy as np
import pytest

logger = logging.getLogger(__name__)

_PERMUTE_SKIP_REASON = (
    "Permute layer not compiled in this Caffe build "
    "(no permute_param in caffe.proto, no PermuteLayer class registered). "
    "Reference numpy implementation preserved for future enablement."
)


def _permute_ref(data, order):
    """Numpy reference implementation for Permute (transpose).

    Permute reorders dimensions according to the order list.
    Equivalent to np.transpose(data, order).

    Args:
        data: Input numpy array
        order: List of dimension indices, e.g. [0,2,3,1] for NCHW->NHWC

    Returns:
        Transposed numpy array
    """
    return np.transpose(data, order).astype(np.float32)


@pytest.mark.skip(reason=_PERMUTE_SKIP_REASON)
class TestPermuteNotAvailable:
    """All Permute tests skipped - layer not available in this build."""

    def test_permute_forward(self, caffe_test_dir):
        """Permute forward tests - SKIPPED (layer unavailable)."""
        pass

    @pytest.mark.correctness
    def test_permute_correctness(self, caffe_test_dir):
        """Permute correctness test with numpy transpose reference - SKIPPED."""
        np.random.seed(42)

        test_cases = [
            ((2, 3, 4), [0, 1, 2], "identity-3d"),
            ((2, 3, 4), [0, 2, 1], "swap-hw-3d"),
            ((2, 3, 4), [2, 1, 0], "reverse-3d"),
            ((2, 3, 4, 5), [0, 2, 3, 1], "nchw-to-nhwc-4d"),
            ((4, 5), [1, 0], "2d-transpose"),
        ]
        for shape, order, desc in test_cases:
            x = np.random.randn(*shape).astype(np.float32)
            ref = _permute_ref(x, order)
            assert ref.shape == tuple(shape[i] for i in order), (
                f"Ref impl shape wrong for {desc}: {ref.shape}"
            )

    @pytest.mark.edge
    def test_permute_edge_cases(self, caffe_test_dir):
        """Permute edge cases - SKIPPED."""
        pass
