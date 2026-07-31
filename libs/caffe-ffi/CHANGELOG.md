# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-07-31

### Added
- **InsertSplits graph transformation edge case tests** (`test_insert_splits.py`): 18 test cases covering:
  - Zero-consumer dead-end blobs, single-consumer no-split
  - In-place ReLU with multi-consumer (split named after last producer)
  - Loss weight as implicit consumer, chained splits (fan-out after fan-out)
  - Idempotence (explicit splits not duplicated), forward correctness with in-place+split
  - Multiple external inputs ordering, linear chain zero splits, double in-place chains
  - Mixed explicit Input layer + `param.input()` external inputs
  - Split→Concat→Split nesting (Inception-style topology)
  - Multiple independent splits with correct position verification
  - Empty network (zero layers) robustness
  - Input layer with 3+ consumers, loss_weight + multiple downstream consumers
  - Unknown bottom blob reference error path
- **P3-C Transformer test suite** (`test_p3c_transformer.py`): 13 test cases covering:
  - Positional encoding, scaled dot-product attention, multi-head projection
  - Transformer encoder block forward pass correctness
- **Diagnostic logging for core layers**: Detailed shape mismatch error messages to simplify debugging
- **InsertSplits Pass 2b detailed logging**: Before/after layer order logging for external input split movement
- **Documentation**:
  - `INSERT_SPLITS_GRAPH_TRANSFORM.md`: Complete InsertSplits algorithm reference (passes, naming, edge cases, debugging)
  - `TESTING_GUIDELINES.md`: Caffe-FFI test authoring guidelines (prototxt construction, float assertions, graph transform testing, anti-patterns)

### Fixed
- **InsertSplits external input split ordering**: When multiple `param.input()` sources needed splits, splits were inserted in reverse consumer order instead of input declaration order. Fixed by collecting all external input splits first and inserting them en masse at position 0.
- **Sigmoid saturation precision assertions**: Three tests used exact equality (`==0.0`, `<1.0` at saturated values) that failed due to IEEE754 float32 subnormal behavior:
  - `sigmoid(-88) ≈ 6e-39` (subnormal, not exactly 0) → threshold assertion `< 1e-37`
  - `sigmoid(x ≥ 17)` is exactly `1.0` in float32 → transition zone verification uses `x ≤ 14` for strict `< 1.0`
  - Added NaN/Inf guards to all saturation tests
- **pytest InsertSplits test compatibility**: Replaced EuclideanLoss layers with Concat layers in structural tests to avoid shape mismatch from missing label blobs.

### Verified
- InsertSplits naming convention fully aligns with native Caffe (verified against `caffex/src/caffe/util/insert_splits.cpp`)
- InsertSplits 18/18 edge case tests pass
- P3-C activation tests (including fixed Sigmoid assertions) all pass
- P3-C Transformer 13/13 tests pass
- Split→Concat→Split nested topology forward produces correct output shapes
- Mixed Input layer + param.input() splits verified at correct positions (data split at position 0, Input layer split immediately after Input layer)

## [1.1.0] - 2026-07-30

### Added
- Copy-on-Write (COW) mechanism for Blob tensor sharing:
  - `ShareData`/`ShareDiff`: O(1) zero-copy tensor sharing between Blobs (reference-counted)
  - `UnshareData`/`UnshareDiff`: Explicit deep copy to break sharing
  - `IsDataShared`/`IsDiffShared`: Query sharing state (returns true only when original owner with refcount > 1)
  - `DataRefCount`/`DiffRefCount`: Query tensor reference counts (returns 0 for undefined/empty tensors)
  - `mutable_data_tensor()`/`mutable_diff_tensor()`: Auto-trigger COW clone on first write to shared tensor
  - Environment variables `CAFFE_FFI_ENABLE_COW` and `CAFFE_FFI_ENABLE_COW_PHASE3` for runtime control
  - `data_shared_`/`diff_shared_` flags to accurately track sharing state independent of reference count
- Memory lifecycle tracking tools (`caffe_ffi.tools.memory`): `BlobRef`, `tracked_blob`, `blob_snapshot`, `mem_check`
- Memory stress test suite: `test_create_destroy_loop_no_leak`, `test_reshape_loop_no_leak`
- Comprehensive COW test suite (test_cow.py): 21 test cases covering API, split topology, snapshot, and refcount behaviors

### Fixed
- **Memory leak in `_tensor_to_numpy`**: Reference cycle caused by attaching `_blob_ref` to `ctypes.cast()` LP_c_float pointer; fixed by attaching to numpy's internal ctypes array object (`arr.base.obj`) instead, enabling immediate refcount-based cleanup without GC
- **COW invalidation on Reshape**: `Reshape()` no longer unconditionally clears sharing flags; flags only clear when shape actually changes and new tensors are allocated (fixes in-place ReLU COW failure)
- **Incorrect `IsDataShared()` for shared Blobs**: Now returns `data_shared_ && use_count > 1` to correctly identify the original owner vs shared copies
- **Tensor item assignment error**: `mutable_data_tensor()`/`mutable_diff_tensor()` return numpy arrays (via ctypes zero-copy) instead of raw TVM Tensor objects, supporting `arr[i,j] = val` syntax
- **`DataRefCount()` returns 1 for empty Blob**: Now returns 0 for undefined/zero-element tensors

### Verified
- Docker Linux Python 3.14.6: 561/562 tests passed (1 skipped), 0 failures
- test_create_destroy_loop_no_leak: 500 create/fill/destroy cycles with zero memory leak
- COW tests: All 21 test cases pass including split/in-place/forward snapshot scenarios

## [0.1.0] - 2026-07-29

### Added
- Initial project extraction and migration from `projects/xuanspace/vendor/caffe/caffe-ffi` (matured vendor code elevated to independent library)
- CMake atomization refactoring (second round): 9 modular CMake files with zero code duplication
  - DetectBLAS.cmake, CompilerConfig.cmake, WindowsDllCopy.cmake (8 fine-grained DLL copy functions)
  - Tests.cmake reduced from 123 lines to 21 lines (83% reduction)
  - Added CAFFE_FFI_BUILD_TESTS option for conditional test compilation
  - Linux symbol visibility configuration aligned with MSVC behavior
- C++ test harness with per-test execution timing, per-suite summary, and Top 5 slowest tests report
- Python unit tests (test_python_api.py) with 65 test cases mirroring C++ tests, including timing statistics
- CAFFE_FFI_DISABLE_BACKTRACE environment variable for Python unittest compatibility
- Docker development environment at `apps/caffe-ffi-jupyter`:
  - Based on `jupyter-ssh-base` image (retains SSH + Jupyter dual services)
  - Python 3.14+ Miniconda environment
  - One-click deployment scripts (wsl-deploy.sh / deploy.ps1)
  - Integrated test-cpp-tests.sh for C++/Python unit test verification
- CMakePresets.json with default/debug/developer presets
- Development scripts (dev.sh for Linux/WSL, dev.ps1 for Windows)
- FFI prefix consistency checker (check_ffi_prefix.py)
- Installation verification script (verify_install.py)
- Protobuf code generation script (gen_proto.py)
- Conda recipe packaging configuration
- Updated pyproject.toml with sdist configuration, requires-python=">=3.14", and project metadata
- Updated environment.yml for conda development (Python 3.14)
- BSD-2-Clause LICENSE
- Comprehensive README.md with WSL development guide and Docker environment instructions
- Project AGENTS.md with AI agent routing and development conventions
- Project structure:
  - include/caffe_ffi/ - C++ headers (blob, net, layer, 20+ layer implementations)
  - src/caffe_ffi/ - C++ implementation
  - python/caffe_ffi/ - Python bindings with zero-copy Tensor API
  - proto/caffe/proto/ - Protobuf definitions
  - tests/python/ - Python unit tests (65 tests, 7 test suites)
  - tests/cpp/ - C++ unit tests (40 tests, 2 test suites)
  - cmake/ - CMake modules (9 modular files + README)
  - docs/ - Performance reports, memory logging documentation
  - examples/ - Usage examples (benchmark, MLP creation, zero-copy demo)

### Verified
- Docker Linux Python 3.14.6 environment: C++ 40/40 tests passed, Python 65/65 tests passed
- Editable installation (pip install -e .) works correctly
- C++/Python timing statistics output aligned (Per-suite summary + Top 5 slowest format)
