# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
