# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-29

### Added
- Initial project extraction and migration from caffe vendor
- CMakePresets.json with default/debug/developer presets
- Development scripts (dev.sh for Linux/WSL, dev.ps1 for Windows)
- FFI prefix consistency checker (check_ffi_prefix.py)
- Installation verification script (verify_install.py)
- Protobuf code generation script (gen_proto.py)
- Conda recipe packaging configuration
- Updated pyproject.toml with sdist configuration and project metadata
- Updated environment.yml for conda development
- BSD-2-Clause LICENSE
- Comprehensive README.md with WSL development guide
- Project structure:
  - include/caffe_ffi/ - C++ headers
  - src/caffe_ffi/ - C++ implementation
  - python/caffe_ffi/ - Python bindings
  - proto/caffe/proto/ - Protobuf definitions
  - tests/python/ - Python unit tests
  - cmake/ - CMake modules
