#!/usr/bin/env python3
"""Generate Python protobuf bindings from vta_config.proto.

Usage:
    python scripts/gen_proto.py

This script regenerates python/npu_ffi/vta/vta_config_pb2.py from proto/vta_config.proto.
Requires grpcio-tools package (pip install grpcio-tools).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
OUTPUT_DIR = ROOT / "python" / "npu_ffi" / "vta"
PROTO_FILE = PROTO_DIR / "vta_config.proto"


def main() -> int:
    if not PROTO_FILE.exists():
        print(f"Error: proto file not found at {PROTO_FILE}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating Python bindings from {PROTO_FILE}...")

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUTPUT_DIR}",
        PROTO_FILE.name,
    ]

    print(f"Running: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"protoc failed with exit code {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    generated_file = OUTPUT_DIR / PROTO_FILE.name.replace(".proto", "_pb2.py")
    if not generated_file.exists():
        print(f"Error: expected generated file not found: {generated_file}", file=sys.stderr)
        return 1

    print(f"Generated: {generated_file}")

    try:
        sys.path.insert(0, str(ROOT / "python"))
        from npu_ffi.vta import vta_config_pb2
        c = vta_config_pb2.VTAConfig()
        c.target = "test"
        c.log_bus_width = 3
        data = c.SerializeToString()
        c2 = vta_config_pb2.VTAConfig()
        c2.ParseFromString(data)
        assert c2.target == "test"
        assert c2.log_bus_width == 3
        print("Verification: OK - serialization roundtrip successful")
    except Exception as e:
        print(f"Warning: verification failed: {e}", file=sys.stderr)
        return 1

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
