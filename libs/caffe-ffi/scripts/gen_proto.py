#!/usr/bin/env python3
"""Generate Python protobuf bindings from caffe.proto.

Usage:
    python scripts/gen_proto.py

This script regenerates python/caffe_ffi/caffe_pb2.py from proto/caffe/proto/caffe.proto.
Requires grpcio-tools package (pip install grpcio-tools).
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "proto"
OUTPUT_DIR = ROOT / "python" / "caffe_ffi"
PROTO_FILE = PROTO_DIR / "caffe" / "proto" / "caffe.proto"


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
        str(PROTO_FILE.relative_to(PROTO_DIR)),
    ]

    print(f"Running: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"protoc failed with exit code {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    generated_file = OUTPUT_DIR / "caffe" / "proto" / "caffe_pb2.py"
    expected_file = OUTPUT_DIR / "caffe_pb2.py"
    
    if generated_file.exists():
        if expected_file.exists():
            expected_file.unlink()
        shutil.move(str(generated_file), str(expected_file))

        # Clean up the now-empty intermediate package dirs left by protoc
        # (caffe/proto/caffe) so the redundant compat package is not recreated.
        for d in (OUTPUT_DIR / "caffe" / "proto" / "caffe",
                  OUTPUT_DIR / "caffe" / "proto",
                  OUTPUT_DIR / "caffe"):
            if d.exists():
                try:
                    d.rmdir()
                except OSError:
                    pass

    if not expected_file.exists():
        print(f"Error: expected generated file not found: {expected_file}", file=sys.stderr)
        return 1

    print(f"Generated: {expected_file}")

    try:
        sys.path.insert(0, str(ROOT / "python"))
        from caffe_ffi import caffe_pb2
        net_param = caffe_pb2.NetParameter()
        net_param.name = "test"
        data = net_param.SerializeToString()
        net_param2 = caffe_pb2.NetParameter()
        net_param2.ParseFromString(data)
        assert net_param2.name == "test"
        print("Verification: OK - serialization roundtrip successful")
    except Exception as e:
        print(f"Warning: verification failed: {e}", file=sys.stderr)
        return 1

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
