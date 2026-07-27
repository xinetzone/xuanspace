#!/usr/bin/env python3
import sys
print("Testing imports...")
import caffe
print(f"  caffe version: {caffe.version()}")
print("  ✓ caffe imported successfully")
import pycaffe
print(f"  pycaffe version: {pycaffe.__version__}")
print("  ✓ pycaffe imported successfully")
print("\nAll imports passed!")
