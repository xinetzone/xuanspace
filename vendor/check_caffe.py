#!/usr/bin/env python3
import caffe
import os
print(f"caffe.__file__: {caffe.__file__}")
print(f"caffe.__path__: {getattr(caffe, '__path__', 'N/A')}")
print(f"Net class: {caffe.Net}")
print(f"Net class attributes with 'blob' or 'layer' or 'param':")
for attr in dir(caffe.Net):
    if any(x in attr.lower() for x in ['blob', 'layer', 'param', 'forward']):
        print(f"  - {attr}")
print(f"\nhasattr(Net, 'blobs'): {hasattr(caffe.Net, 'blobs')}")
print(f"hasattr(Net, 'layers'): {hasattr(caffe.Net, 'layers')}")
print(f"hasattr(Net, 'params'): {hasattr(caffe.Net, 'params')}")
print(f"hasattr(Net, 'layer_dict'): {hasattr(caffe.Net, 'layer_dict')}")

# Check __init__.py content
init_path = os.path.join(os.path.dirname(caffe.__file__), '__init__.py')
print(f"\nChecking __init__.py at: {init_path}")
with open(init_path, 'r') as f:
    content = f.read(5000)
print(f"First 5000 chars of __init__.py:")
print(content)
