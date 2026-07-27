#!/usr/bin/env python3
import caffe
import os
caffe_dir = os.path.dirname(caffe.__file__)
print(f"CAFFE_DIR={caffe_dir}")
