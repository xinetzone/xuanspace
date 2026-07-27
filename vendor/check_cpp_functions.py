#!/usr/bin/env python3
import caffe
mod = caffe._find_lib()
print("Functions available in _caffe module:")
for func_name in ['Net_NumLayers', 'Net_LayerNames', 'Net_LayerType', 
                  'Net_ParamLayerIndices', 'Param_GetShape', 'Param_GetData',
                  'Net_TopIds', 'Net_BottomIds']:
    has_func = hasattr(mod, func_name)
    print(f"  - {func_name}: {'✓' if has_func else '✗ MISSING'}")
