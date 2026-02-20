# ORT Custom Op: `Conv_ReLU6`

This folder contains a minimal ONNX Runtime custom op library that implements `Conv_ReLU6` for CPU.

## What it supports
- Op name: `Conv_ReLU6`
- Domain: `ai.custom`
- Inputs: `X, W, B, clip_min, clip_max` (float32)
- Output: `Y` (float32)
- Layout: NCHW 2D conv only
- Attributes read from fused node:
  - `Conv0_pads`
  - `Conv0_strides`
  - `Conv0_dilations`
  - `Conv0_group`
  - `Conv0_auto_pad`

## Build (Windows)
Prerequisites:
- CMake >= 3.18
- Visual Studio C++ toolchain
- ONNX Runtime binary package (with headers + import libs)

Example:

```powershell
cd ort_custom_ops
cmake -S . -B build -DORT_ROOT="C:\path\to\onnxruntime-win-x64-1.xx.x"
cmake --build build --config Release
```

Output DLL:
- `ort_custom_ops/build/Release/ort_custom_ops.dll`

## Use from Python
```python
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library(r"D:\projects\AIInfra\onnx-tool\ort_custom_ops\build\Release\ort_custom_ops.dll")
sess = ort.InferenceSession("data/public/mobilenetv2-12/mobilenetv2_fused.onnx", so, providers=["CPUExecutionProvider"])
```

## Important
Fused nodes in the model must use:
- `op_type="Conv_ReLU6"`
- `domain="ai.custom"`

The notebook has been updated to fuse with `nodedomain='ai.custom'`.

