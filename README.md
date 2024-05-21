# onnx_builder
utility for write ONNX and edit

## Convert ONNX to code

```python
import onnx_builder
generator = onnx_builder.CodeGenerator()
generator.generate('/path/to/onnx', 'generated')
```

## Modify ONNX

```python
import onnx_builder
import onnx
builder = onnx_builder.builder.from_onnx("/path/to/onnx")
# e.g. update initializer value
builder.values['xxx'].value = new_numpy_array
model = builder.build() #
onnx.save(model, "updated.onnx")
```

## Write ONNX

```python
import numpy as np
import onnx
import onnx_builder

builder = onnx_builder.Builder()
x = builder.Input(shape=(3, 4, 5), dtype=np.float32, name='x')
y = builder.Input(shape=(5,), dtype=np.float32, name='y')
x = builder.Add(x, y)
result = builder.Add(x, y)
builder.Output(result, name='result')
model = builder.build()
onnx.save(model, 'written.onnx')
```
