# onnx_builder
utility for write ONNX and edit

## Convert ONNX to code

```python
import onnx_builder
generator = onnx_builder.CodeGenerator('/path/to/outputdir')
# generator.from_test_case('/path/to/onnx_test_case')
generator.from_onnx('/path/to/onnx')
```
