import onnx
import onnx_builder
import numpy as np


def test_tensor_from_proto() -> None:
    onnx_tensor = onnx.helper.make_tensor_type_proto(
        onnx.TensorProto.DataType.FLOAT, [1, 2, 3]
    ).tensor_type
    tensor = onnx_builder.type.Tensor.from_proto(onnx_tensor)
    assert tensor.shape == (1, 2, 3)
    assert tensor.dtype == np.float32


def test_tensor_str() -> None:
    tensor = onnx_builder.type.Tensor((1, 2, 3))
    assert str(tensor) == "Tensor(shape=(1, 2, 3), dtype=np.float32)"
