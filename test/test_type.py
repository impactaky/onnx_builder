import onnx
import onnx_builder
import numpy as np
from onnx_builder.util import dtype_to_str


def test_tensor_from_proto() -> None:
    onnx_tensor = onnx.helper.make_tensor_type_proto(
        onnx.TensorProto.DataType.FLOAT, [1, 2, 3]
    ).tensor_type
    tensor = onnx_builder.type.Tensor.from_proto(onnx_tensor)
    assert tensor.shape == (1, 2, 3)
    assert tensor.dtype == np.float32


def test_dtype_to_str() -> None:
    assert dtype_to_str(np.float32) == "np.float32"
    assert dtype_to_str(np.bool_) == "np.bool_"
    assert dtype_to_str(np.object_) == "np.object_"


def test_tensor_str() -> None:
    tensor = onnx_builder.type.Tensor(shape=(1, 2, 3))
    assert str(tensor) == "Tensor(shape=(1, 2, 3), dtype=np.float32)"
    tensor = onnx_builder.type.Tensor(np.array((1, 2, 3)))
    assert str(tensor) == "np.array([1, 2, 3], dtype=np.int64)"


def test_tensor_to_code() -> None:
    tensor = onnx_builder.type.Tensor(shape=(1, 2, 3))
    assert str(tensor) == "Tensor(shape=(1, 2, 3), dtype=np.float32)"
    tensor = onnx_builder.type.Tensor(np.array((1, 2, 3)))
    assert str(tensor) == "np.array([1, 2, 3], dtype=np.int64)"
