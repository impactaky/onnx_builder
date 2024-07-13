import onnx
import numpy as np
from .util import value_info_to_numpy_info


class Tensor:
    def __init__(
        self, shape: tuple[int | str | None, ...] | None, dtype: np.dtype = np.float32
    ):
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_proto(cls, proto: onnx.TypeProto.Tensor) -> "Tensor":
        shape, dtype = value_info_to_numpy_info(proto)
        return cls(shape, dtype)


class SequenceTensorType:
    def __init__(self, dim: int, dtype: np.dtype):
        self.dim = dim
        self.dtype = dtype

    @classmethod
    def from_proto(cls, proto: onnx.TypeProto.Sequence) -> "SequenceTensorType":
        elem_type: onnx.TypeProto = proto.elem_type
        if elem_type.WhichOneof("value") == "tensor_type":
            dim = elem_type.tensor_type.shape.dim[0].dim_value
            dtype = onnx.helper.tensor_dtype_to_np_dtype(
                proto.elem_type.tensor_type.elem_type
            )
            return cls(dim, dtype)
        else:
            raise ValueError("Only tensor type is supported")

    def to_code(self) -> str:
        return f"onnx_builder.type.SequenceTensorType(dim={self.dim}, dtype=np.{self.dtype})"

    def to_proto(self) -> onnx.TypeProto:
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(self.dtype))
        shape = [self.dim]
        # elem_shape_denotation = onnx.helper.ShapeProto.Dimension(dim_value=1)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
        sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        return sequence_type_proto
