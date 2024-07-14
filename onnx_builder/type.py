import onnx
import numpy as np
from pathlib import Path
from .util import value_info_to_numpy_info, ndarray_to_str, dtype_to_str
from numpy import typing as npt

ShapeType = tuple[int | str | None, ...]
TensorDType = (
    np.float32
    | np.uint8
    | np.int8
    | np.uint16
    | np.int16
    | np.int32
    | np.int64
    | np.bool_
    | np.float16
    | np.float64
    | np.complex64
    | np.complex128
    | np.uint32
    | np.uint64
    | np.object_
)
ArrayType = npt.NDArray[TensorDType]


class Tensor:
    def __init__(
        self,
        array: npt.NDArray[np.float32] | None = None,
        shape: ShapeType | None = None,
        dtype: npt.DTypeLike = np.float32,
    ):
        self.array = array
        if array is not None:
            if shape is None:
                shape = array.shape
            if dtype is None:
                dtype = array.dtype
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_proto(cls, proto: onnx.TypeProto.Tensor) -> "Tensor":
        shape, dtype = value_info_to_numpy_info(proto)
        return cls(shape=shape, dtype=dtype)

    def __str__(self) -> str:
        if self.array is None:
            return f"Tensor(shape={self.shape}, dtype={dtype_to_str(self.dtype)})"
        return ndarray_to_str(self.array)

    def to_code(
        self, name: str | None = None, storage_dir: Path | str | None = None
    ) -> str:
        if self.array is None:
            return f"Tensor(shape={self.shape}, dtype={dtype_to_str(self.dtype)})"
        return ndarray_to_str(self.array, name=name, storage_dir=storage_dir)


class SequenceTensorType:
    def __init__(self, dim: int, dtype: npt.DTypeLike):
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
