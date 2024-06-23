import onnx
from onnx import numpy_helper
import numpy as np
import numpy.typing as npt

from typing import List, Optional, Sequence, Union

ValueType = Union["Value", "OptionalValue"]
ValueContentType = list[npt.NDArray] | npt.NDArray | None


def make_tensor_optional_value_info(
    name: str,
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    doc_string: str = "",
    elem_shape_denotation: Optional[List[str]] = None,
) -> onnx.ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    value_info_proto = onnx.ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = onnx.helper.make_tensor_type_proto(
        elem_type, shape, elem_shape_denotation
    )
    optional_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
    value_info_proto.type.optional_type.CopyFrom(optional_type_proto.optional_type)

    return value_info_proto


def make_tensor_sequence_optional_value_info(
    name: str,
    elem_type: int,
    shape: Optional[Sequence[Union[str, int, None]]],
    doc_string: str = "",
    elem_shape_denotation: Optional[List[str]] = None,
) -> onnx.ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    value_info_proto = onnx.ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = onnx.helper.make_tensor_type_proto(
        elem_type, shape, elem_shape_denotation
    )
    sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
    optional_type_proto = onnx.helper.make_optional_type_proto(sequence_type_proto)
    value_info_proto.type.optional_type.CopyFrom(optional_type_proto.optional_type)

    return value_info_proto


def ndarray_to_value_info(
    arr: npt.NDArray | None,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
) -> onnx.ValueInfoProto:
    dtype_: np.dtype = np.dtype(np.float32)
    if dtype is None:
        if isinstance(arr, np.ndarray):
            dtype_ = arr.dtype
    else:
        dtype_ = np.dtype(dtype)
    if shape is None:
        if isinstance(arr, np.ndarray):
            shape = arr.shape
    if shape is None:
        return onnx.helper.make_empty_tensor_value_info(name)
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.helper.np_dtype_to_tensor_dtype(dtype_),
        shape=shape,
    )


class Value:
    def __init__(
        self,
        name: str,
        value: list[npt.NDArray] | npt.NDArray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        is_optional: bool = False,
    ):
        self.is_optional = is_optional
        self.name = name
        self.value = value
        if isinstance(value, np.ndarray):
            self.shape: tuple[int, ...] | None = value.shape
            self.dtype: np.dtype | None = value.dtype
        self.shape = shape
        self.dtype = dtype
        if isinstance(value, list) and not self.is_sequence():
            self.value = np.array(value)

    def is_sequence(self) -> bool:
        return isinstance(self.value, list) and (
            len(self.value) == 0 or isinstance(self.value[0], np.ndarray)
        )

    def value_info(self) -> onnx.ValueInfoProto:
        if self.is_optional:
            if self.is_sequence():
                return make_tensor_sequence_optional_value_info(
                    self.name,
                    onnx.helper.np_dtype_to_tensor_dtype(np.dtype(self.dtype)),
                    self.shape,
                )
            else:
                return make_tensor_optional_value_info(
                    self.name,
                    onnx.helper.np_dtype_to_tensor_dtype(np.dtype(self.dtype)),
                    self.shape,
                )
        elif self.is_sequence():
            return onnx.helper.make_tensor_sequence_value_info(
                self.name,
                onnx.helper.np_dtype_to_tensor_dtype(np.dtype(self.dtype)),
                self.shape,
            )
        elif self.value is None or isinstance(self.value, np.ndarray):
            return ndarray_to_value_info(self.value, self.name, self.shape, self.dtype)
        else:
            raise ValueError(f"Unsupported value type: {type(self.value)}")

    def tensorProto(self) -> onnx.TensorProto:
        assert isinstance(self.value, np.ndarray)
        return numpy_helper.from_array(self.value, name=self.name)

    def proto(self) -> onnx.SequenceProto | onnx.TensorProto:
        if self.is_sequence():
            assert isinstance(self.value, list)
            return numpy_helper.from_list(
                self.value,
                name=self.name,
            )
        else:
            return self.tensorProto()


class OptionalValue(Value):
    def __init__(
        self,
        name: str,
        value: list[npt.NDArray] | npt.NDArray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
    ):
        super().__init__(name, value, shape, dtype, is_optional=True)
