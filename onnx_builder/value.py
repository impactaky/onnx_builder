import onnx
from onnx import numpy_helper
import numpy as np

from typing import List, Optional, Sequence, Union


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
    arr: np.ndarray, name: str, shape=None, dtype=None
) -> onnx.ValueInfoProto:
    if dtype is None:
        if isinstance(arr, np.ndarray):
            dtype = arr.dtype
    else:
        dtype = np.dtype(dtype)
    if shape is None:
        if isinstance(arr, np.ndarray):
            shape = arr.shape
    if shape is None:
        return onnx.helper.make_empty_tensor_value_info(name)
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.helper.np_dtype_to_tensor_dtype(dtype),
        shape=shape,
    )


class Value:
    def __init__(self, name, value=None, shape=None, dtype=None, is_optional=False):
        self.is_optional = is_optional
        self.name = name
        self.value = value
        if isinstance(value, np.ndarray):
            self.shape = value.shape
            self.dtype = value.dtype
        self.shape = shape
        self.dtype = dtype
        if isinstance(value, list) and not self.is_sequence():
            if isinstance(value[0], float):
                self.value = np.array(value).astype(np.float32)
            else:
                self.value = np.array(value)

    def is_sequence(self):
        return isinstance(self.value, list) and (
            len(self.value) == 0 or isinstance(self.value[0], np.ndarray)
        )

    def value_info(self):
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
        else:
            return ndarray_to_value_info(self.value, self.name, self.shape, self.dtype)

    def proto(self):
        if self.is_sequence():
            return numpy_helper.from_list(self.value, self.name, self.dtype)
        else:
            return numpy_helper.from_array(self.value, name=self.name)


class OptionalValue(Value):
    def __init__(self, name, value=None, shape=None, dtype=None):
        super().__init__(name, value, shape, dtype, is_optional=True)
