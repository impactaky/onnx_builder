import onnx
from onnx import numpy_helper
import onnx_builder.util
import numpy as np


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
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype],
        shape=shape,
    )


class Value:
    def __init__(self, name, value=None, shape=None, dtype=None):
        self.name = name
        self.value = value
        if isinstance(value, np.ndarray):
            self.shape = value.shape
            self.dtype = value.dtype
        self.shape = shape
        self.dtype = dtype
        if isinstance(value, list) and not self.is_sequence():
            if type(value[0]) is float: 
                self.value = np.array(value).astype(np.float32)
            else:
                self.value = np.array(value)

    def is_sequence(self):
        return isinstance(self.value, list) and (
            len(self.value) == 0 or isinstance(self.value[0], np.ndarray)
        )

    def value_info(self):
        if self.is_sequence():
            return onnx.helper.make_tensor_sequence_value_info(
                self.name,
                onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(self.dtype)],
                self.shape,
            )
        else:
            return ndarray_to_value_info(self.value, self.name, self.shape, self.dtype)

    def proto(self):
        if self.is_sequence():
            return numpy_helper.from_list(self.value, self.name, self.dtype)
        else:
            return numpy_helper.from_array(self.value, name=self.name)
