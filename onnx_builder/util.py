import onnx
import numpy as np


def make_value_info(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def ndarray_to_value_info(arr: np.ndarray, name: str) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        shape=arr.shape,
    )


def get_shape_from_value_info(vi: onnx.ValueInfoProto):
    t = vi.type
    if t.WhichOneof("value") == "tensor_type":
        if t.tensor_type.HasField("shape"):
            if len(t.tensor_type.shape.dim):

                def dim_to_val(dim):
                    which = dim.WhichOneof("value")
                    assert which is not None
                    return getattr(dim, which)

                return tuple(map(dim_to_val, t.tensor_type.shape.dim))
            else:
                return 1
    if t.WhichOneof("value") is None:
        return 0
    return "Unknown type {}".format(t.WhichOneof("value"))
