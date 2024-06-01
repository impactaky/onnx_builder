import onnx
import numpy as np


class SequenceTensorType:
    def __init__(self, dim, dtype):
        self.dim = dim
        self.dtype = dtype

    @classmethod
    def from_proto(cls, proto: onnx.SequenceProto):
        if proto.elem_type.WhichOneof("value") == "tensor_type":
            dim = proto.elem_type.tensor_type.shape.dim[0].dim_value
            dtype = onnx.helper.tensor_dtype_to_np_dtype(
                proto.elem_type.tensor_type.elem_type
            )
            return cls(dim, dtype)
        else:
            raise ValueError("Only tensor type is supported")

    def to_code(self):
        return f"onnx_builder.type.SequenceTensorType(dim={self.dim}, dtype=np.{self.dtype})"

    def to_proto(self):
        elem_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(self.dtype))
        shape = [self.dim]
        # elem_shape_denotation = onnx.helper.ShapeProto.Dimension(dim_value=1)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
        sequence_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        return sequence_type_proto
