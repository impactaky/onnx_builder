import onnx
import numpy as np
from pathlib import Path
import glob


def make_value_info(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def ndarray_to_value_info(
    arr: np.ndarray, name: str, shape=None, dtype=None
) -> onnx.ValueInfoProto:
    if dtype is None:
        dtype = arr.dtype
    else:
        dtype = np.dtype(dtype)
    if shape is None:
        shape = arr.shape
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype],
        shape=shape,
    )


def value_info_to_numpy_info(vi: onnx.ValueInfoProto):
    t = vi.type
    shape = 0
    elem_type = np.float32
    if t.WhichOneof("value") == "tensor_type":
        if t.tensor_type.HasField("shape"):
            if len(t.tensor_type.shape.dim):

                def dim_to_val(dim):
                    which = dim.WhichOneof("value")
                    assert which is not None
                    return getattr(dim, which)

                shape = tuple(map(dim_to_val, t.tensor_type.shape.dim))
            else:
                shape = []
        if t.tensor_type.HasField("elem_type"):
            elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                getattr(t.tensor_type, "elem_type")
            ]
    return (shape, elem_type)


def get_uninitialized_input_names(model):
    initializers = [x.name for x in model.graph.initializer]
    input_names = []
    for input_ in model.graph.input:
        if input_.name in initializers:
            continue
        input_names.append(input_.name)
    return input_names


def load_inputs_from_test_case(test_case_dir, test_case_name="test_data_set_0"):
    test_case_dir = Path(test_case_dir)
    model = onnx.load(test_case_dir / "model.onnx")
    input_names = get_uninitialized_input_names(model)
    input_pbs = glob.glob(str(test_case_dir / test_case_name / "input_*.pb"))
    inputs = {}
    for input_pb in input_pbs:
        with open(input_pb, "rb") as f:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(f.read())
        if tensor.name in input_names:
            input_names.remove(tensor.name)
        else:
            tensor.name = input_names.pop(0)
        inputs[tensor.name] = onnx.numpy_helper.to_array(tensor)
    return inputs
