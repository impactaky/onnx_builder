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


def value_info_to_numpy_info(tp: onnx.TensorProto):
    shape = None
    elem_type = np.float32
    if tp.HasField("shape"):
        if len(tp.shape.dim):

            def dim_to_val(dim):
                which = dim.WhichOneof("value")
                if which is None:
                    return None
                return getattr(dim, which)

            shape = tuple(map(dim_to_val, tp.shape.dim))
        else:
            shape = []
    if tp.HasField("elem_type"):
        elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[getattr(tp, "elem_type")]
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


def load_outputs_from_test_case(test_case_dir, test_case_name="test_data_set_0"):
    test_case_dir = Path(test_case_dir)
    model = onnx.load(test_case_dir / "model.onnx")
    output_names = [x.name for x in model.graph.output]
    output_pbs = glob.glob(str(test_case_dir / test_case_name / "output_*.pb"))
    outputs = {}
    for output_pb in output_pbs:
        with open(output_pb, "rb") as f:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(f.read())
        if tensor.name in output_names:
            output_names.remove(tensor.name)
        else:
            tensor.name = output_names.pop(0)
        outputs[tensor.name] = onnx.numpy_helper.to_array(tensor)
    return outputs
