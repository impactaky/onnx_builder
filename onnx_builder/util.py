import onnx
import numpy as np
from pathlib import Path
import glob


def make_value_info(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


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


def _impl_load_pb(pb, type_, names):
    f = open(pb, "rb")
    if type_ == "tensor_type":
        value = onnx.TensorProto()
    elif type_ == "sequence_type":
        value = onnx.SequenceProto()
    elif type_ == "map_type":
        value = onnx.MapProto()
    value.ParseFromString(f.read())
    if value.name not in names:
        value.name = names[0]
    return value


def load_pbs(pbs, vis):
    values = {}
    names = [x.name for x in vis]
    vis = {x.name: x for x in vis}
    for pb in pbs:
        tensor = _impl_load_pb(pb, "tensor_type", names)
        value_type = vis[tensor.name].type.WhichOneof("value")
        # TODO support map type
        if value_type == "tensor_type":
            values[tensor.name] = onnx.numpy_helper.to_array(tensor)
        elif value_type == "sequence_type":
            seq = _impl_load_pb(pb, value_type, names)
            if vis[seq.name].type.WhichOneof("value") != value_type:
                raise RuntimeError("Failure load {}".format(pb))
            values[tensor.name] = onnx.numpy_helper.to_list(seq)
        else:
            raise RuntimeError("Unsupported value_type: {}".format(value_type))
        if tensor.name in names:
            names.remove(tensor.name)
        else:
            names.pop(0)
    return values


def load_inputs_from_test_case(test_case_dir, test_case_name="test_data_set_0", onnx_name="model.onnx"):
    test_case_dir = Path(test_case_dir)
    model = onnx.load(test_case_dir / onnx_name)
    input_pbs = glob.glob(str(test_case_dir / test_case_name / "input_*.pb"))
    return load_pbs(input_pbs, model.graph.input)


def load_outputs_from_test_case(test_case_dir, test_case_name="test_data_set_0", onnx_name="model.onnx"):
    test_case_dir = Path(test_case_dir)
    model = onnx.load(test_case_dir / onnx_name)
    output_pbs = glob.glob(str(test_case_dir / test_case_name / "output_*.pb"))
    return load_pbs(output_pbs, model.graph.output)
