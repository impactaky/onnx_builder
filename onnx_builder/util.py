import onnx
import numpy as np
import numpy.typing as npt
from pathlib import Path
import glob
import os
import sys
import re


def make_value_info(name: str) -> onnx.ValueInfoProto:
    vi = onnx.ValueInfoProto()
    vi.name = name
    return vi


def value_info_to_numpy_info(
    tp: onnx.TypeProto.Tensor,
) -> tuple[tuple[int | str | None, ...] | None, npt.DTypeLike]:
    shape = None
    elem_type: npt.DTypeLike = np.float32
    if tp.HasField("shape"):
        if len(tp.shape.dim):

            def dim_to_val(dim: onnx.TensorShapeProto.Dimension) -> int | str | None:
                which = dim.WhichOneof("value")
                if which is None:
                    return None
                val = getattr(dim, which)
                if not isinstance(val, (str, int)) and val is None:
                    raise NotImplementedError("{}".format(type(val).__name__))
                return val

            shape = tuple(map(dim_to_val, tp.shape.dim))
        else:
            shape = ()
    if tp.HasField("elem_type"):
        elem_type = onnx.helper.tensor_dtype_to_np_dtype(getattr(tp, "elem_type"))
    return (shape, elem_type)


def dtype_to_str(dtype: npt.DTypeLike) -> str:
    ret = f"np.{np.dtype(dtype).name}"
    if dtype == np.bool_:
        ret = "np.bool_"
    elif dtype == np.object_:
        ret = "np.object_"
    return ret


def to_python_name(name: str) -> str:
    return "v_" + re.sub(r"\W", "_", name)


def ndarray_to_str(
    array: npt.NDArray,
    name: str | None = None,
    inline_threshold: int = 12,
    storage_dir: Path | str | None = None,
) -> str:
    if array.size == 0:
        return "np.empty(shape={}, dtype={})".format(
            array.shape, dtype_to_str(array.dtype)
        )

    if array.size <= inline_threshold:
        val = (
            np.array2string(array, max_line_width=sys.maxsize, separator=", ")
            .replace("\n", "")
            .replace("  ", " ")
            .replace("inf", "np.inf")
            .replace("nan", "np.nan")
        )
        return "np.array({}, dtype={})".format(
            val,
            dtype_to_str(array.dtype),
        )

    if storage_dir is None:
        if name is None:
            name = "unnamed"
        return f"np.load('{name}.npy')"

    assert name is not None
    name = to_python_name(name)
    save_path = Path(storage_dir) / f"{name}.npy"
    np.save(save_path, array)
    return f"np.load('{save_path}')"


def _impl_load_pb(
    pb: str, type_: str, names: list[str]
) -> onnx.TensorProto | onnx.SequenceProto | onnx.MapProto:
    f = open(pb, "rb")
    if type_ == "tensor_type":
        value = onnx.TensorProto()
    elif type_ == "sequence_type":
        value = onnx.SequenceProto()
    elif type_ == "map_type":
        value = onnx.MapProto()
    elif type_ == "optional_type":
        value = onnx.OptionalProto()
    else:
        raise NotImplementedError("Unsupported type: {}".format(type_))
    value.ParseFromString(f.read())
    if value.name not in names:
        value.name = names[0]
    return value


def load_pbs(
    pbs: list[str], vis: list[onnx.ValueInfoProto]
) -> dict[str, np.ndarray | list[np.ndarray]]:
    values = {}
    names = [x.name for x in vis]
    vis_dict = {x.name: x for x in vis}
    for pb in pbs:
        tensor = _impl_load_pb(pb, "tensor_type", names)
        value_type = vis_dict[tensor.name].type.WhichOneof("value")
        name = tensor.name
        # TODO support map type
        if value_type == "tensor_type":
            values[name] = onnx.numpy_helper.to_array(tensor)
        elif value_type == "sequence_type":
            seq = _impl_load_pb(pb, value_type, names)
            name = seq.name
            if vis_dict[seq.name].type.WhichOneof("value") != value_type:
                raise RuntimeError("Failure load {}".format(pb))
            values[name] = onnx.numpy_helper.to_list(seq)
        elif value_type == "optional_type":
            opt = _impl_load_pb(pb, value_type, names)
            name = opt.name
            values[name] = onnx.numpy_helper.to_optional(opt)
        else:
            raise RuntimeError("Unsupported value_type: {}".format(value_type))
        if tensor.name in names:
            names.remove(name)
        else:
            names.pop(0)
    return values


def load_inputs_from_test_case(
    test_case_dir: os.PathLike,
    test_case_name: str = "test_data_set_0",
    onnx_name: str = "model.onnx",
) -> dict[str, np.ndarray | list[np.ndarray]]:
    test_case_path = Path(test_case_dir)
    model = onnx.load(test_case_path / onnx_name)
    input_pbs = glob.glob(str(test_case_path / test_case_name / "input_*.pb"))
    return load_pbs(input_pbs, model.graph.input)


def load_outputs_from_test_case(
    test_case_dir: os.PathLike,
    test_case_name: str = "test_data_set_0",
    onnx_name: str = "model.onnx",
) -> dict[str, np.ndarray | list[np.ndarray]]:
    test_case_path = Path(test_case_dir)
    model = onnx.load(test_case_path / onnx_name)
    output_pbs = glob.glob(str(test_case_path / test_case_name / "output_*.pb"))
    return load_pbs(output_pbs, model.graph.output)
