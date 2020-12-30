import onnx
from onnx import numpy_helper
import numpy as np
import re
from pathlib import Path
import onnx_builder.util

np.set_printoptions(linewidth=np.inf, threshold=np.inf)


def to_python_name(name):
    return "v_" + re.sub(r"\W", "_", name)


class ImplTensorToStr:
    def __init__(self, storage_dir, inline_threshold):
        self.storage_dir = storage_dir
        self.inline_threshold = inline_threshold

    def ndarray_to_str(self, array, name):
        if not array.shape:
            return "np.array([{}], dtype=np.{}).reshape([])".format(
                np.array2string(array, max_line_width=np.inf, separator=", ").replace(
                    "\n", ""
                ),
                array.dtype,
            )
        if array.size <= self.inline_threshold:
            return "np.array({}, dtype=np.{})".format(
                np.array2string(array, max_line_width=np.inf, separator=", ").replace(
                    "\n", ""
                ),
                array.dtype,
            )
        else:
            name = to_python_name(name)
            np.save(self.storage_dir / "{}.npy".format(name), array)
            return "np.load(storage/'{}.npy')".format(name)

    def tensor_to_str(self, tensor):
        array = numpy_helper.to_array(tensor)
        name = to_python_name(tensor.name)
        return self.ndarray_to_str(array, name)


def from_onnx(model, output_dir, inputs=None):
    if isinstance(model, str):
        model = onnx.load(model)
    output_dir = Path(output_dir)
    storage_dir = Path(output_dir) / "storage"
    impl_tensor_to_str = ImplTensorToStr(storage_dir, 12)
    # storage_dir, numpy_inline_threshold)
    storage_dir.mkdir(parents=True, exist_ok=True)
    ndarray_to_str = impl_tensor_to_str.ndarray_to_str
    tensor_to_str = impl_tensor_to_str.tensor_to_str
    python_file = open(output_dir / "exporter.py", "w")

    python_file.write(
        """import numpy as np
from pathlib import Path
import onnx_builder

cwd = Path(__file__).parent
storage = cwd/'storage'
builder = onnx_builder.Builder(value_prefix='tmp')

"""
    )

    python_file.write("# inputs\n")
    if not inputs:
        initializers = [x.name for x in model.graph.initializer]
        for input_ in model.graph.input:
            if input_.name in initializers:
                continue
            python_file.write(
                "{} = builder.Input(np.empty({}, dtype=np.{}), name='{}')\n".format(
                    to_python_name(input_.name),
                    onnx_builder.util.get_shape_from_value_info(input_),
                    onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                        getattr(input_.type.tensor_type, "elem_type")
                    ],
                    input_.name,
                )
            )
    else:
        for input_ in inputs:
            python_file.write(
                "{} = builder.Input({}, name='{}')\n".format(
                    to_python_name(input_[0]),
                    ndarray_to_str(input_[1], input_[0]),
                    input_[0],
                )
            )
    python_file.write("\n")

    python_file.write("# initializers\n")
    for initializer in model.graph.initializer:
        python_file.write(
            "{} = builder.Initializer({}, name='{}')\n".format(
                to_python_name(initializer.name),
                tensor_to_str(initializer),
                initializer.name,
            )
        )
    python_file.write("\n")

    python_file.write("# nodes\n")
    for node in model.graph.node:
        attributes = ""
        if len(node.output) > 1:
            attributes += "outs={}, ".format(len(node.output))
        if node.name:
            attributes += "name='{}', ".format(node.name)
        for i, attr in enumerate(node.attribute):
            if i != 0:
                attributes += ", "
            value = onnx.helper.get_attribute_value(attr)
            if type(value) == onnx.TensorProto:
                value = tensor_to_str(value)
            attributes += "{}={}".format(attr.name, value)

        outputs = [to_python_name(output) for output in node.output]
        if outputs:
            outputs = "{}".format(outputs)
            outputs = re.sub(r"[\[\]\']", "", outputs)
        else:
            outputs = ""
        inputs = [to_python_name(input_) if input_ else None for input_ in node.input]
        if inputs:
            inputs = "{}, ".format(inputs)
            inputs = re.sub(r"[\[\]\']", "", inputs)
        else:
            inputs = ""

        python_file.write(
            "{} = builder.{}({}{})\n".format(outputs, node.op_type, inputs, attributes)
        )
    python_file.write("\n")

    python_file.write("#outputs\n")
    for output in model.graph.output:
        python_file.write(
            "builder.Output({}, name='{}')\n".format(
                to_python_name(output.name), output.name
            )
        )
    python_file.write("\n")

    python_file.write("builder.export(cwd/'exported')\n")
