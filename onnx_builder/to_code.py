import onnx
from onnx import numpy_helper
import numpy as np
import re
from pathlib import Path
import onnx_builder.util

np.set_printoptions(linewidth=np.inf, threshold=np.inf)


def to_python_name(name):
    return "v_" + re.sub(r"\W", "_", name)


class CodeGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.storage_dir = self.output_dir / "storage"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.python_file = open(self.output_dir / "exporter.py", "w")
        self.inline_threshold = 12

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

    def _impl_from_onnx(self, model, inputs=None):
        self.python_file.write(
            """import numpy as np
from pathlib import Path
import onnx_builder

cwd = Path(__file__).parent
storage = cwd/'storage'
builder = onnx_builder.Builder(value_prefix='tmp')

"""
        )

        self.python_file.write("# inputs\n")
        if not inputs:
            initializers = [x.name for x in model.graph.initializer]
            for input_ in model.graph.input:
                if input_.name in initializers:
                    continue
                self.python_file.write(
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
            for name, array in inputs.items():
                self.python_file.write(
                    "{} = builder.Input({}, name='{}')\n".format(
                        to_python_name(name),
                        self.ndarray_to_str(array, name),
                        name,
                    )
                )
        self.python_file.write("\n")

        self.python_file.write("# initializers\n")
        for initializer in model.graph.initializer:
            self.python_file.write(
                "{} = builder.Initializer({}, name='{}')\n".format(
                    to_python_name(initializer.name),
                    self.tensor_to_str(initializer),
                    initializer.name,
                )
            )
        self.python_file.write("\n")

        self.python_file.write("# nodes\n")
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
                    value = self.tensor_to_str(value)
                attributes += "{}={}".format(attr.name, value)

            outputs = [to_python_name(output) for output in node.output]
            if outputs:
                outputs = "{}".format(outputs)
                outputs = re.sub(r"[\[\]\']", "", outputs)
            else:
                outputs = ""
            inputs = [
                to_python_name(input_) if input_ else None for input_ in node.input
            ]
            if inputs:
                inputs = "{}, ".format(inputs)
                inputs = re.sub(r"[\[\]\']", "", inputs)
            else:
                inputs = ""

            self.python_file.write(
                "{} = builder.{}({}{})\n".format(
                    outputs, node.op_type, inputs, attributes
                )
            )
        self.python_file.write("\n")

        self.python_file.write("#outputs\n")
        for output in model.graph.output:
            self.python_file.write(
                "builder.Output({}, name='{}')\n".format(
                    to_python_name(output.name), output.name
                )
            )
        self.python_file.write("\n")

    def from_onnx(self, model):
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model)
        self._impl_from_onnx(model)
        self.python_file.write("builder.export(cwd/'exported')\n")

    def from_test_case(self, test_case_dir):
        test_case_dir = Path(test_case_dir)
        inputs = onnx_builder.util.load_inputs_from_test_case(test_case_dir)
        model = onnx.load(test_case_dir / "model.onnx")
        self._impl_from_onnx(model, inputs)
