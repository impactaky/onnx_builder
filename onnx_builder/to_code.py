import re
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from google.protobuf.pyext._message import RepeatedCompositeContainer

import onnx_builder.util

np.set_printoptions(linewidth=np.inf, threshold=np.inf)


def to_python_name(name):
    return "v_" + re.sub(r"\W", "_", name)


def proto_to_code(obj, indent=0):
    if isinstance(obj, RepeatedCompositeContainer):
        ret = " " * indent + "[\n"
        for o in obj:
            ret += proto_to_code(o, indent + 4) + ",\n"
        ret += " " * indent + "]\n,"
        return ret
    if isinstance(obj, str):
        return " " * indent + "'{}'".format(obj)
    else:
        return " " * indent + "{}".format(obj)


class CodeGenerator:
    def __init__(self):
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
import onnx
import onnx_builder

cwd = Path('{}')
storage = cwd/'storage'
builder = onnx_builder.Builder(value_prefix='tmp')

""".format(
                self.output_dir.resolve()
            )
        )

        self.python_file.write("# inputs\n")
        if not inputs:
            initializers = [x.name for x in model.graph.initializer]
            for input_ in model.graph.input:
                if input_.name in initializers:
                    continue
                (shape, dtype) = onnx_builder.util.value_info_to_numpy_info(input_)
                self.python_file.write(
                    "{} = builder.Input(np.empty({}, dtype=np.{}), name='{}')\n".format(
                        to_python_name(input_.name),
                        shape,
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
            output_str = "builder.Output({}, name='{}'".format(
                to_python_name(output.name), output.name
            )
            (shape, dtype) = onnx_builder.util.value_info_to_numpy_info(output)
            if shape:
                output_str += ", shape={}".format(shape)
            if dtype != np.float32:
                output_str += ", dtype=np.{}".format(dtype)
            output_str += ")\n"
            self.python_file.write(output_str)
        self.python_file.write("\n")

    def generate(self, model_or_test_case, output_dir):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.storage_dir = self.output_dir / "storage"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.python_file = open(self.output_dir / "exporter.py", "w")
        if isinstance(model_or_test_case, onnx.ModelProto):
            model = model_or_test_case
            self.with_test_case = False
        elif str(model_or_test_case).endswith(".onnx"):
            model = onnx.load(model_or_test_case)
            self.with_test_case = False
        else:
            test_case_dir = Path(model_or_test_case)
            inputs = onnx_builder.util.load_inputs_from_test_case(test_case_dir)
            model = onnx.load(test_case_dir / "model.onnx")
            self.with_test_case = True

        if self.with_test_case:
            self._impl_from_onnx(model, inputs)
        else:
            self._impl_from_onnx(model)

        opset_imports = getattr(model, "opset_import")
        if opset_imports:
            self.python_file.write("opset_imports = []\n")
            for opset_import in opset_imports:
                self.python_file.write(
                    "opset_imports.append(onnx.OperatorSetIdProto())\n"
                )
                if opset_import.domain:
                    self.python_file.write(
                        "opset_imports[-1].domain = {}\n".format(
                            proto_to_code(opset_import.domain)
                        )
                    )
                if opset_import.version:
                    self.python_file.write(
                        "opset_imports[-1].version = {}\n".format(opset_import.version)
                    )

        if self.with_test_case:
            self.python_file.write("builder.export(\n")
            self.python_file.write("    cwd/'exported',\n")
        else:
            self.python_file.write("model = builder.build(\n")

        for field in onnx.ModelProto.DESCRIPTOR.fields:
            if field.name in ["graph", "producer_name", "producer_version"]:
                continue
            v = getattr(model, field.name)
            if not v:
                continue
            if field.name == "opset_import":
                self.python_file.write("    opset_imports = opset_imports,\n")
            else:
                self.python_file.write(
                    "    {} = {},\n".format(field.name, proto_to_code(v))
                )

        if self.with_test_case:
            self.python_file.write(")\n")
        else:
            self.python_file.write(")\n")
            self.python_file.write("(cwd/'exported').mkdir(exist_ok=True)\n")
            self.python_file.write("onnx.save(model, cwd/'exported'/'model.onnx')\n")
        self.python_file.close()
