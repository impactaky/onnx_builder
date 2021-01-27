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
    def __init__(self, builder_name="builder", subgraph_index=0, inline_threshold=12):
        self.base_indent = 0
        self.inline_threshold = inline_threshold
        self.subgraph_index = subgraph_index
        self.builder_name = builder_name

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

    def write(self, line=""):
        self.python_file.write(" " * self.base_indent + str(line) + "\n")

    def value_info_to_code(self, vi):
        if vi.type.WhichOneof("value") == "tensor_type":
            (shape, dtype) = onnx_builder.util.value_info_to_numpy_info(
                vi.type.tensor_type
            )
            return (
                "",
                "shape={}, dtype=np.{}, name='{}'".format(
                    shape,
                    dtype,
                    vi.name,
                ),
            )
        elif vi.type.WhichOneof("value") == "sequence_type":
            (shape, dtype) = onnx_builder.util.value_info_to_numpy_info(
                vi.type.sequence_type.elem_type.tensor_type
            )
            return (
                "Sequence",
                "shape={}, dtype=np.{}, name='{}'".format(
                    shape,
                    dtype,
                    vi.name,
                ),
            )
        else:
            return ("", "name='{}'".format(vi.name))

    def graph_to_code(self, graph, inputs=None):
        self.write("# inputs")
        if not inputs:
            initializers = [x.name for x in graph.initializer]
            for input_ in graph.input:
                if input_.name in initializers:
                    continue
                (input_type, input_args) = self.value_info_to_code(input_)
                self.write(
                    "{} = {}.Input{}({})".format(
                        to_python_name(input_.name),
                        self.builder_name,
                        input_type,
                        input_args,
                    )
                )
        else:
            for name, array in inputs.items():
                self.write(
                    "{} = {}.Input({}, name='{}')".format(
                        to_python_name(name),
                        self.builder_name,
                        self.ndarray_to_str(array, name),
                        name,
                    )
                )
        self.write()

        self.write("# initializers")
        for initializer in graph.initializer:
            self.python_file.write(
                "{} = {}.Initializer({}, name='{}')\n".format(
                    to_python_name(initializer.name),
                    self.builder_name,
                    self.tensor_to_str(initializer),
                    initializer.name,
                )
            )
        self.write()

        self.write("# nodes")
        input_names = [x.name for x in graph.input]
        for node in graph.node:
            attributes = ""
            if len(node.output) > 1:
                attributes += "outs={}, ".format(len(node.output))
            if node.output:
                attributes += "output_names={}, ".format(node.output)
            if node.name:
                attributes += "name='{}', ".format(node.name)
            for i, attr in enumerate(node.attribute):
                if i != 0:
                    attributes += ", "
                value = onnx.helper.get_attribute_value(attr)
                if type(value) == onnx.TensorProto:
                    value = self.tensor_to_str(value)
                if type(value) == onnx.GraphProto:
                    graph_name = "subgraph{}".format(self.subgraph_index)
                    self.subgraph_index += 1
                    generator = onnx_builder.CodeGenerator(
                        builder_name=graph_name + "_builder",
                        subgraph_index=self.subgraph_index,
                    )
                    generator.python_file = self.python_file
                    generator.write(
                        "{}_builder = onnx_builder.Builder(value_prefix='{}_tmp')".format(
                            graph_name, graph_name
                        )
                    )
                    generator.graph_to_code(value)
                    generator.write(
                        "{} = {}_builder.make_graph()".format(graph_name, graph_name)
                    )
                    self.subgraph_index = generator.subgraph_index
                    value = graph_name
                attributes += "{}={}".format(attr.name, value)

            outputs = [to_python_name(output) for output in node.output]
            input_names += node.output
            if outputs:
                outputs = "{}".format(outputs)
                outputs = re.sub(r"[\[\]\']", "", outputs)
            else:
                outputs = ""
            inputs = []
            for input_ in node.input:
                if input_:
                    inputs.append(to_python_name(input_))
                else:
                    inputs.append(None)

            if inputs:
                inputs = "{}, ".format(inputs)
                inputs = re.sub(r"[\[\]\']", "", inputs)
            else:
                inputs = ""

            self.write(
                "{} = {}.{}({}{})".format(
                    outputs, self.builder_name, node.op_type, inputs, attributes
                )
            )
        self.write()

        self.write("#outputs")
        for output in graph.output:
            (output_type, output_args) = self.value_info_to_code(output)
            self.write(
                "{}.Output{}({}, {})".format(
                    self.builder_name,
                    output_type,
                    to_python_name(output.name),
                    output_args,
                )
            )
        self.write()

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

        self.write("import numpy as np")
        self.write("from pathlib import Path")
        self.write("import onnx")
        self.write("import onnx_builder")
        self.write()
        self.write("cwd = Path('{}')".format(self.output_dir.resolve()))
        self.write("storage = cwd/'storage'")
        self.write(
            "{} = onnx_builder.Builder(value_prefix='tmp')".format(self.builder_name)
        )

        if self.with_test_case:
            self.graph_to_code(model.graph, inputs)
        else:
            self.graph_to_code(model.graph)

        opset_imports = getattr(model, "opset_import")
        if opset_imports:
            self.write("opset_imports = []")
            for opset_import in opset_imports:
                self.write("opset_imports.append(onnx.OperatorSetIdProto())")
                if opset_import.domain:
                    self.write(
                        "opset_imports[-1].domain = {}".format(
                            proto_to_code(opset_import.domain)
                        )
                    )
                if opset_import.version:
                    self.write(
                        "opset_imports[-1].version = {}".format(opset_import.version)
                    )

        if self.with_test_case:
            self.write("{}.export(".format(self.builder_name))
            self.write("    cwd/'exported',")
        else:
            self.write("model = {}.build(".format(self.builder_name))

        for field in onnx.ModelProto.DESCRIPTOR.fields:
            if field.name in ["graph", "producer_name", "producer_version"]:
                continue
            v = getattr(model, field.name)
            if not v:
                continue
            if field.name == "opset_import":
                self.write("    opset_imports = opset_imports,")
            else:
                self.write("    {} = {},".format(field.name, proto_to_code(v)))
        self.write(")")
        if not self.with_test_case:
            self.write("(cwd/'exported').mkdir(exist_ok=True)")
            self.write("onnx.save(model, cwd/'exported'/'model.onnx')")
        self.python_file.close()
