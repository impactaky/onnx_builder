from pathlib import Path
import onnx
from onnx import numpy_helper
import numpy as np
import numpy.typing as npt
import onnx_builder
from onnx_builder.value import Value, OptionalValue, ValueType, ValueContentType
from .type import SequenceTensorType
from typing import Callable, Any
from google.protobuf.internal.containers import RepeatedScalarFieldContainer


ModelEvalFuncType = Callable[
    [onnx.ModelProto, dict[str, ValueContentType], list[str]], list[npt.NDArray]
]


def _eval_with_onnxruntime(
    model: onnx.ModelProto, inputs: dict[str, ValueContentType], output_names: list[str]
) -> list[npt.NDArray]:
    import onnxruntime

    session = onnxruntime.InferenceSession(model.SerializeToString())
    ret: list[npt.NDArray] = session.run(output_names, inputs)
    return ret


class Builder:
    def __init__(
        self,
        opset_imports: int | None = None,
        value_prefix: str = "onnx_builder_tmp",
        eval_func: ModelEvalFuncType = _eval_with_onnxruntime,
    ):
        self.opset_imports = opset_imports
        self.value_prefix = value_prefix
        self.__eval_func = eval_func
        self.__value_idx = 0
        self.__nodes: list[onnx.NodeProto] = []
        self.values: dict[str, ValueType] = {}
        self.initializers: list[str] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []

    def __GenValueName(self) -> str:
        self.__value_idx += 1
        return self.value_prefix + "_" + str(self.__value_idx)

    def rename_value(self, from_: str, to_: str) -> None:
        self.inputs = [to_ if x == from_ else x for x in self.inputs]
        self.outputs = [to_ if x == from_ else x for x in self.outputs]
        for node in self.__nodes:
            for i, name in enumerate(node.input):
                if name == from_:
                    node.input[i] = to_
            for i, name in enumerate(node.output):
                if name == from_:
                    node.output[i] = to_
        if from_ in self.values:
            self.values[to_] = self.values[from_]
            del self.values[from_]

    def add_value(self, name: str, value: npt.NDArray | None = None) -> None:
        self.values[name] = Value(name, value)

    def remove_value(self, name: str) -> None:
        if name in self.inputs:
            self.inputs.remove(name)
        if name in self.outputs:
            self.outputs.remove(name)
        if name in self.initializers:
            self.initializers.remove(name)
        del self.values[name]

    def nodes(self) -> list[onnx.NodeProto]:
        return self.__nodes

    def Initializer(self, value: ValueContentType, name: str = "") -> ValueType:
        if not name:
            name = self.__GenValueName()
        self.values[name] = Value(name, value)
        self.inputs.append(name)
        self.initializers.append(name)
        return self.values[name]

    def Input(
        self,
        value: ValueContentType | None = None,
        name: str = "",
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        value_type: str = "tensor_type",
    ) -> ValueType:
        if value_type == "sequence_type" and value is None:
            value = []
        if not name:
            name = self.__GenValueName()
        if value_type == "optional_type":
            self.values[name] = OptionalValue(name, value, shape=shape, dtype=dtype)
        else:
            self.values[name] = Value(name, value, shape=shape, dtype=dtype)
        self.inputs.append(name)
        return self.values[name]

    def OptionalInput(
        self,
        value: ValueContentType | None = None,
        name: str = "",
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        value_type: str = "tensor_type",
    ) -> ValueType:
        if value_type == "sequence_type" and value is None:
            value = []
        if not name:
            name = self.__GenValueName()
        self.values[name] = OptionalValue(name, value, shape=shape, dtype=dtype)
        self.inputs.append(name)
        return self.values[name]

    def InputSequence(
        self,
        list_: list[npt.NDArray] = [],
        name: str = "",
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
    ) -> ValueType:
        return self.Input(
            list_, name=name, shape=shape, dtype=dtype, value_type="sequence_type"
        )

    def ValueToOutput(self, name: str) -> None:
        if name not in self.outputs:
            self.outputs.append(name)

    def Output(
        self,
        value: ValueType,
        name: str = "",
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        value_type: str = "tensor_type",
    ) -> ValueType:
        if value_type == "sequence_type" and value.value is None:
            value.value = []
        if name:
            for node in self.__nodes:
                if value.name in node.output:
                    index = list(node.output).index(value.name)
                    node.output[index] = name
                    break
            value.name = name
            self.values[name] = value
            value = self.values[name]
            value.name = name
        if shape is not None:
            if value.shape:
                assert shape != value.shape
            value.shape = shape
        if dtype is not None:
            if value.dtype:
                assert dtype != value.dtype
            value.dtype = dtype
        self.outputs.append(value.name)
        return value

    def OptionalOutput(
        self,
        value: ValueType,
        name: str = "",
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        value_type: str = "tensor_type",
    ) -> ValueType:
        value.is_optional = True
        if value_type == "sequence_type" and value.value is None:
            value.value = []
        if name:
            for node in self.__nodes:
                if value.name in node.output:
                    index = list(node.output).index(value.name)
                    node.output[index] = name
                    break
            value.name = name
            self.values[name] = value
            value = self.values[name]
            value.name = name
        if shape is not None:
            if value.shape:
                assert shape != value.shape
            value.shape = shape
        if dtype is not None:
            if value.dtype:
                assert dtype != value.dtype
            value.dtype = dtype
        self.outputs.append(value.name)
        return value

    def OutputSequence(
        self,
        value: ValueType,
        name: str = "",
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
    ) -> ValueType:
        return self.Output(
            value, name=name, shape=shape, dtype=dtype, value_type="sequence_type"
        )

    def Model(
        self,
        *args: ValueType,
        file_path: str = "",
        prefix: str | None = None,
        **kwargs: ValueType,
    ) -> ValueType | list[ValueType]:
        if prefix is None:
            prefix = file_path + "_"

        model = onnx.load(file_path)
        self.opset_imports = getattr(model, "opset_import")

        value_table = {}
        args_index = 0

        initializer_names = [x.name for x in model.graph.initializer]
        for input_ in model.graph.input:
            if input_.name in kwargs:
                value_table[input_.name] = kwargs[input_.name].name
            elif input_.name in initializer_names:
                pass
            else:
                value_table[input_.name] = args[args_index].name
                args_index += 1

        def resolve_value_names(names: RepeatedScalarFieldContainer[str]) -> None:
            for i in range(len(names)):
                if names[i] in value_table:
                    names[i] = value_table[names[i]]
                else:
                    names[i] = prefix + names[i]

        for node in model.graph.node:
            if node.name:
                node.name = prefix + node.name
            resolve_value_names(node.input)
            resolve_value_names(node.output)
            for output in node.output:
                self.add_value(output)
            self.__nodes.append(node)

        for initializer in model.graph.initializer:
            initializer.name = prefix + initializer.name
            array = numpy_helper.to_array(initializer)
            self.add_value(initializer.name, array)
            self.initializers.append(initializer.name)

        outputs = [Value(prefix + output.name) for output in model.graph.output]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def make_graph(self, name: str = "") -> onnx.GraphProto:
        if not name:
            name = self.__GenValueName()
        input_vis = [self.values[x].value_info() for x in self.inputs]
        initializers = [self.values[x].tensorProto() for x in self.initializers]
        output_vis = [self.values[x].value_info() for x in self.outputs]
        return onnx.helper.make_graph(
            self.__nodes,
            name,
            inputs=input_vis,
            outputs=output_vis,
            initializer=initializers,
        )

    def build(self, graph_name: str = "model.root", **kwargs: Any) -> onnx.ModelProto:
        if "opset_imports" not in kwargs or kwargs["opset_imports"] is None:
            kwargs["opset_imports"] = self.opset_imports
        graph = self.make_graph(name=graph_name)
        model = onnx.helper.make_model(
            graph, producer_name="onnx_builder", producer_version="0.01", **kwargs
        )
        # model = onnx.shape_inference.infer_shapes(model)
        return model

    def eval(
        self, eval_func: ModelEvalFuncType | None = None, **kwargs: Any
    ) -> tuple[onnx.ModelProto, list[npt.NDArray]]:
        model = self.build(**kwargs)
        inputs = {x: self.values[x].value for x in self.inputs}
        if eval_func is not None:
            outputs = eval_func(model, inputs, self.outputs)
        else:
            outputs = self.__eval_func(model, inputs, self.outputs)

        for name, value in zip(self.outputs, outputs):
            self.values[name].value = value
        return (model, outputs)

    def export(self, output_dir: str | Path, **kwargs: Any) -> None:
        model, outputs = self.eval(**kwargs)
        output_dir = Path(output_dir)
        (output_dir / "test_data_set_0").mkdir(parents=True, exist_ok=True)
        # save inputs
        initializer_names = self.initializers
        input_names = self.inputs
        for i, input_ in enumerate(self.inputs):
            if input_names[i] in initializer_names:
                continue
            with open(
                output_dir / "test_data_set_0" / "input_{}.pb".format(i), "wb"
            ) as f:
                f.write(self.values[input_].proto().SerializeToString())
        # save outputs
        output_names = self.outputs
        for i, output_ in enumerate(outputs):
            with open(
                output_dir / "test_data_set_0" / "output_{}.pb".format(i), "wb"
            ) as f:
                f.write(Value(output_names[i], output_).proto().SerializeToString())
        onnx.save(model, output_dir / "model.onnx")

    def __getattr__(self, op: str) -> Callable[..., ValueType | list[ValueType]]:
        def fn(
            *args: ValueType | ValueContentType | None,
            outs: int = 1,
            output_names: list[str] = [],
            name: str | None = None,
            insert_index: int | None = None,
            **kwargs: Any,
        ) -> ValueType | list[ValueType]:
            inputs = list(args)
            input_names = []
            for i, input_ in enumerate(inputs):
                if type(input_) == Value:
                    input_names.append(input_.name)
                    inputs[i] = input_.value
                elif type(input_) == OptionalValue:
                    input_names.append(input_.name)
                    inputs[i] = input_.value
                elif input_ is None:
                    input_names.append("")
                elif isinstance(input_, np.ndarray):
                    input_names.append(self.__GenValueName())
                    constant_tensor = numpy_helper.from_array(
                        input_, name=input_names[-1] + "_val"
                    )
                    new_node = onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[input_names[-1]],
                        value=constant_tensor,
                    )
                    if insert_index is None:
                        self.__nodes.append(new_node)
                    else:
                        self.__nodes.insert(insert_index, new_node)
                        if insert_index >= 0:
                            insert_index += 1
                    self.add_value(input_names[-1], input_)
                else:
                    raise ValueError("Invalid input type: {}".format(type(input_)))

            if not output_names:
                output_names = [self.__GenValueName() for i in range(outs)]
            for k, v in kwargs.items():
                if type(v) == np.ndarray:
                    kwargs[k] = onnx.numpy_helper.from_array(v, self.__GenValueName())
                if k == "to" and not isinstance(v, int):
                    kwargs[k] = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(v))
                if isinstance(v, SequenceTensorType):
                    kwargs[k] = v.to_proto()
            node = onnx.helper.make_node(
                op, inputs=input_names, outputs=output_names, name=name, **kwargs
            )
            if insert_index is None:
                self.__nodes.append(node)
            else:
                self.__nodes.insert(insert_index, node)
                if insert_index >= 0:
                    insert_index += 1
            for name in output_names:
                self.add_value(name)

            if outs == 1:
                return self.values[output_names[0]]
            else:
                return [self.values[name] for name in output_names]

        return fn


def from_onnx(path: str) -> Builder:
    # model = onnx.load(path)
    builder = Builder()
    inputs: list[ValueType] = []
    # TODO write test for
    # initializer_names = set(x.name for x in model.graph.initializer)
    # for input_ in model.graph.input:
    #     pass
    #     if input_.name not in initializer_names:
    #         if input_.type.WhichOneof("value") == "tensor_type":
    #             (shape, dtype) = onnx_builder.util.value_info_to_numpy_info(
    #                 input_.type.tensor_type
    #             )
    #         inputs.append(builder.Input(np.zeros(shape, dtype=dtype), input_.name))
    outputs = builder.Model(*inputs, file_path=path, prefix="")
    if isinstance(outputs, list):
        for output in outputs:
            builder.Output(output, output.name)
    else:
        builder.Output(outputs, outputs.name)
    return builder


def from_test_case(
    path: str | Path,
    onnx_name: str = "model.onnx",
    test_case_name: str = "test_data_set_0",
) -> Builder:
    path = Path(path)
    input_values = onnx_builder.util.load_inputs_from_test_case(
        path, test_case_name, onnx_name=onnx_name
    )
    model = onnx.load(path / onnx_name)
    builder = Builder()
    inputs = []
    initializer_names = set(x.name for x in model.graph.initializer)
    for input_ in model.graph.input:
        if input_.name not in initializer_names:
            inputs.append(builder.Input(input_values[input_.name], input_.name))
    outputs = builder.Model(*inputs, file_path=str(path / onnx_name), prefix="")
    if isinstance(outputs, list):
        for output in outputs:
            builder.Output(output, output.name)
    else:
        builder.Output(outputs, outputs.name)
    return builder
