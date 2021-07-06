from pathlib import Path
import onnx
from onnx import numpy_helper
import numpy as np
import onnx_builder.util
from onnx_builder.value import Value


def _eval_with_onnxruntime(model, inputs, output_names):
    import onnxruntime

    session = onnxruntime.InferenceSession(model.SerializeToString())
    return session.run(output_names, inputs)


class Builder:
    def __init__(
        self,
        opset_imports=None,
        eval_each_node=False,
        value_prefix="onnx_builder_tmp",
        eval_func=_eval_with_onnxruntime,
    ):
        self.opset_imports = opset_imports
        self.eval_each_node = eval_each_node
        self.value_prefix = value_prefix
        self.__eval_func = eval_func
        self.__value_idx = 0
        self.__nodes = []
        self.__input_vis = []
        self.__output_vis = []
        self.__initializers = []
        self.__inputs = []
        self.__outputs = []

    def __GenValueName(self):
        self.__value_idx += 1
        return self.value_prefix + "_" + str(self.__value_idx)

    def Initializer(self, value, name=""):
        if not name:
            name = self.__GenValueName()
        self.__inputs.append(value)
        ret = Value(name, value)
        self.__input_vis.append(ret.value_info())
        self.__initializers.append(ret.proto())
        return ret

    def Input(
        self, value=None, name="", shape=None, dtype=None, value_type="tensor_type"
    ):
        if value_type == "sequence_type" and value is None:
            value = []
        if not name:
            name = self.__GenValueName()
        ret = Value(name, value, shape=shape, dtype=dtype)
        if value is not None:
            self.__inputs.append(ret.value)
        self.__input_vis.append(ret.value_info())
        return ret

    def InputSequence(self, list_=[], name="", shape=None, dtype=None):
        return self.Input(
            list_, name=name, shape=shape, dtype=dtype, value_type="sequence_type"
        )

    def Output(self, value, name="", shape=None, dtype=None, value_type="tensor_type"):
        if value_type == "sequence_type" and value.value is None:
            value.value = []
        if name:
            for node in self.__nodes:
                if value.name in node.output:
                    index = list(node.output).index(value.name)
                    node.output[index] = name
                    break
            value.name = name
        if shape is not None:
            if value.shape:
                assert shape != value.shape
            value.shape = shape
        if dtype is not None:
            if value.dtype:
                assert dtype != value.dtype
            value.dtype = dtype
        self.__output_vis.append(value.value_info())
        self.__outputs.append(value)
        return self.__outputs[-1]

    def OutputSequence(self, value, name="", shape=None, dtype=None):
        return self.Output(
            value, name=name, shape=shape, dtype=dtype, value_type="sequence_type"
        )

    def Model(self, *args, file_path="", prefix="", **kwargs):
        if not prefix:
            prefix = file_path+"_"

        model = onnx.load(file_path)
        self.opset_imports = getattr(model, "opset_import")

        value_table = {}
        args_index = 0
        for input_ in model.graph.input:
            if input_.name in kwargs:
                value_table[input_.name] = kwargs[input_.name].name
            else:
                value_table[input_.name] = args[args_index].name
                args_index += 1

        def resolve_value_names(names):
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
            self.__nodes.append(node)

        for initializer in model.graph.initializer:
            initializer.name = prefix + initializer.name
            self.__initializers.append(initializer)

        outputs = [Value(prefix + output.name) for output in model.graph.output]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def make_graph(self, name=""):
        if not name:
            name = self.__GenValueName()
        return onnx.helper.make_graph(
            self.__nodes,
            name,
            inputs=self.__input_vis,
            outputs=self.__output_vis,
            initializer=self.__initializers,
        )

    def build(self, graph_name="model.root", **kwargs):
        if 'opset_imports' not in kwargs or kwargs['opset_imports'] is None:
            kwargs['opset_imports'] = self.opset_imports
        graph = self.make_graph(name=graph_name)
        model = onnx.helper.make_model(
            graph, producer_name="onnx_builder", producer_version="0.01", **kwargs
        )
        # model = onnx.shape_inference.infer_shapes(model)
        return model

    def eval(self, **kwargs):
        model = self.build(**kwargs)
        input_names = [vi.name for vi in self.__input_vis]
        inputs = dict(zip(input_names, self.__inputs))
        output_names = [vi.name for vi in self.__output_vis]
        outputs = self.__eval_func(model, inputs, output_names)
        for name, value in zip(output_names, outputs):
            for holder in self.__outputs:
                if holder.name == name:
                    holder.value = value
        return (model, outputs)

    def export(self, output_dir, **kwargs):
        model, outputs = self.eval(**kwargs)
        output_dir = Path(output_dir)
        (output_dir / "test_data_set_0").mkdir(parents=True, exist_ok=True)
        # save inputs
        initializer_names = [x.name for x in self.__initializers]
        input_names = [vi.name for vi in self.__input_vis]
        for i, input_ in enumerate(self.__inputs):
            if input_names[i] in initializer_names:
                continue
            with open(
                output_dir / "test_data_set_0" / "input_{}.pb".format(i), "wb"
            ) as f:
                f.write(Value(input_names[i], input_).proto().SerializeToString())
        # save outputs
        output_names = [vi.name for vi in self.__output_vis]
        for i, output_ in enumerate(outputs):
            with open(
                output_dir / "test_data_set_0" / "output_{}.pb".format(i), "wb"
            ) as f:
                f.write(Value(output_names[i], output_).proto().SerializeToString())
        onnx.save(model, output_dir / "model.onnx")

    def __getattr__(self, op):
        def fn(*args, outs=1, output_names=[], name=None, **kwargs):
            inputs = list(args)
            input_names = []
            for i, input_ in enumerate(inputs):
                if type(input_) == Value:
                    input_names.append(input_.name)
                    inputs[i] = input_.value
                elif input_ is None:
                    input_names.append("")
                else:
                    input_names.append(self.__GenValueName())
                    constant_tensor = numpy_helper.from_array(
                        input_, name=input_names[-1] + "_val"
                    )
                    self.__nodes.append(
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[input_names[-1]],
                            value=constant_tensor,
                        )
                    )

            if not output_names:
                output_names = [self.__GenValueName() for i in range(outs)]
            for k, v in kwargs.items():
                if type(v) == np.ndarray:
                    kwargs[k] = onnx.numpy_helper.from_array(v, self.__GenValueName())
                if k == "to" and not isinstance(v, int):
                    kwargs[k] = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(v)]
            node = onnx.helper.make_node(
                op, inputs=input_names, outputs=output_names, name=name, **kwargs
            )
            self.__nodes.append(node)

            if not self.eval_each_node:
                outputs = [Value(name) for name in output_names]
                if outs == 1:
                    return outputs[0]
                else:
                    return outputs

            input_vis = [
                Value(input_names, inputs).value_info()
                for n, a in zip(input_names, inputs)
            ]
            output_vis = [onnx_builder.util.make_value_info(n) for n in output_names]
            graph = onnx.helper.make_graph(
                [node],
                "onnx_eval.each_eval",
                inputs=input_vis,
                outputs=output_vis,
                initializer=self.__initializers,
            )
            model = onnx.helper.make_model(graph, opset_imports=self.opset_imports)

            inputs = dict(zip(input_names, inputs))
            outputs = self.__eval_func(model, inputs, output_names)
            for i, output_ in enumerate(outputs):
                outputs[i] = Value(output_names[i], output_)
            if outs == 1:
                return outputs[0]
            else:
                return outputs

        return fn
