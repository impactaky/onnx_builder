import onnx


def diff_onnx(expect_onnx, actual_onnx):
    expect = onnx.load(expect_onnx)
    actual = onnx.load(actual_onnx)

    assert (
        expect.graph.input == actual.graph.input
    ), "{}\n------------------------\n{}".format(expect.graph.input, actual.graph.input)
    # assert (
    #     expect.graph.initializer == actual.graph.initializer
    # ), "{}\n------------------------\n{}".format(
    #     expect.graph.initializer, actual.graph.initializer
    # )
    assert (
        expect.graph.output == actual.graph.output
    ), "{}\n------------------------\n{}".format(
        expect.graph.output, actual.graph.output
    )

    for field in onnx.ModelProto.DESCRIPTOR.fields:
        if field.name in ["graph", "producer_name", "producer_version"]:
            continue
        orig_attr = getattr(expect, field.name)
        exported_attr = getattr(actual, field.name)
        if field.name == "opset_import":
            for opset_import in orig_attr:
                if not opset_import.domain:
                    opset_import.domain = ""
            for opset_import in exported_attr:
                if not opset_import.domain:
                    opset_import.domain = ""
        assert orig_attr == exported_attr, "{}\n------------------------\n{}".format(
            orig_attr, exported_attr
        )


if __name__ == "__main__":
    import sys

    diff_onnx(sys.argv[1], sys.argv[2])
