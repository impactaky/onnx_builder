import glob
from pathlib import Path

import onnx
import onnx_builder

import pytest

onnx_files = glob.glob("test/onnx/**/model.onnx", recursive=True)


@pytest.mark.parametrize("onnx_file", onnx_files)
def test_function(onnx_file):
    print(onnx_file)
    onnx_file = Path(onnx_file)
    work_dir = Path("test/work") / onnx_file.parent.name
    generator = onnx_builder.CodeGenerator()
    generator.generate(onnx_file, work_dir)
    exec(open(work_dir / "exporter.py").read())

    orig_model = onnx.load(onnx_file)
    exported_model = onnx.load(work_dir / "exported" / "model.onnx")

    assert orig_model.graph.input == exported_model.graph.input
    assert orig_model.graph.initializer == exported_model.graph.initializer
    assert orig_model.graph.output == exported_model.graph.output

    for field in onnx.ModelProto.DESCRIPTOR.fields:
        if field.name in ["graph", "producer_name", "producer_version"]:
            continue
        orig_attr = getattr(orig_model, field.name)
        exported_attr = getattr(exported_model, field.name)
        assert orig_attr == exported_attr
