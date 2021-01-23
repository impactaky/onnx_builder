import glob
from pathlib import Path

import onnx
import onnx_builder

import pytest
import diff_onnx

import os
import sys

onnx_files = glob.glob("test/onnx/**/model.onnx", recursive=True)
sys_path = sys.path
# onnx_files = ['test/onnx/onnx/backend/test/data/node/test_range_int32_type_negative_delta_expanded/model.onnx']


@pytest.mark.parametrize("onnx_file", onnx_files)
def test_function(onnx_file):
    print(onnx_file)
    onnx_file = Path(onnx_file)
    work_dir = Path("test/work") / onnx_file.parent.name
    generator = onnx_builder.CodeGenerator()
    generator.generate(onnx_file, work_dir)
    # sys.path.append(str(work_dir))
    exec(
        open(work_dir / "exporter.py").read(), {"sys.path": sys.path + [str(work_dir)]}
    )

    orig_model = onnx.load(onnx_file)
    exported_model = onnx.load(work_dir / "exported" / "model.onnx")

    diff_onnx.diff_onnx(onnx_file, work_dir / "exported" / "model.onnx")
