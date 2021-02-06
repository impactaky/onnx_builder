import glob
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest

import onnx_builder
import diff_onnx

import sys

onnx_files = glob.glob("test/onnx/**/model.onnx", recursive=True)
onnx_files = [f for f in onnx_files if "BFLOAT16" not in f]


@pytest.mark.parametrize("onnx_file", onnx_files)
def test_from_onnx(onnx_file):
    onnx_file = Path(onnx_file)
    work_dir = Path("test/from_onnx") / onnx_file.parent.name
    generator = onnx_builder.CodeGenerator()
    generator.generate(onnx_file, work_dir)
    exec(
        open(work_dir / "exporter.py").read(), {"sys.path": sys.path + [str(work_dir)]}
    )
    diff_onnx.diff_onnx(onnx_file, work_dir / "exported" / "model.onnx")


onnx_runtime_fail_list = [
    "test_gradient_of_add_and_mul",
    "test_gradient_of_add",
    "test_bitshift_left_uint16",
    "test_adagrad",
]
test_cases = onnx_files
test_cases = [
    Path(f).parent
    for f in test_cases
    if Path(f).parent.name not in onnx_runtime_fail_list
]


@pytest.mark.parametrize("test_case", test_cases)
def test_from_test_case(test_case):
    test_dir = Path(test_case)
    work_dir = Path("test/from_testcase") / test_case.name
    generator = onnx_builder.CodeGenerator()
    generator.generate(test_dir, work_dir)
    exec(
        open(work_dir / "exporter.py").read(), {"sys.path": sys.path + [str(work_dir)]}
    )
    orig_outputs = onnx_builder.util.load_outputs_from_test_case(test_dir)
    exported_outputs = onnx_builder.util.load_outputs_from_test_case(
        work_dir / "exported"
    )
    for k, v in orig_outputs.items():
        v2 = exported_outputs[k]
        try:
            assert np.array_equal(v, v2)
        except:
            assert np.allclose(v, v2)
