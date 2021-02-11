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
    # See broken_test in
    # https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/onnx/main.cc
    "test_BERT_Squad",
    "test_constantofshape_float_ones",
    "test_constantofshape_int_zeros",
    "test_cast_STRING_to_FLOAT",
    "test_cast_FLOAT_to_STRING",
    "test_tf_nasnet_large",
    "test_tf_nasnet_mobile",
    "test_tf_pnasnet_large",
    "test_shrink",
    "test_maxpool_with_argmax_2d_precomputed_strides",
    "test_tf_inception_v2",
    "test_tf_resnet_v1_50",
    "test_tf_resnet_v1_101",
    "test_tf_resnet_v1_152",
    "test_mxnet_arcface",
    "test_unique_not_sorted_without_axis",
    "test_cumsum_1d_reverse_exclusive",
    "test_resize_downsample_scales_cubic_align_corners",
    "test_resize_downsample_scales_linear_align_corners",
    "test_resize_tf_crop_and_resize",
    "test_resize_upsample_sizes_nearest_ceil_half_pixel",
    "test_resize_upsample_sizes_nearest_floor_align_corners",
    "test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric",
    "test_bitshift_right_uint16",
    "test_bitshift_left_uint16",
    "test_maxunpool_export_with_output_shape",
    "test_training_dropout",
    "test_training_dropout_default",
    "test_training_dropout_default_mask",
    "test_training_dropout_mask",
    "test_adagrad",
    "test_adagrad_multiple",
    "test_adam",
    "test_adam_multiple",
    "test_gradient_of_add",
    "test_gradient_of_add_and_mul",
    "test_momentum",
    "test_momentum_multiple",
    "test_nesterov_momentum",
    "test_cast_FLOAT_to_BFLOAT16",
    "test_cast_BFLOAT16_to_FLOAT",
    "test_sequence_insert_at_back",
    "test_sequence_insert_at_front",
    "test_loop13_seq",
    # Other test gave the not implmented error
    # Min(13)
    "test_min_int8",
    "test_min_uint8",
    # Pow(13)
    "test_pow_types_float32_uint32",
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
