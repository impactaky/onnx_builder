from pathlib import Path
import diff_onnx
import glob
import numpy as np
import onnx
import onnx_builder
import pytest
import re
import sys

onnx_files = glob.glob(f"{onnx.__path__[0]}/backend/test/**/model.onnx", recursive=True)

numpy_unsupported_dtypes = [
    "BFLOAT16",
    "FLOAT8E4M3FN",
    "FLOAT8E4M3FNUZ",
    "FLOAT8E5M2",
    "FLOAT8E5M2FNUZ",
    "INT4",
    "e4m3fn",
    "e5m2",
    "int4",
]


def numpy_unsupported(case_name: str) -> bool:
    return any(dtype in case_name for dtype in numpy_unsupported_dtypes)


@pytest.mark.parametrize("onnx_file", onnx_files)
def test_from_onnx(onnx_file: str) -> None:
    if numpy_unsupported(onnx_file):
        pytest.skip("Contain numpy unsupporetd dtype")
    onnx_path = Path(onnx_file)
    work_dir = Path("test/from_onnx") / onnx_path.parent.name
    generator = onnx_builder.CodeGenerator()
    generator.generate(onnx_path, str(work_dir))
    exec(
        open(work_dir / "exporter.py").read(), {"sys.path": sys.path + [str(work_dir)]}
    )
    diff_onnx.diff_onnx(onnx_path, work_dir / "exported" / "model.onnx")


onnxruntime_fail_pattern = re.compile(
    "".join(
        (
            "deform_conv"  # deform_conv is not supported in onnxruntime
            "|dft"  # Max absolute difference > atol=1e-07. shall be able to set atol (https://github.com/onnx/onnx/issues/5897)
            "|group_normalization"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|identity_opt"  # fixed in ort 1.18 (https://github.com/microsoft/onnxruntime/pull/19273)
            "|image_decoder"  # image_decoder is not supported in onnxruntime
            "|optional_get_element_optional_sequence"  # fixed in ort 1.18 (https://github.com/microsoft/onnxruntime/pull/19273)
            "|qlinearmatmul_2D_int8"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_2D_uint8_float16"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_int8"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_uint8_float16"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_2D_uint8_float32"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_uint8_float32"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|tree_ensemble"  # tree_ensemble not yet implemented in ort
        )
    )
)

onnxruntime_fail_list = [
    "test_adagrad",
    "test_adagrad_multiple",
    "test_adam",
    "test_adam_multiple",
    "test_add_uint8",
    "test_ai_onnx_ml_tree_ensemble_set_membership",
    "test_ai_onnx_ml_tree_ensemble_single_tree",
    "test_basic_deform_conv_with_padding",
    "test_basic_deform_conv_without_padding",
    "test_bernoulli",
    "test_bernoulli_double",
    "test_bernoulli_double_expanded",
    "test_bernoulli_expanded",
    "test_bernoulli_seed",
    "test_bernoulli_seed_expanded",
    "test_bitshift_left_uint16",
    "test_bitshift_right_uint16",
    "test_cast_BFLOAT16_to_FLOAT",
    "test_cast_FLOAT16_to_FLOAT8E4M3FNUZ",
    "test_cast_FLOAT16_to_FLOAT8E4M3FN",
    "test_cast_FLOAT16_to_FLOAT8E5M2FNUZ",
    "test_cast_FLOAT16_to_FLOAT8E5M2",
    "test_cast_FLOAT16_to_INT4",
    "test_cast_FLOAT16_to_UINT4",
    "test_cast_FLOAT8E4M3FNUZ_to_FLOAT16",
    "test_cast_FLOAT8E4M3FNUZ_to_FLOAT",
    "test_cast_FLOAT8E4M3FN_to_FLOAT16",
    "test_cast_FLOAT8E4M3FN_to_FLOAT",
    "test_cast_FLOAT8E5M2FNUZ_to_FLOAT16",
    "test_cast_FLOAT8E5M2FNUZ_to_FLOAT",
    "test_cast_FLOAT8E5M2_to_FLOAT16",
    "test_cast_FLOAT8E5M2_to_FLOAT",
    "test_cast_FLOAT_to_BFLOAT16",
    "test_cast_FLOAT_to_FLOAT8E4M3FNUZ",
    "test_cast_FLOAT_to_FLOAT8E4M3FN",
    "test_cast_FLOAT_to_FLOAT8E5M2FNUZ",
    "test_cast_FLOAT_to_FLOAT8E5M2",
    "test_cast_FLOAT_to_INT4",
    "test_cast_FLOAT_to_STRING",
    "test_cast_FLOAT_to_UINT4",
    "test_cast_INT4_to_FLOAT16",
    "test_cast_INT4_to_FLOAT",
    "test_cast_INT4_to_INT8",
    "test_cast_UINT4_to_FLOAT16",
    "test_cast_UINT4_to_FLOAT",
    "test_cast_UINT4_to_UINT8",
    "test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FNUZ",
    "test_cast_no_saturate_FLOAT16_to_FLOAT8E4M3FN",
    "test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2FNUZ",
    "test_cast_no_saturate_FLOAT16_to_FLOAT8E5M2",
    "test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FNUZ",
    "test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN",
    "test_cast_no_saturate_FLOAT_to_FLOAT8E5M2FNUZ",
    "test_cast_no_saturate_FLOAT_to_FLOAT8E5M2",
    "test_castlike_BFLOAT16_to_FLOAT",
    "test_castlike_BFLOAT16_to_FLOAT_expanded",
    "test_castlike_FLOAT8E4M3FNUZ_to_FLOAT",
    "test_castlike_FLOAT8E4M3FNUZ_to_FLOAT_expanded",
    "test_castlike_FLOAT8E4M3FN_to_FLOAT",
    "test_castlike_FLOAT8E4M3FN_to_FLOAT_expanded",
    "test_castlike_FLOAT8E5M2FNUZ_to_FLOAT",
    "test_castlike_FLOAT8E5M2FNUZ_to_FLOAT_expanded",
    "test_castlike_FLOAT8E5M2_to_FLOAT",
    "test_castlike_FLOAT8E5M2_to_FLOAT_expanded",
    "test_castlike_FLOAT_to_BFLOAT16",
    "test_castlike_FLOAT_to_BFLOAT16_expanded",
    "test_castlike_FLOAT_to_FLOAT8E4M3FNUZ",
    "test_castlike_FLOAT_to_FLOAT8E4M3FNUZ_expanded",
    "test_castlike_FLOAT_to_FLOAT8E4M3FN",
    "test_castlike_FLOAT_to_FLOAT8E4M3FN_expanded",
    "test_castlike_FLOAT_to_FLOAT8E5M2FNUZ",
    "test_castlike_FLOAT_to_FLOAT8E5M2FNUZ_expanded",
    "test_castlike_FLOAT_to_FLOAT8E5M2",
    "test_castlike_FLOAT_to_FLOAT8E5M2_expanded",
    "test_castlike_FLOAT_to_STRING",
    "test_castlike_FLOAT_to_STRING_expanded",
    "test_clip_default_int8_max_expanded",
    "test_clip_default_int8_min_expanded",
    "test_deform_conv_with_mask_bias",
    "test_deform_conv_with_multiple_offset_groups",
    "test_dequantizelinear_blocked",
    "test_dequantizelinear_e4m3fn",
    "test_dequantizelinear_e4m3fn_float16",
    "test_dequantizelinear_e4m3fn_zero_point",
    "test_dequantizelinear_e5m2",
    "test_dequantizelinear_int4",
    "test_dequantizelinear_uint4",
    "test_dft_axis",
    "test_dft_axis_opset19",
    "test_dft",
    "test_dft_inverse",
    "test_dft_inverse_opset19",
    "test_dft_opset19",
    "test_div_uint8",
    "test_gru_batchwise",
    "test_image_decoder_decode_bmp_rgb",
    "test_image_decoder_decode_jpeg2k_rgb",
    "test_image_decoder_decode_jpeg_bgr",
    "test_image_decoder_decode_jpeg_grayscale",
    "test_image_decoder_decode_jpeg_rgb",
    "test_image_decoder_decode_png_rgb",
    "test_image_decoder_decode_pnm_rgb",
    "test_image_decoder_decode_tiff_rgb",
    "test_image_decoder_decode_webp_rgb",
    "test_loop16_seq_none",
    "test_lstm_batchwise",
    "test_max_int16",
    "test_max_int8",
    "test_max_uint16",
    "test_max_uint8",
    "test_maxpool_2d_ceil_output_size_reduce_by_one",
    "test_maxunpool_export_with_output_shape",
    "test_min_int16",
    "test_min_int8",
    "test_min_uint16",
    "test_min_uint8",
    "test_momentum",
    "test_momentum_multiple",
    "test_mul_uint8",
    "test_nesterov_momentum",
    "test_pow_types_float32_uint32",
    "test_pow_types_float32_uint64",
    "test_qlinearmatmul_2D_int8_float16",
    "test_qlinearmatmul_2D_int8_float32",
    "test_qlinearmatmul_2D_uint8_float16",
    "test_qlinearmatmul_2D_uint8_float32",
    "test_qlinearmatmul_3D_int8_float16",
    "test_qlinearmatmul_3D_int8_float32",
    "test_qlinearmatmul_3D_uint8_float16",
    "test_qlinearmatmul_3D_uint8_float32",
    "test_quantizelinear_blocked_asymmetric",
    "test_quantizelinear_blocked_symmetric",
    "test_quantizelinear_e4m3fn",
    "test_quantizelinear_e5m2",
    "test_quantizelinear_int4",
    "test_quantizelinear_uint4",
    "test_resize_downsample_scales_cubic_align_corners",
    "test_resize_downsample_scales_linear_align_corners",
    "test_simple_rnn_batchwise",
    "test_stft",
    "test_stft_with_window",
    "test_sub_uint8",
    "test_training_dropout",
    "test_training_dropout_default",
    "test_training_dropout_default_mask",
    "test_training_dropout_mask",
    "test_AvgPool1d",
    "test_AvgPool1d_stride",
    "test_AvgPool2d",
    "test_AvgPool2d_stride",
    "test_AvgPool3d",
    "test_AvgPool3d_stride1_pad0_gpu_input",
    "test_AvgPool3d_stride",
    "test_BatchNorm1d_3d_input_eval",
    "test_BatchNorm2d_eval",
    "test_BatchNorm2d_momentum_eval",
    "test_BatchNorm3d_eval",
    "test_BatchNorm3d_momentum_eval",
    "test_GLU",
    "test_GLU_dim",
    "test_Linear",
    "test_PReLU_1d",
    "test_PReLU_1d_multiparam",
    "test_PReLU_2d",
    "test_PReLU_2d_multiparam",
    "test_PReLU_3d",
    "test_PReLU_3d_multiparam",
    "test_PoissonNLLLLoss_no_reduce",
    "test_Softsign",
    "test_operator_add_broadcast",
    "test_operator_add_size1_broadcast",
    "test_operator_add_size1_right_broadcast",
    "test_operator_add_size1_singleton_broadcast",
    "test_operator_addconstant",
    "test_operator_addmm",
    "test_operator_basic",
    "test_operator_mm",
    "test_operator_non_float_params",
    "test_operator_params",
    "test_operator_pow",
    "test_gradient_of_add_and_mul",
    "test_gradient_of_add",
]
test_cases = onnx_files
test_cases = [
    str(Path(f).parent)
    for f in test_cases
    if Path(f).parent.name not in onnxruntime_fail_list
]


@pytest.mark.parametrize("test_case", test_cases)
def test_from_test_case(test_case: str) -> None:
    if numpy_unsupported(str(test_case)):
        pytest.skip("Contain numpy unsupporetd dtype")
    if onnx.load(Path(test_case) / "model.onnx").opset_import[0].version < 7:
        pytest.skip("opset version < 7")
    test_dir = Path(test_case)
    work_dir = Path("test/from_testcase") / test_dir.name
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
            assert np.array_equal(v, v2, equal_nan=True)
        except AssertionError:
            assert np.allclose(v, v2, atol=1.0e-4, rtol=1.0e-6, equal_nan=True)
