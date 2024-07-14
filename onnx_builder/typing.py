import numpy as np
from numpy import typing as npt

ShapeType = tuple[int | str | None, ...]
DType = (
    np.float32
    | np.uint8
    | np.int8
    | np.uint16
    | np.int16
    | np.int32
    | np.int64
    | np.bool_
    | np.float16
    | np.float64
    | np.complex64
    | np.complex128
    | np.uint32
    | np.uint64
    | np.object_
)
ArrayType = npt.NDArray[DType]
