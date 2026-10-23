"""
Automatic tests for the DequantizeLinear operator (SONNX)
"DequantizeLinear (int32, float)":
    X            : int32 tensor
    x_scale      : float scalar
    x_zero_point : int32 scalar
    Y            : float tensor                  
    Y[i] = float( X[i] -(i32) x_zero_point ) * x_scale

    Per-tensor dequantization only
    The attributes (axis, block_size, output_dtype) are not set: they take their default value.

Two categories of tests, reported separately:
    - num    : numerical tests (values).
    - struct : structural tests (shapes).
"""
import itertools
import argparse

import numpy as np

import onnx
import onnx.checker
from onnx import helper, TensorProto
from onnxruntime import InferenceSession

# Opset 25
OPSET = 25

# Equivalence classes for the shape of x:
#   - Ranks: r_scalar;r_min;]r_min,r_sufficient[;r_sufficient = 0;1;]1,4[;4
#   - Dims: dTi_no_element;dTi_min;]dTi_min,dTi_sufficient[;dTi_sufficient = 0;1;]1,3[;3
#   - Combination of ranks and dims, with choice within the open intervals:
#      - {(0,-),(1,{0;1;2;3}),(2,{0;1;]1,3[;3}X{0;1;]1,3[;3}),(4,{0;1;]1,3[;3}X{0;1;]1,3[;3}X{0;1;]1,3[;3}X{0;1;]1,3[;3})}

# Equivalence classes for x_scale (float scalar, no constraint in the specification):
#   -maxfloat;]-maxfloat,-tiny[;-tiny;-0;0;tiny;]tiny,maxfloat[;maxfloat;+inf;-inf;NaN
#      - selected values: -max_32; {-1.0 ; -0.5 ; -3.7}; -tiny_32; -0.0; 0.0; tiny_32; {1.0 ; 0.5 ; 3.7}; max_32; +inf; -inf; NaN

# Equivalence classes for x_zero_point (int32 scalar, always present):
#   - minint;]minint,0[;0;]0,maxint[;maxint  => selected values: -2^31; -100; 0; 100; 2^31-1

# Equivalence classes for elements of x (int32):
#   - minint;]minint,0[;0;]0,maxint[;maxint
#   - completed by the singular points of the int32 -> float32 conversion: the integers up to 2^24
#     are exactly representable, 2^24+1 is not (and -2^24-1), minint+1 and maxint-1 are rounded.
#   - selected values: -2^31; -2^31+1; -2^24-1; -1; 0; 1; 2^24+1; 2^31-2; 2^31-1
#
# Two families of tests:
#      - Tests of the numerical aspects
#      - Tests of the structural aspects

MAX_INT32 = 2 ** 31 - 1
MIN_INT32 = -2 ** 31

info_32 = np.finfo(np.float32)
max_32 = info_32.max    # 3.40e+38
tiny_32 = info_32.tiny  # 1.17e-38

T1 = TensorProto.INT32
T2 = TensorProto.FLOAT
T3 = TensorProto.FLOAT

# Elements of x
X_VALUES = [MIN_INT32, MIN_INT32 + 1, -2 ** 24 - 1, -1, 0, 1, 2 ** 24 + 1, MAX_INT32 - 1, MAX_INT32]

SCALE_SETS = [
    [-max_32, -1.0, -tiny_32, tiny_32, 1.0, max_32],
    [-max_32, -0.5, -tiny_32, tiny_32, 0.5, max_32],
    [-max_32, -3.7, -tiny_32, tiny_32, 3.7, max_32],
    [np.inf, -np.inf, np.nan, 0.0, -0.0],
]
ZP_VALUES = [MIN_INT32, -100, 0, 100, MAX_INT32]

counters = {"num": [0, 0], "struct": [0, 0]}
VERBOSE = False
WANTED = {"num", "struct"}


def i32(v, shape=()):
    return np.full(shape, v, dtype=np.int32)


def f32(v, shape=()):
    return np.full(shape, v, dtype=np.float32)


fmt = lambda t: np.array2string(t, separator=",", max_line_width=np.inf)


# ---------------------------------------------------------------------------
# Reference implementation 
# ---------------------------------------------------------------------------
def DequantizeLinear(x, x_scale, x_zp):
    """
    Reference of DequantizeLinear (int32 -> float32), per-tensor: x_scale and x_zp are scalars.
    Y = float32(x -(i32) x_zp) * x_scale
    """
    d = np.asarray(x, dtype=np.int32).astype(np.int64) - np.int64(x_zp)   # exact difference
    d = ((d - MIN_INT32) % (2 ** 32)) + MIN_INT32                         # wrap-around in the int32 range
    with np.errstate(all="ignore"):
        y = np.asarray(d).astype(np.float32) * np.float32(x_scale)        # float32 cast, then IEEE multiplication
    return np.asarray(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------
def make_and_execute_session(x, x_scale, x_zp, ref_y):
    inputs_info = [
        helper.make_tensor_value_info("x", T1, x.shape),
        helper.make_tensor_value_info("x_scale", T2, x_scale.shape),
        helper.make_tensor_value_info("x_zero_point", T1, x_zp.shape),
    ]
    feeds = {"x": x, "x_scale": x_scale, "x_zero_point": x_zp}
    y_info = helper.make_tensor_value_info("y", T3, ref_y.shape)

    node = helper.make_node("DequantizeLinear", inputs=["x", "x_scale", "x_zero_point"], outputs=["y"])

    graph_def = helper.make_graph([node], "Test_DequantizeLinear", inputs_info, [y_info])
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def check_constraints(x, x_scale, x_zp, y):
    # X: int32 tensor
    assert x.dtype == np.int32
    # x_scale: float scalar
    assert x_scale.dtype == np.float32 and x_scale.ndim == 0
    # x_zero_point: int32 scalar
    assert x_zp.dtype == np.int32 and x_zp.ndim == 0
    # Y: same shape as X, float
    assert y.dtype == np.float32 and y.shape == x.shape


def run_one_test(category, label, x, x_scale, x_zp, expected=None):
    if category not in WANTED:
        return
    ref_y = DequantizeLinear(x, x_scale, x_zp)
    if expected is not None:
        assert np.allclose(ref_y, expected, rtol=1e-6, atol=0.0, equal_nan=True), (ref_y, expected)
    check_constraints(x, x_scale, x_zp, ref_y)

    counters[category][0] += 1
    try:
        y = make_and_execute_session(x, x_scale, x_zp, ref_y)
        ok = y.dtype == np.float32 and y.shape == ref_y.shape and \
            np.allclose(y, ref_y, rtol=1.2e-07, atol=0.0, equal_nan=True)
    except Exception as e:
        y, ok = None, False
        print(f"   ERROR: {str(e)[:200]}")
    counters[category][1] += 0 if ok else 1

    print(f"[{category}] {label} -> {'OK' if ok else 'KO'}")
    if VERBOSE or not ok:
        print(f"   x {x.shape} = {fmt(x)}")
        print(f"   x_scale = {fmt(x_scale)}   x_zero_point = {fmt(x_zp)}")
        print(f"   ONNX Y = {None if y is None else fmt(y)}")
        print(f"   ref  Y = {fmt(ref_y)}")


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------

def test_DequantizeLinear_num_one_elt(x_values, scale_values):
    # x (1,1,1,1), x_scale and x_zero_point scalars. All the combinations are tested.
    print("\ntest_DequantizeLinear_num_one_elt:\n")
    for val_x, val_s, val_zp in itertools.product(x_values, scale_values, ZP_VALUES):
        run_one_test("num", f"one_elt x={val_x} scale={val_s} zp={val_zp}",
                     i32(val_x, (1, 1, 1, 1)), f32(val_s), i32(val_zp))


def test_DequantizeLinear_num_all_x_in_one_tensor(x_values, scale_values):
    # All the x values in one single execution, for each (x_scale, x_zero_point): x (1,1,1,n)
    print("\nAll x values in one tensor")
    x = np.array(x_values, dtype=np.int32).reshape(1, 1, 1, len(x_values))
    for val_s, val_zp in itertools.product(scale_values, ZP_VALUES):
        run_one_test("num", f"all x in one tensor scale={val_s} zp={val_zp}", x, f32(val_s), i32(val_zp))


def test_DequantizeLinear_num():
    print("\nNumerical tests\n")
    for num_test_set, scale_values in enumerate(SCALE_SETS, start=1):
        print("num_test_set = ", num_test_set)
        test_DequantizeLinear_num_one_elt(X_VALUES, scale_values)
        test_DequantizeLinear_num_all_x_in_one_tensor(X_VALUES, scale_values)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------
def test_DequantizeLinear_struct():
    # Shapes of x:
    #   - rank 0: ()                                   => 1 shape
    #   - rank 1: (d0), d0 in {0,1,2,3}                => 4 shapes
    #   - rank 2: (d0,d1), d in {0,1,2,3}              => 16 shapes
    #   - rank 4: (d0,d1,d2,d3), d in {0,1,2,3}        => 256 shapes
    # x_scale = 1.7 and x_zero_point random (scalars), for every shape.
    print("\nStructural tests\n")
    rng = np.random.default_rng(1234)

    shapes = [()]
    for rank in (1, 2, 4):
        shapes += list(itertools.product(range(4), repeat=rank))

    for shape in shapes:
        x = rng.integers(-1000, 1001, size=shape).astype(np.int32)
        run_one_test("struct", f"rank {len(shape)} dims {shape}", x, f32(1.7), i32(rng.integers(-1000, 1001)))


parser = argparse.ArgumentParser(description="Tests of the ONNX DequantizeLinear operator (int32 -> float32), per-tensor, adapted to the specification.")
parser.add_argument("--tests", choices=["num", "struct", "all"], default="all",
                    help="num: numerical tests; struct: structural tests; all")
parser.add_argument("--verbose", action="store_true", help="Print the tensors of every test (always done for KO)")
args = parser.parse_args()
VERBOSE = args.verbose

WANTED = {"num", "struct"} if args.tests == "all" else {args.tests}
if "num" in WANTED:
    test_DequantizeLinear_num()
if "struct" in WANTED:
    test_DequantizeLinear_struct()

print("\nSummary:")
labels = {"num": "numerical", "struct": "structural"}
for cat, (n, ko) in counters.items():
    if n:
        print(f"  {labels[cat]:26s}: {n:5d} tests, {n - ko:5d} OK, {ko} KO")