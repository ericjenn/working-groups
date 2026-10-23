"""
Automatic tests for the QuantizeLinear operator (SONNX)
"QuantizeLinear (float, uint8)":
    X            : float tensor
    y_scale      : float scalar,  y_scale != 0                      [E_QUANTIZELINEAR_UINT8_CONSTR_YSCALE_0010]
    y_zero_point : uint8 scalar
    Y            : uint8 tensor, same shape as X                    [..._CONSTR_X_0010, ..._CONSTR_Y_0010]
    Y[i] = clamp( round_half_to_even( X[i] / y_scale ) + y_zero_point, 0, 255 )   [E_QUANTIZELINEAR_UINT8_FUNC_0010]
           (division in float32, IEEE 754)                          [R4]
    0 <= Y[i] <= 255                                                [..._CONSTR_Y_0020]
    Per-tensor quantization only
    The attributes (axis, block_size, output_dtype, precision, saturate) are not set: they take their default value.

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

# Equivalence classes for y_scale (float scalar; the specification requires y_scale != 0):
#   -maxfloat;]-maxfloat,-tiny[;-tiny;tiny;]tiny,maxfloat[;maxfloat; +inf; -inf;  NaN
#      - selected values: -max_32; {-1.0 ; -0.5 ; -3.7}; -tiny_32; tiny_32; {1.0 ; 0.5 ; 3.7}; max_32; +inf; -inf;  NaN

# Equivalence classes for y_zero_point (uint8 scalar, always present):
#   - 0;]0,255[;255  => selected values: 0; 100; 255

# Equivalence classes for elements of x:
#   NaN;;-inf;minfloat;]minfloat,-0[;-0;0;]0,maxfloat[;maxfloat;+inf
#
# Two families of tests:
#      - Tests of the numerical aspects
#      - Tests of the structural aspects

# Result expected
#   NaN -> 0 (whatever y_zero_point); +inf -> 255 and -inf -> 0 (saturation) for a positive scale.
NAN_RESULT = 0

info_32 = np.finfo(np.float32)
max_32 = info_32.max    # 3.40e+38
min_32 = info_32.min    # -3.40e+38
tiny_32 = info_32.tiny  # 1.17e-38

T1 = TensorProto.FLOAT
T2 = TensorProto.FLOAT
T3 = TensorProto.UINT8

# Elements of x 
X_VALUES = [np.nan, -np.inf, min_32, -1.0, -0.5, -0.25, -0.0, 0.0, tiny_32, 0.25, 0.5, 0.75, 1.0, 1.25,
            1.5, 2.5, 127.5, 254.5, 255.0, 255.5, max_32, np.inf]

SCALE_SETS = [
    [-max_32, -1.0, -tiny_32, tiny_32, 1.0, max_32],
    [-max_32, -0.5, -tiny_32, tiny_32, 0.5, max_32],
    [-max_32, -3.7, -tiny_32, tiny_32, 3.7, max_32],
    [np.inf, -np.inf, np.nan],
]
ZP_VALUES = [0, 100, 255]

counters = {"num": [0, 0], "struct": [0, 0]} 
VERBOSE = False
WANTED = {"num", "struct"}


def f32(v, shape=()):
    return np.full(shape, v, dtype=np.float32)


def u8(v, shape=()):
    return np.full(shape, v, dtype=np.uint8)


fmt = lambda t: np.array2string(t, separator=",", max_line_width=np.inf)


# ---------------------------------------------------------------------------
# Reference implementation   [E_QUANTIZELINEAR_UINT8_FUNC_0010]
# ---------------------------------------------------------------------------
def QuantizeLinear(x, y_scale, y_zp):
    """
    Reference of QuantizeLinear (float32 -> uint8), per-tensor: y_scale and y_zp are scalars.
    Y = clamp(round_half_to_even(x / y_scale) + y_zp, 0, 255)
    """
    x = np.asarray(x, dtype=np.float32)
    scale = np.asarray(y_scale, dtype=np.float32)
    zp = np.int64(y_zp)

    with np.errstate(all="ignore"):
        q = x / scale                     # float32 division, IEEE 754
        r = np.rint(q)                    # round to nearest, ties to even
        nan_mask = np.isnan(r)
        r = np.where(nan_mask, 0.0, r)
        r = np.clip(r, -1e9, 1e9).astype(np.int64)
        y = np.clip(r + zp, 0, 255)     
        y = np.where(nan_mask, NAN_RESULT, y)
    return np.asarray(y, dtype=np.uint8)


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------
def make_and_execute_session(x, y_scale, y_zp, ref_y):
    inputs_info = [
        helper.make_tensor_value_info("x", T1, x.shape),
        helper.make_tensor_value_info("y_scale", T2, y_scale.shape),
        helper.make_tensor_value_info("y_zero_point", T3, y_zp.shape),
    ]
    feeds = {"x": x, "y_scale": y_scale, "y_zero_point": y_zp}
    y_info = helper.make_tensor_value_info("y", T3, ref_y.shape)

    node = helper.make_node("QuantizeLinear", inputs=["x", "y_scale", "y_zero_point"], outputs=["y"])

    graph_def = helper.make_graph([node], "Test_QuantizeLinear", inputs_info, [y_info])
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def check_constraints(spec_defined, x, y_scale, y_zp, y):
    # X: float tensor
    assert x.dtype == np.float32
    # y_scale: float scalar
    assert y_scale.dtype == np.float32 and y_scale.ndim == 0
    if spec_defined:
        assert y_scale != 0 and not np.isnan(y_scale) 
    # y_zero_point: uint8 scalar
    assert y_zp.dtype == np.uint8 and y_zp.ndim == 0
    # Y: same shape as X, uint8 in [0,255]
    assert y.dtype == np.uint8 and y.shape == x.shape


def run_one_test(category, label, x, y_scale, y_zp, expected=None, spec_defined=True):
    if category not in WANTED:
        return
    ref_y = QuantizeLinear(x, y_scale, y_zp)
    if expected is not None:
        assert np.array_equal(ref_y, expected), (ref_y, expected)
    check_constraints(spec_defined, x, y_scale, y_zp, ref_y)

    if not spec_defined:
        label += " [not defined by the spec]"
    counters[category][0] += 1
    try:
        y = make_and_execute_session(x, y_scale, y_zp, ref_y)
        ok = y.dtype == np.uint8 and y.shape == ref_y.shape and np.array_equal(y, ref_y)
    except Exception as e:
        y, ok = None, False
        print(f"   ERROR: {str(e)[:200]}")
    counters[category][1] += 0 if ok else 1

    print(f"[{category}] {label} -> {'OK' if ok else 'KO'}")
    if VERBOSE or not ok:
        print(f"   x {x.shape} = {fmt(x)}")
        print(f"   y_scale = {fmt(y_scale)}   y_zero_point = {fmt(y_zp)}")
        print(f"   ONNX Y = {None if y is None else fmt(y)}")
        print(f"   ref  Y = {fmt(ref_y)}")


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------

def is_spec_case(val_x, val_s):
    return bool(np.isfinite(val_x)) and val_s != 0 and not np.isnan(val_s)


def test_QuantizeLinear_num_one_elt(x_values, scale_values):
    # x (1,1,1,1), y_scale and y_zero_point scalars. All the combinations are tested.
    print("\ntest_QuantizeLinear_num_one_elt:\n")
    for val_x, val_s, val_zp in itertools.product(x_values, scale_values, ZP_VALUES):
        run_one_test("num", f"one_elt x={val_x} scale={val_s} zp={val_zp}",
                     f32(val_x, (1, 1, 1, 1)), f32(val_s), u8(val_zp),
                     spec_defined=is_spec_case(val_x, val_s))


def test_QuantizeLinear_num_all_x_in_one_tensor(x_values, scale_values):
    # All the finite x values in one single execution, for each (y_scale, y_zero_point): x (1,1,1,n)
    print("\nAll x values in one tensor")
    x = np.array(x_values, dtype=np.float32).reshape(1, 1, 1, len(x_values))
    for val_s, val_zp in itertools.product(scale_values, ZP_VALUES):
        if is_spec_case(0.0, val_s):
            run_one_test("num", f"all x in one tensor scale={val_s} zp={val_zp}", x, f32(val_s), u8(val_zp))


def test_QuantizeLinear_num():
    print("\nNumerical tests\n")
    for num_test_set, scale_values in enumerate(SCALE_SETS, start=1):
        print("num_test_set = ", num_test_set)
        test_QuantizeLinear_num_one_elt(X_VALUES, scale_values)
        test_QuantizeLinear_num_all_x_in_one_tensor(X_VALUES, scale_values)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------
def test_QuantizeLinear_struct():
    # Shapes of x:
    #   - rank 0: ()                                   => 1 shape
    #   - rank 1: (d0), d0 in {0,1,2,3}                => 4 shapes
    #   - rank 2: (d0,d1), d in {0,1,2,3}              => 16 shapes
    #   - rank 4: (d0,d1,d2,d3), d in {0,1,2,3}        => 256 shapes
    # y_scale = 1.7 and y_zero_point random (scalars), for every shape.
    print("\nStructural tests\n")
    rng = np.random.default_rng(1234)

    shapes = [()]
    for rank in (1, 2, 4):
        shapes += list(itertools.product(range(4), repeat=rank))

    for shape in shapes:
        x = rng.uniform(-50.0, 350.0, size=shape).astype(np.float32)
        run_one_test("struct", f"rank {len(shape)} dims {shape}", x, f32(1.7), u8(rng.integers(0, 256)))


parser = argparse.ArgumentParser(description="Tests of the ONNX QuantizeLinear operator (float32 -> uint8), per-tensor, adapted to the specification.")
parser.add_argument("--tests", choices=["num", "struct", "all"], default="all",
                    help="num: numerical tests; struct: structural tests; all")
parser.add_argument("--verbose", action="store_true", help="Print the tensors of every test (always done for KO)")
args = parser.parse_args()
VERBOSE = args.verbose

WANTED = {"num", "struct"} if args.tests == "all" else {args.tests}
if "num" in WANTED:
    test_QuantizeLinear_num()
if "struct" in WANTED:
    test_QuantizeLinear_struct()

print("\nSummary:")
labels = {"num": "numerical", "struct": "structural"}
for cat, (n, ko) in counters.items():
    if n:
        print(f"  {labels[cat]:26s}: {n:5d} tests, {n - ko:5d} OK, {ko} KO")