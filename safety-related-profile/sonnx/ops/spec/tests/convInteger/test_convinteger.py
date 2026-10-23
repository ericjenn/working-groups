"""
Automatic tests for the ConvInteger operator (SONNX) - 2D integer convolution
    T1 (X, x_zero_point) = uint8
    T2 (W, w_zero_point) = int8
    T3 (Y)               = int32
    Y = sum( (X - x_zero_point) * (W - w_zero_point) )
"""
import math
import itertools
import argparse

import numpy as np

import onnx
import onnx.checker
from onnx import helper, TensorProto
from onnxruntime import InferenceSession

# Equivalence classes for input tensor X (rank 4, layout N,C,H,W):
#   - Rank: 4 (2D convolution).
#   - Singular points and classes for dims: dTi_no_element;dTi_min;]dTi_min,dTi_sufficient[;dTi_sufficient;]dTi_sufficient,dTi_max[;dTi_max
#      - dTi_no_element = 0   : excluded, the constraints of ConvInteger require every dimension > 0
#      - dTi_min = 1
#      - dTi_sufficient = 5 for H and W (3 for C, 2 for N and M)
#      - dTi_max = max positive integer. Not tractable.
#   - Selected values: N in {1,2}; C in {1,3}; H and W in {1,3,5}
# Equivalence classes for W (weights), shape (M, C/group, kH, kW):
#      - M in {1,2}; C/group = C (group == 1 fixed)
# Equivalence class for auto_pad: 'NOTSET' (fixed)
# Equivalence class for group: 1 (fixed)
# Equivalence class for dilations: (1,1) (fixed)
# Equivalence classes for kernel_shape: {(1,1);(3,3)}
# Equivalence classes for pads: {(0,0,0,0);(1,1,1,1)}
# Equivalence classes for strides: {(1,1);(2,2)}
# Equivalence classes for x_zero_point (uint8, scalar) : absent;0;]0,255[;255
# Equivalence classes for w_zero_point (int8) : absent;scalar;1D of size M
#      - scalar values: -128;]-128,0[;0;]0,127[;127
# Equivalence classes for elements of X (uint8):
#      - EQCELTS_X: 0;]0,255[;255
# Equivalence classes for elements of W (int8):
#      - EQCELTS_W: -128;]-128,0[;0;]0,127[;127
#
# Two families of tests:
#      - Tests of the numerical aspects (extreme values, zero points)
#      - Tests of the structural aspects (shapes x kernel_shape x pads x strides x zero points)

# Fixed attributes
AUTO_PAD = "NOTSET"
GROUP = 1
DILATIONS = [1, 1]
T1 = TensorProto.UINT8
T2 = TensorProto.INT8
T3 = TensorProto.INT32

num_test_set = 0
nb_tests = 0
nb_ko = 0

# Values (uint8 for X, int8 for W) used by the numerical tests
X_SETS = [
    [0, 1, 127, 254, 255],
    [0, 1, 128, 254, 255],
]
W_SETS = [
    [-128, -127, -1, 0, 1, 126, 127],
    [-128, -127, -64, 0, 63, 126, 127],
]


def print_attributes(pads, kernel_shape, dilation, strides, auto_pad, group):
    print("\nAttributes:")
    print(f"pads = {pads}")
    print(f"kernel_shape = {kernel_shape}")
    print(f"dilations = {dilation}")
    print(f"strides = {strides}")
    print(f"auto_pad = {auto_pad}")
    print(f"group = {group}\n")


def u8(v, shape=()):
    return np.full(shape, v, dtype=np.uint8)


def i8(v, shape=()):
    return np.full(shape, v, dtype=np.int8)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def compute_ConvInteger_output_shape(x_shape, w_shape, pads, dilation, strides):
    dx0, dx1, dx2, dx3 = x_shape
    dw0, dw1, dw2, dw3 = w_shape

    alpha = dx2 + pads[0] + pads[2]
    beta = dx3 + pads[1] + pads[3]
    theta = dilation[0] * (dw2 - 1) + 1
    gamma = dilation[1] * (dw3 - 1) + 1

    dy2 = math.floor((alpha - theta) / strides[0]) + 1
    dy3 = math.floor((beta - gamma) / strides[1]) + 1

    return [dx0, dw0, dy2, dy3]


def ConvInteger(X, W, x_zp, w_zp, strides, dilation, pads):
    """
    Naive reference of ONNX ConvInteger (rank 4, group = 1).
    X: uint8 (N,C,H,W) ; W: int8 (M,C,kH,kW)
    x_zp: None or uint8 scalar ; w_zp: None, int8 scalar or int8 1D of size M
    pads = (top, left, bottom, right)
    result in int32.
    """
    N, C, H, Wd = X.shape
    M, _, kH, kW = W.shape
    s_h, s_w = strides
    d_h, d_w = dilation
    p_top, p_left, _, _ = pads

    xz = 0 if x_zp is None else int(x_zp)
    if w_zp is None:
        wz = [0] * M
    elif np.ndim(w_zp) == 0:
        wz = [int(w_zp)] * M
    else:
        wz = [int(v) for v in w_zp]

    y_shape = compute_ConvInteger_output_shape(X.shape, W.shape, pads, dilation, strides)
    Y = np.zeros(y_shape, dtype=np.int32)

    for n in range(N):
        for m in range(M):
            for i in range(y_shape[2]):
                for j in range(y_shape[3]):
                    acc = 0
                    for c in range(C):
                        for u in range(kH):
                            for v in range(kW):
                                h = i * s_h + u * d_h - p_top
                                w = j * s_w + v * d_w - p_left
                                if 0 <= h < H and 0 <= w < Wd:
                                    acc += (int(X[n, c, h, w]) - xz) * (int(W[m, c, u, v]) - wz[m])
                    Y[n, m, i, j] = acc
    return Y


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------
def make_and_execute_session(auto_pad, dilation, group, k_shape, pads, strides, x, w, x_zp, w_zp, ref_y):

    inputs_names = ["X", "W"]
    inputs_info = [
        helper.make_tensor_value_info("X", T1, x.shape),
        helper.make_tensor_value_info("W", T2, w.shape),
    ]
    feeds = {"X": x, "W": w}

    if x_zp is not None or w_zp is not None:
        if x_zp is not None:
            inputs_names.append("x_zero_point")
            inputs_info.append(helper.make_tensor_value_info("x_zero_point", T1, x_zp.shape))
            feeds["x_zero_point"] = x_zp
        else:
            inputs_names.append("")
        if w_zp is not None:
            inputs_names.append("w_zero_point")
            inputs_info.append(helper.make_tensor_value_info("w_zero_point", T2, w_zp.shape))
            feeds["w_zero_point"] = w_zp

    y_info = helper.make_tensor_value_info("Y", T3, ref_y.shape)

    node = helper.make_node(
        "ConvInteger",
        inputs=inputs_names,
        outputs=["Y"],
        auto_pad=auto_pad,
        dilations=dilation,
        group=group,
        kernel_shape=k_shape,
        pads=pads,
        strides=strides,
    )

    graph_def = helper.make_graph([node], "Test_ConvInteger", inputs_info, [y_info])
    model = helper.make_model(graph_def, opset_imports=[helper.make_opsetid("", 22)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    y = session.run(None, feeds)[0]

    fmt = lambda t: np.array2string(t, separator=",", max_line_width=np.inf)
    print("X shape:", x.shape)
    print("X =", fmt(x))
    print("W shape:", w.shape)
    print("W =", fmt(w))
    if x_zp is not None:
        print("x_zero_point =", fmt(x_zp))
    if w_zp is not None:
        print("w_zero_point =", fmt(w_zp))
    print("Y shape:", y.shape)
    print("ONNX Y =", fmt(y))
    print("ref_y =", fmt(ref_y))

    return y


def check_test_result_ConvInteger(Y, refY):
    global nb_tests, nb_ko
    nb_tests += 1
    print("\ncheck_test_result_ConvInteger\n")
    fmt = lambda t: np.array2string(t, separator=",", max_line_width=np.inf)
    print(f"Y = {fmt(Y)}")
    print(f"refY = {fmt(refY)}")
    # Exact comparison
    if Y.dtype == np.int32 and Y.shape == refY.shape and np.array_equal(Y, refY):
        print("-> Test of ConvInteger is OK\n")
    else:
        nb_ko += 1
        print("-> Test of ConvInteger is KO\n")


def check_constraints(x, w, x_zp, w_zp, auto_pad, group, kernel_shape, pads, strides, dilation, y):
    # x (T1)
    assert x.dtype == np.uint8 and x.ndim == 4 and all(d > 0 for d in x.shape)
    # w (T2)
    assert w.dtype == np.int8 and w.ndim == 4 and all(d > 0 for d in w.shape)
    assert list(w.shape[2:]) == list(kernel_shape)
    assert w.shape[1] == x.shape[1] // group
    # zero points
    if x_zp is not None:
        assert x_zp.dtype == np.uint8 and x_zp.ndim == 0
    if w_zp is not None:
        assert w_zp.dtype == np.int8
        assert w_zp.ndim == 0 or w_zp.shape == (w.shape[0],)
    # attributes
    assert auto_pad == "NOTSET" and group == 1
    assert len(kernel_shape) == 2 and all(k > 0 for k in kernel_shape)
    assert len(strides) == 2 and all(s > 0 for s in strides)
    assert len(dilation) == 2 and all(d > 0 for d in dilation)
    assert len(pads) == 4 and all(p >= 0 for p in pads)
    # y (T3)
    assert y.dtype == np.int32
    assert list(y.shape) == compute_ConvInteger_output_shape(x.shape, w.shape, pads, dilation, strides)


def run_one_test(k_shape, pads, strides, x, w, x_zp=None, w_zp=None):
    ref_y = ConvInteger(x, w, x_zp, w_zp, strides, DILATIONS, pads)
    check_constraints(x, w, x_zp, w_zp, AUTO_PAD, GROUP, k_shape, pads, strides, DILATIONS, ref_y)
    y = make_and_execute_session(AUTO_PAD, DILATIONS, GROUP, k_shape, pads, strides, x, w, x_zp, w_zp, ref_y)
    check_test_result_ConvInteger(y, ref_y)


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------
def test_ConvInteger_num_one_elt(x_values, w_values):
    # X (1,1,1,1) uint8, W (1,1,1,1) int8, no zero point
    #  kernel_shape: (1,1) ; pads: (0,0,0,0) ; strides: (1,1) ; Y shape: (1,1,1,1)
    print("\ntest_ConvInteger_num_one_elt:\n")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    index = 1
    for val_x in x_values:
        x = u8(val_x, (1, 1, 1, 1))
        for val_w in w_values:
            w = i8(val_w, (1, 1, 1, 1))
            print(f"\nTest Nb: {index}")
            run_one_test(k_shape, pads, strides, x, w)
            index += 1


def test_ConvInteger_num_in_one_tensor(x_values, w_values):
    # All the pairs in one single execution:
    #  X (1,1,nx,1): X[0,0,i,0] = x_values[i]
    #  W (nw,1,1,1): W[m,0,0,0] = w_values[m]
    #  Y (1,nw,nx,1): Y[0,m,i,0] = X[i] * W[m]
    print("\nAll in one tensor")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    x = np.array(x_values, dtype=np.uint8).reshape(1, 1, len(x_values), 1)
    w = np.array(w_values, dtype=np.int8).reshape(len(w_values), 1, 1, 1)
    run_one_test(k_shape, pads, strides, x, w)


def test_ConvInteger_num_x_zero_point(x_values):
    # X (1,1,1,1) = x in x_values, x_zero_point in x_values, W (1,1,1,1) = 50
    print("\ntest_ConvInteger_num_x_zero_point:\n")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    w = i8(50, (1, 1, 1, 1))
    index = 1
    for val_x in x_values:
        x = u8(val_x, (1, 1, 1, 1))
        for val_zp in x_values:
            print(f"\nTest Nb: {index}")
            run_one_test(k_shape, pads, strides, x, w, x_zp=u8(val_zp))
            index += 1


def test_ConvInteger_num_w_zero_point(w_values):
    # X (1,1,1,1) = 31, W (1,1,1,1) = w in w_values, w_zero_point in w_values
    # (x_zero_point omitted: exercises the empty optional input)
    print("\ntest_ConvInteger_num_w_zero_point:\n")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    x = u8(31, (1, 1, 1, 1))
    index = 1
    for val_w in w_values:
        w = i8(val_w, (1, 1, 1, 1))
        for val_zp in w_values:
            print(f"\nTest Nb: {index}")
            run_one_test(k_shape, pads, strides, x, w, w_zp=i8(val_zp))
            index += 1


def test_ConvInteger_num_both_zero_points():
    # Extreme combinations of values and zero points (both zero points present)
    print("\ntest_ConvInteger_num_both_zero_points:\n")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    index = 1
    for val_x, val_xzp, val_w, val_wzp in itertools.product([0, 255], [0, 255], [-128, 127], [-128, 127]):
        print(f"\nTest Nb: {index}")
        run_one_test(k_shape, pads, strides,
                     u8(val_x, (1, 1, 1, 1)), i8(val_w, (1, 1, 1, 1)), u8(val_xzp), i8(val_wzp))
        index += 1


def test_ConvInteger_num_max_accumulation():
    # Largest magnitude of the accumulator: C=3, kernel 3x3 => 27 products of (0-255)*(-128-127) = 65025
    #  Expected Y = 27 * 65025 = 1755675 (fits in int32), and its opposite with W = 127, w_zero_point = -128
    print("\ntest_ConvInteger_num_max_accumulation:\n")
    k_shape, pads, strides = [3, 3], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    x = u8(0, (1, 3, 3, 3))
    run_one_test(k_shape, pads, strides, x, i8(-128, (1, 3, 3, 3)), u8(255), i8(127))
    run_one_test(k_shape, pads, strides, x, i8(127, (1, 3, 3, 3)), u8(255), i8(-128))


def test_ConvInteger_num_3x3_padded(x_values, w_values):
    # X (1,1,1,1), W (1,1,3,3) filled with val_w, pads (1,1,1,1), strides (1,1), Y shape (1,1,1,1)
    #  with x_zero_point = 5 and w_zero_point = -7: the padded positions must not contribute.
    print("\ntest_ConvInteger_num_3x3_padded:\n")
    k_shape, pads, strides = [3, 3], [1, 1, 1, 1], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    index = 1
    for val_x in x_values:
        x = u8(val_x, (1, 1, 1, 1))
        for val_w in w_values:
            w = i8(val_w, (1, 1, 3, 3))
            print(f"\nTest Nb: {index}")
            run_one_test(k_shape, pads, strides, x, w, u8(5), i8(-7))
            index += 1


def test_ConvInteger_num():
    global num_test_set
    print("\nNumerical tests\n")

    for x_values, w_values in zip(X_SETS, W_SETS):
        num_test_set += 1
        print("num_test_set = ", num_test_set)
        test_ConvInteger_num_one_elt(x_values, w_values)
        test_ConvInteger_num_in_one_tensor(x_values, w_values)
        test_ConvInteger_num_x_zero_point(x_values)
        test_ConvInteger_num_w_zero_point(w_values)
        test_ConvInteger_num_3x3_padded(x_values, w_values)

    num_test_set += 1
    print("num_test_set = ", num_test_set)
    test_ConvInteger_num_both_zero_points()
    test_ConvInteger_num_max_accumulation()


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------
def test_ConvInteger_struct():
    # Fixed: auto_pad = 'NOTSET', group = 1, dilations = (1,1)
    # Variable:
    #   - X shape (N,C,H,W): N in {1,2}; C in {1,3}; H,W in {1,3,5}   => 36 shapes
    #   - M (nb of filters): {1,2}
    #   - kernel_shape: {(1,1);(3,3)}
    #   - pads: {(0,0,0,0);(1,1,1,1)}
    #   - strides: {(1,1);(2,2)}
    #   - zero points: none; x_zp only; w_zp (scalar) only; x_zp + w_zp (scalar); x_zp + w_zp (1D, size M;
    #     only if M == 1 or --per_channel_w_zp, see below)
    # The combinations for which the output has an empty spatial dim (e.g. H=1, kernel 3x3, pads 0)
    # are outside of the domain of the operator (constraints) and are skipped.
    print("\nStructural tests\n")
    rng = np.random.default_rng(1234)
    nb_skipped = 0

    for N, C, H, W in itertools.product([1, 2], [1, 3], [1, 3, 5], [1, 3, 5]):
        for M in [1, 2]:
            for ks, pad, stride in itertools.product([1, 3], [0, 1], [1, 2]):
                k_shape = [ks, ks]
                pads = [pad] * 4
                strides = [stride, stride]

                y_shape = compute_ConvInteger_output_shape((N, C, H, W), (M, C, ks, ks), pads, DILATIONS, strides)
                if y_shape[2] < 1 or y_shape[3] < 1:
                    nb_skipped += 1
                    continue

                x = rng.integers(0, 256, size=(N, C, H, W)).astype(np.uint8)
                w = rng.integers(-128, 128, size=(M, C, ks, ks)).astype(np.int8)
                x_zp = np.array(rng.integers(0, 256), dtype=np.uint8)
                w_zp_scalar = np.array(rng.integers(-128, 128), dtype=np.int8)
                w_zp_1d = rng.integers(-128, 128, size=(M,)).astype(np.int8)

                zp_variants = {
                    "no zp": (None, None),
                    "x_zp": (x_zp, None),
                    "w_zp scalar": (None, w_zp_scalar),
                    "x_zp + w_zp scalar": (x_zp, w_zp_scalar),
                }
                # Per-channel w_zero_point (1D of size M): allowed by the ONNX spec, but ONNX Runtime
                # only supports it when M == 1 (use --per_channel_w_zp to test M > 1 as well).
                if M == 1 or PER_CHANNEL_W_ZP:
                    zp_variants["x_zp + w_zp 1D"] = (x_zp, w_zp_1d)
                for label, (xz, wz) in zp_variants.items():
                    print(f"\nStructural test: X={x.shape}, W={w.shape}, {label}, "
                          f"pads={pads}, strides={strides}\n")
                    run_one_test(k_shape, pads, strides, x, w, xz, wz)

    print(f"\nStructural tests: {nb_skipped} invalid combinations skipped (empty output)\n")


parser = argparse.ArgumentParser(description="Tests of the ONNX ConvInteger operator (2D, uint8 x int8 -> int32).")
parser.add_argument("--tests", choices=["num", "struct", "all"], default="all",
                    help="Family of tests to run: num, struct or all")
parser.add_argument("--per_channel_w_zp", action="store_true",
                    help="Also test a 1D w_zero_point of size M > 1 (not supported by ONNX Runtime)")
args = parser.parse_args()

PER_CHANNEL_W_ZP = args.per_channel_w_zp

if args.tests in ("num", "all"):
    test_ConvInteger_num()
if args.tests in ("struct", "all"):
    test_ConvInteger_struct()

print(f"\nSummary: {nb_tests} tests, {nb_tests - nb_ko} OK, {nb_ko} KO")
