"""
Automatic tests for the Conv operator (SONNX) - 2D convolution, float32
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
#   - X shape is of the form (N, C, H, W)
#   - Singular points and classes for dims: dTi_no_element;dTi_min;]dTi_min,dTi_sufficient[;dTi_sufficient;]dTi_sufficient,dTi_max[;dTi_max
#     Let's set the values:
#      - dTi_no_element = 0   : excluded, the constraints of Conv require every dimension > 0
#      - dTi_min = 1
#      - dTi_sufficient = 5 for H and W, 3 for C (and 2 for N, M: batch and nb of filters)
#      - dTi_max = max positive integer. Not tractable.
#   - Selected values:
#      - N in {1,2}; C in {1,3}; H and W in {1,3,5}
# Equivalence classes for W (weights), shape (M, C/group, kH, kW):
#      - M in {1,2}; C/group = C (group == 1 fixed)
#      - (kH,kW) = kernel_shape, see below
# Equivalence classes for B (optional bias), shape (M): absent;present
# Equivalence class for auto_pad: 'NOTSET' (fixed)
# Equivalence class for group: 1 (fixed)
# Equivalence class for dilations: (1,1) (fixed)
# Equivalence classes for kernel_shape:
#      - kernel_shape is of the form (ks0,ks1); selected values: {(1,1);(3,3)}
# Equivalence classes for pads:
#      - pads is of the form (pad0,pad1,pad2,pad3); selected values: {(0,0,0,0);(1,1,1,1)}
# Equivalence classes for strides:
#      - strides is of the form (str0,str1); selected values: {(1,1);(2,2)}
# Equivalence classes for elements of X, W and B:
#      - EQCELTS: NaN;;-inf;minfloat;]minfloat,-0[;-0;0;]0,maxfloat[;maxfloat;+inf
#
# Two families of tests:
#      - Tests of the numerical aspects (special values)
#      - Tests of the structural aspects (shapes x kernel_shape x pads x strides)

# Fixed attributes
AUTO_PAD = "NOTSET"
GROUP = 1
DILATIONS = [1, 1]
onnx_type = TensorProto.FLOAT

# Semantics of the padding in the reference implementation:
#   - False: the padded positions are excluded from the sum (they do not contribute).
#   - True : the padded positions are real zeros which are multiplied by the weights
#            (so 0 * inf = NaN when a weight is infinite or NaN).
COUNT_PADDING_PRODUCTS = True   # observed behaviour of ONNX Runtime

info_32 = np.finfo(np.float32)
max_32 = info_32.max    # Approx. 3.40e+38
min_32 = info_32.min    # Approx. -3.40e+38
tiny_32 = info_32.tiny  # Approx. 1.17e-38

num_test_set = 0
nb_tests = 0
nb_ko = 0


def print_attributes(pads, kernel_shape, dilation, strides, auto_pad, group):
    print("\nAttributes:")
    print(f"pads = {pads}")
    print(f"kernel_shape = {kernel_shape}")
    print(f"dilations = {dilation}")
    print(f"strides = {strides}")
    print(f"auto_pad = {auto_pad}")
    print(f"group = {group}\n")


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def compute_Conv_output_shape(x_shape, w_shape, pads, dilation, strides):
    dx0, dx1, dx2, dx3 = x_shape
    dw0, dw1, dw2, dw3 = w_shape

    alpha = dx2 + pads[0] + pads[2]
    beta = dx3 + pads[1] + pads[3]
    theta = dilation[0] * (dw2 - 1) + 1
    gamma = dilation[1] * (dw3 - 1) + 1

    dy2 = math.floor((alpha - theta) / strides[0]) + 1
    dy3 = math.floor((beta - gamma) / strides[1]) + 1

    return [dx0, dw0, dy2, dy3]


def Conv(X, W, B, strides, dilation, pads):
    """
    Naive reference of ONNX Conv (rank 4, group = 1), float32 accumulation.
    X: (N,C,H,W) ; W: (M,C,kH,kW) ; B: (M) or None
    pads = (top, left, bottom, right)
    """
    N, C, H, Wd = X.shape
    M, _, kH, kW = W.shape
    s_h, s_w = strides
    d_h, d_w = dilation
    p_top, p_left, _, _ = pads

    y_shape = compute_Conv_output_shape(X.shape, W.shape, pads, dilation, strides)
    Y = np.zeros(y_shape, dtype=np.float32)

    with np.errstate(all="ignore"):
        for n in range(N):
            for m in range(M):
                for i in range(y_shape[2]):
                    for j in range(y_shape[3]):
                        acc = np.float32(0.0)
                        for c in range(C):
                            for u in range(kH):
                                for v in range(kW):
                                    h = i * s_h + u * d_h - p_top
                                    w = j * s_w + v * d_w - p_left
                                    if 0 <= h < H and 0 <= w < Wd:
                                        acc = np.float32(acc + X[n, c, h, w] * W[m, c, u, v])
                                    elif COUNT_PADDING_PRODUCTS:
                                        acc = np.float32(acc + np.float32(0.0) * W[m, c, u, v])
                        if B is not None:
                            acc = np.float32(acc + B[m])
                        Y[n, m, i, j] = acc
    return Y


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------
def make_and_execute_session(auto_pad, dilation, group, k_shape, pads, strides, x, w, b, ref_y, dtype=onnx_type):

    inputs_names = ["X", "W"]
    inputs_info = [
        helper.make_tensor_value_info("X", dtype, x.shape),
        helper.make_tensor_value_info("W", dtype, w.shape),
    ]
    feeds = {"X": x, "W": w}
    if b is not None:
        inputs_names.append("B")
        inputs_info.append(helper.make_tensor_value_info("B", dtype, b.shape))
        feeds["B"] = b

    y_info = helper.make_tensor_value_info("Y", dtype, ref_y.shape)

    conv_node = helper.make_node(
        "Conv",
        inputs=inputs_names,
        outputs=["Y"],
        auto_pad=auto_pad,
        dilations=dilation,
        group=group,
        kernel_shape=k_shape,
        pads=pads,
        strides=strides,
    )

    graph_def = helper.make_graph([conv_node], "Test_Conv", inputs_info, [y_info])

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
    if b is not None:
        print("B shape:", b.shape)
        print("B =", fmt(b))
    print("Y shape:", y.shape)
    print("ONNX Y =", fmt(y))
    print("ref_y =", fmt(ref_y))

    return y


def check_test_result_Conv(Y, refY):
    global nb_tests, nb_ko
    nb_tests += 1
    print("\ncheck_test_result_Conv\n")
    fmt = lambda t: np.array2string(t, separator=",", max_line_width=np.inf)
    print(f"Y = {fmt(Y)}")
    print(f"refY = {fmt(refY)}")
    if Y.shape == refY.shape and np.allclose(Y, refY, rtol=1.2e-07, atol=1.2e-01, equal_nan=True):
        print("-> Test of Conv is OK\n")
    else:
        nb_ko += 1
        print("-> Test of Conv is KO\n")


def check_constraints(x, w, b, auto_pad, group, kernel_shape, pads, strides, dilation, y):
    # x
    assert x.ndim == 4 and all(d > 0 for d in x.shape)
    # w
    assert w.ndim == 4 and all(d > 0 for d in w.shape)
    assert list(w.shape[2:]) == list(kernel_shape)
    assert w.shape[1] == x.shape[1] // group
    # b
    if b is not None:
        assert b.shape == (w.shape[0],)
    # attributes
    assert auto_pad == "NOTSET" and group == 1
    assert len(kernel_shape) == 2 and all(k > 0 for k in kernel_shape)
    assert len(strides) == 2 and all(s > 0 for s in strides)
    assert len(dilation) == 2 and all(d > 0 for d in dilation)
    assert len(pads) == 4 and all(p >= 0 for p in pads)
    # y
    assert list(y.shape) == compute_Conv_output_shape(x.shape, w.shape, pads, dilation, strides)


def run_one_test(k_shape, pads, strides, x, w, b):
    ref_y = Conv(x, w, b, strides, DILATIONS, pads)
    check_constraints(x, w, b, AUTO_PAD, GROUP, k_shape, pads, strides, DILATIONS, ref_y)
    y = make_and_execute_session(AUTO_PAD, DILATIONS, GROUP, k_shape, pads, strides, x, w, b, ref_y)
    check_test_result_Conv(y, ref_y)


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------
def test_Conv_num_one_elt(test_values):
    # Numeric test: X (1,1,1,1), W (1,1,1,1), no bias
    #  kernel_shape: (1,1) ; pads: (0,0,0,0) ; strides: (1,1) ; Y shape: (1,1,1,1)
    #  All the pairs (val_x, val_w) of test_values are tested.
    print("\ntest_Conv_num_one_elt:\n")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    index = 1
    for val_x in test_values:
        x = np.full((1, 1, 1, 1), val_x, dtype="float32")
        for val_w in test_values:
            w = np.full((1, 1, 1, 1), val_w, dtype="float32")
            print(f"\nTest Nb: {index}")
            run_one_test(k_shape, pads, strides, x, w, None)
            index += 1


def test_Conv_num_one_elt_bias(test_values):
    # Numeric test: X (1,1,1,1) = 31.0, W (1,1,1,1) = 50.0, B (1) in test_values
    print("\ntest_Conv_num_one_elt_bias:\n")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    x = np.full((1, 1, 1, 1), 31.0, dtype="float32")
    w = np.full((1, 1, 1, 1), 50.0, dtype="float32")
    for index, val_b in enumerate(test_values, start=1):
        b = np.full((1,), val_b, dtype="float32")
        print(f"\nTest Nb: {index}")
        run_one_test(k_shape, pads, strides, x, w, b)


def test_Conv_num_in_one_tensor(test_values):
    # All the pairs in one single execution:
    #  X (1,1,n,1): X[0,0,i,0] = test_values[i]
    #  W (n,1,1,1): W[m,0,0,0] = test_values[m]
    #  Y (1,n,n,1): Y[0,m,i,0] = X[i] * W[m]
    print("\nAll in one tensor")
    k_shape, pads, strides = [1, 1], [0, 0, 0, 0], [1, 1]
    n = len(test_values)
    x = np.array(test_values, dtype="float32").reshape(1, 1, n, 1)
    w = np.array(test_values, dtype="float32").reshape(n, 1, 1, 1)
    run_one_test(k_shape, pads, strides, x, w, None)


def test_Conv_num_3x3_padded(test_values):
    # Numeric test: X (1,1,1,1), W (1,1,3,3) filled with val_w, pads (1,1,1,1), strides (1,1)
    #  Y shape: (1,1,1,1). Only the central weight meets X (the 8 others are on the padding).
    print("\ntest_Conv_num_3x3_padded:\n")
    k_shape, pads, strides = [3, 3], [1, 1, 1, 1], [1, 1]
    print_attributes(pads, k_shape, DILATIONS, strides, AUTO_PAD, GROUP)

    index = 1
    for val_x in test_values:
        x = np.full((1, 1, 1, 1), val_x, dtype="float32")
        for val_w in test_values:
            w = np.full((1, 1, 3, 3), val_w, dtype="float32")
            print(f"\nTest Nb: {index}")
            run_one_test(k_shape, pads, strides, x, w, None)
            index += 1


def test_one_input_pattern(test_values):
    test_Conv_num_one_elt(test_values)
    test_Conv_num_in_one_tensor(test_values)
    test_Conv_num_one_elt_bias(test_values)
    test_Conv_num_3x3_padded(test_values)


def test_Conv_num():
    # EQCELTS: NaN;;-inf;minfloat;]minfloat,-0.0[;-0.0;0.0;]0.0,maxfloat[;maxfloat;+inf
    # Choices of values in the intervals:
    #   - ]minfloat,0.0[: {minfloat/2.0, -0.0-tiny_32}
    #   - ]0.0,maxfloat[: {0.0+tiny_32, maxfloat/2.0}
    global num_test_set
    print("\nNumerical tests\n")

    num_test_set += 1
    print("num_test_set = ", num_test_set)
    test_one_input_pattern([np.nan, -np.inf, min_32, min_32 / 2.0, -0.0, 0.0, 0.0 + tiny_32, max_32, np.inf])

    num_test_set += 1
    print("num_test_set = ", num_test_set)
    test_one_input_pattern([np.nan, -np.inf, min_32, min_32 / 2.0, -0.0, 0.0, max_32 / 2.0, max_32, np.inf])

    num_test_set += 1
    print("num_test_set = ", num_test_set)
    test_one_input_pattern([np.nan, -np.inf, min_32, -0.0 - tiny_32, -0.0, 0.0, 0.0 + tiny_32, max_32, np.inf])

    num_test_set += 1
    print("num_test_set = ", num_test_set)
    test_one_input_pattern([np.nan, -np.inf, min_32, -0.0 - tiny_32, -0.0, 0.0, max_32 / 2.0, max_32, np.inf])


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------
def test_Conv_struct():
    # Fixed: auto_pad = 'NOTSET', group = 1, dilations = (1,1)
    # Variable:
    #   - X shape (N,C,H,W): N in {1,2}; C in {1,3}; H,W in {1,3,5}   => 36 shapes
    #   - M (nb of filters): {1,2}
    #   - kernel_shape: {(1,1);(3,3)}
    #   - pads: {(0,0,0,0);(1,1,1,1)}
    #   - strides: {(1,1);(2,2)}
    #   - bias: absent;present
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

                y_shape = compute_Conv_output_shape((N, C, H, W), (M, C, ks, ks), pads, DILATIONS, strides)
                if y_shape[2] < 1 or y_shape[3] < 1:
                    nb_skipped += 1
                    continue

                # Small integer values: exact computations in float32
                x = rng.integers(-5, 6, size=(N, C, H, W)).astype("float32")
                w = rng.integers(-3, 4, size=(M, C, ks, ks)).astype("float32")

                for with_bias in [False, True]:
                    b = rng.integers(-5, 6, size=(M,)).astype("float32") if with_bias else None
                    print(f"\nStructural test: X={x.shape}, W={w.shape}, bias={with_bias}, "
                          f"pads={pads}, strides={strides}\n")
                    run_one_test(k_shape, pads, strides, x, w, b)

    print(f"\nStructural tests: {nb_skipped} invalid combinations skipped (empty output)\n")


parser = argparse.ArgumentParser(description="Tests of the ONNX Conv operator (2D, float32).")
parser.add_argument("--tests", choices=["num", "struct", "all"], default="all",
                    help="Family of tests to run: num, struct or all")
parser.add_argument("--skip_padding_products", action="store_true",
                    help="Reference excludes the padded positions from the sum (default: they are real zeros, 0*inf = NaN)")
args = parser.parse_args()

COUNT_PADDING_PRODUCTS = not args.skip_padding_products

if args.tests in ("num", "all"):
    test_Conv_num()
if args.tests in ("struct", "all"):
    test_Conv_struct()

print(f"\nSummary: {nb_tests} tests, {nb_tests - nb_ko} OK, {nb_ko} KO")
