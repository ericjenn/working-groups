"""
This file uses the Hypothesis library to generate a wide range of test cases
for the Slice operation in ONNX.
"""
import os

import json
import numpy as np
import ml_dtypes

from hypothesis import given, settings
import hypothesis.extra.numpy as hnp
from hypothesis import strategies as st
from hypothesis import assume

from onnx import helper
import onnx.checker
from onnxruntime import InferenceSession
import onnx.reference


from onnx import helper

import tensorflow as tf

if os.path.exists("generated_data.json"):
    os.remove("generated_data.json")

"""
Inputs/attributes details
"""
inputs_attributes = {
    "min_rank_input": 1,
    "max_rank_input": 5,
    "min_dim_size_input": 1,
    "max_dim_size_input": 10
}

# TODO ONNX Runtime does not support bfloat16 yet??
"""
Slice supported types
"""
slice_types = {
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "STRING": np.str_,
    "BOOL": np.bool_,
    "BFLOAT16": ml_dtypes.bfloat16
}


slice_index_types = {
    "INT32": np.int32,
    "INT64": np.int64,
}

"""
Store generated data
"""
generated_data = {
    "rank_input_tensor": [],
    "shape_input_tensor": [],
    "x_tensor": [],
    "a_tensor": [],
    "k_tensor": [],
    "s_tensor": [],
    "e_tensor": [],
    "x_type": [],
    "index_type": []
}

"""
Function to generate valid slice arguments
"""
@st.composite
@settings()
def valid_slice_args(draw):
    # Generate X tensor
    rank_input_tensor = draw(st.integers(
        min_value=inputs_attributes["min_rank_input"],
        max_value=inputs_attributes["max_rank_input"]
    ))

    shape_input_tensor = []
    for _ in range(rank_input_tensor):
        dim_size = draw(st.integers(
            min_value=inputs_attributes["min_dim_size_input"],
            max_value=inputs_attributes["max_dim_size_input"]
        ))
        shape_input_tensor.append(dim_size)

    input_type = draw(st.sampled_from(list(slice_types.keys())))
    dtype = slice_types[input_type]

    if np.issubdtype(dtype, np.integer):
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max
        elements_strategy = st.integers(min_value=min_val, max_value=max_val)
    elif np.issubdtype(dtype, np.floating):
        min_val = np.finfo(dtype).min
        max_val = np.finfo(dtype).max
        elements_strategy = st.floats(min_value=min_val, max_value=max_val)
    elif np.issubdtype(dtype, np.bool_):
        elements_strategy = st.booleans()
    elif np.issubdtype(dtype, np.str_):
        elements_strategy = st.text(
            alphabet=st.characters(codec="utf-8", blacklist_characters='\x00')
        )
    elif input_type == "BFLOAT16":
        min_bfloat16 = float(ml_dtypes.finfo(slice_types["BFLOAT16"]).min)
        max_bfloat16 = float(ml_dtypes.finfo(slice_types["BFLOAT16"]).max)
        elements_strategy = st.floats(min_value=min_bfloat16, max_value=max_bfloat16)


    if input_type == "BFLOAT16":
        temp_tensor = draw(hnp.arrays(dtype=np.float32, shape=shape_input_tensor, elements=elements_strategy))
        tf_tensor = tf.cast(tf.constant(temp_tensor), tf.bfloat16)
        x = tf_tensor.numpy()
    else:
        x = draw(hnp.arrays(dtype=dtype, shape=shape_input_tensor, elements=elements_strategy))

    # Generate indices tensors
    type_index_tensors = draw(st.sampled_from(list(slice_index_types.keys())))
    dtype_index = slice_index_types[type_index_tensors]
    # A [C1] -> X [C1]
    da0 = rank_input_tensor
    possible_indices = draw(st.permutations(range(rank_input_tensor)))[:da0]
    a = []
    for idx in possible_indices:
        make_negative = draw(st.booleans())
        if make_negative:
            a.append(idx - rank_input_tensor)
        else:
            a.append(idx)
    a = np.array(a, dtype=dtype_index)
    a_normalized = np.where(a < 0, a + rank_input_tensor, a)


    # K [C1] -> X [C1]
    dk0 = rank_input_tensor
    k = np.empty(dk0, dtype=dtype_index)
    # K [C2]
    dk0_elements_strategy = st.one_of(
        st.integers(min_value=np.iinfo(dtype_index).min, max_value=-1),
        st.integers(min_value=1, max_value=np.iinfo(dtype_index).max)
    )
    for i in range(rank_input_tensor):
        k[i] = draw(dk0_elements_strategy)

    # S [1] -> X [C1]
    ds0 = rank_input_tensor
    s = np.empty(ds0, dtype=dtype_index)
    # S [C2]
    for i in range(rank_input_tensor):
        if k[i] > 0:
            min_val = - shape_input_tensor[a_normalized[i]]
            max_val = shape_input_tensor[a_normalized[i]] - 1
        else:
            min_val = -shape_input_tensor[a_normalized[i]]
            max_val = shape_input_tensor[a_normalized[i]] - 1
        elements_strategy_s = st.integers(min_value=min_val, max_value=max_val)
        s[i] = draw(elements_strategy_s)
    s_normalized = np.empty(ds0, dtype=dtype_index)
    for i in range(len(s)):
        if s[i] < 0:
            print("s[i]:", s[i], "shape_input_tensor[a[i]]:", shape_input_tensor[a[i]])
            s_normalized[i] = s[i] + shape_input_tensor[a_normalized[i]]
        else:
            s_normalized[i] = s[i]

    # E [1] -> X [C1]
    de0 = rank_input_tensor
    e = np.empty(de0, dtype=dtype_index)
    # E [C2]
    for i in range(rank_input_tensor):
        if k[i] > 0:
            min_val = -shape_input_tensor[a_normalized[i]]
            max_val = shape_input_tensor[a_normalized[i]]
        else:
            #TODO Change this (I THINK DONT NEED CHANGE)
            min_val = -shape_input_tensor[a_normalized[i]] - 1
            max_val =  shape_input_tensor[a_normalized[i]] - 1
        elements_strategy_e = st.integers(min_value=min_val, max_value=max_val)
        e[i] = draw(elements_strategy_e)
    e_normalized = np.empty(de0, dtype=dtype_index)
    for i in range(len(e)):
        if e[i] < 0:
            print("e[i]:", e[i], "shape_input_tensor[a[i]]:", shape_input_tensor[a[i]])
            e_normalized[i] = e[i] + shape_input_tensor[a_normalized[i]]
        else:
            e_normalized[i] = e[i]


    # TODO Erro na informal spec E[C3] tem as restrições com o número mal
    for i in range(rank_input_tensor):
        if k[a_normalized[i]] > 0:
            assume(s_normalized[a_normalized[i]] <= e_normalized[a_normalized[i]])
        elif k[a_normalized[i]] < 0:
            assume(s_normalized[a_normalized[i]] >= e_normalized[a_normalized[i]])

    print("a_normalized:", a_normalized)
    print("s_normalized:", s_normalized)
    print("e_normalized:", e_normalized)
    # Output
    output_shape = [0] * rank_input_tensor
    # Y [C1] -> X [C1]
    for i in range(rank_input_tensor):
        space_i = e_normalized[i] - s_normalized[i]
        print("space_i:", space_i)
        k_val = k[i]
        print("k_val:", k_val)
        f = 0 if space_i % k_val == 0 else 1
        print("f:", f)
        dY_i = (space_i // k_val) + f
        print("dY_i:", dY_i)
        # Y [C2]
        assume(dY_i >= 0)
        #TODO Informal spec mal
        output_shape[a_normalized[i]] = int(dY_i)



    print("output_shape:", output_shape)
    return x, a, k, s, e, output_shape, s_normalized, a_normalized

"""
Function that runs the test
"""
@settings(max_examples=100000, deadline=None)
@given(valid_slice_args())
def test_slice(args):
    x, a, k, s, e, y_shape, s_normalized, a_normalized = args
    generated_data["rank_input_tensor"].append(len(x.shape))
    generated_data["shape_input_tensor"].append(list(x.shape))
    generated_data["x_tensor"].append(x.tolist())
    generated_data["a_tensor"].append(a.tolist())
    generated_data["k_tensor"].append(k)
    generated_data["s_tensor"].append(s.tolist())
    generated_data["e_tensor"].append(e.tolist())
    generated_data["x_type"].append(str(x.dtype))
    generated_data["index_type"].append(str(a.dtype))

    y = run_onnx_slice(x, a, k, s, e, y_shape)
    check_constraints(x, y, s_normalized, k, a_normalized)

def teardown_module():
    """
    Function to write generated data to a json file
    """
    data = {
        "title": "Data generated by Hypothesis for Slice operation tests",
        "min_rank_input": inputs_attributes["min_rank_input"],
        "max_rank_input": inputs_attributes["max_rank_input"],
        "rank_input_tensor": generated_data["rank_input_tensor"],
        "min_dim_size_input": inputs_attributes["min_dim_size_input"],
        "max_dim_size_input": inputs_attributes["max_dim_size_input"],
        "shape_input_tensor": generated_data["shape_input_tensor"],
        "x_tensor": generated_data["x_tensor"],
        "a_tensor": generated_data["a_tensor"],
        "k_tensor": generated_data["k_tensor"],
        "s_tensor": generated_data["s_tensor"],
        "e_tensor": generated_data["e_tensor"],
        "x_type": generated_data["x_type"],
        "index_type": generated_data["index_type"]
    }

    with open("generated_data.json", "w", encoding="utf-8") as f:
        data["k_tensor"] = [(arr.tolist(), str(arr.dtype)) for arr in data["k_tensor"]]
        json.dump(data, f, indent=4)


def run_onnx_slice(x, a, k, s, e, y_shape):
    """
    Function that runs the ONNX Slice operation
    """
    x_onnx = helper.make_tensor_value_info('x', helper.np_dtype_to_tensor_dtype(x.dtype), x.shape)
    a_onnx = helper.make_tensor_value_info('a', helper.np_dtype_to_tensor_dtype(a.dtype), a.shape)
    k_onnx = helper.make_tensor_value_info('k', helper.np_dtype_to_tensor_dtype(k.dtype), k.shape)
    s_onnx = helper.make_tensor_value_info('s', helper.np_dtype_to_tensor_dtype(s.dtype), s.shape)
    e_onnx = helper.make_tensor_value_info('e', helper.np_dtype_to_tensor_dtype(e.dtype), e.shape)
    y_onnx = helper.make_tensor_value_info('y', helper.np_dtype_to_tensor_dtype(x.dtype), y_shape)

    node_def = helper.make_node(
        'Slice',
        inputs=['x', 's', 'e', 'a', 'k'],
        outputs=['y'],
    )

    # Create the graph
    graph_def = helper.make_graph(
        [node_def],
        'test-clip',
        [x_onnx, s_onnx, e_onnx, a_onnx, k_onnx],
        [y_onnx],
    )

    onnx_model = helper.make_model(graph_def)

    #Let's freeze the opset.
    del onnx_model.opset_import[:]
    opset = onnx_model.opset_import.add()
    opset.domain = ''
    opset.version = 13
    onnx_model.ir_version = 10

    # Verify the model
    onnx.checker.check_model(onnx_model)

    #TODO ONNX Runtime does not support bfloat16 yet??
    if str(x.dtype) == "bfloat16":
        # Use ONNX Reference Implementation for bfloat16
        sess = onnx.reference.ReferenceEvaluator(onnx_model)
    else:
        # Use ONNX Runtime for other types
        sess = InferenceSession(onnx_model.SerializeToString(),
                               providers=["CPUExecutionProvider"])

    y = sess.run(None, {'x': x, 's': s, 'e': e, 'a': a, 'k': k})[0]

    print("x shape:", x.shape)
    print("a shape:", a.shape)
    print("a values:", a)
    print("k shape:", k.shape)
    print("k values:", k)
    print("s shape:", s.shape)
    print("s values:", s)
    print("e shape:", e.shape)
    print("e values:", e)

    print("y shape:", y.shape)
    print("y values:", y)
    print("y data type:", y.dtype)
    return y


def check_output_input(x, y, start, step, axis):
    indices = np.ndindex(y.shape)

    for out_idx in indices:
        real_index = []
        for dim in range(len(out_idx)):
            axis_index = axis.tolist().index(dim)
            real_index.append(axis[axis_index])
    
    for out_idx in indices:
        in_idx = tuple(start[real_index[dim]] + out_idx[dim] * step[real_index[dim]] for dim in range(len(out_idx)))
        if x[in_idx] != y[out_idx]:
            print(f"Mismatch at output index {out_idx}: expected {x[in_idx]}, got {y[out_idx]}")
            return False
    
    return True

def check_constraints(x, y, start, step, axis):
    """
    Check constraints for generated data
    """
    assert  check_output_input(x, y, start, step, axis)