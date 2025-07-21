#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By  : Salome Marty Laurent  
# Created Date: 09/07/2025
# version ='2.0'
# ---------------------------------------------------------------------------

""" This file was designed to caracterize and test the ONNX operator: Add

Five main tests are defined: 
    - Add with two scalar values (line 30)
    - Add with two vector inputs (line 100)
    - Add with two input matrixes (line 166)
    - Add with two input matrixes with different sizes (broadcast property illustration) (line 220)
    - Add with two 3D tensors (line 300)
    - Add with two 3D tensors with different data types (line 360)
    - Add with overflow for Int32 (line 460)
    - Add with overflow for Float32 (line ..)
    
    """  


# ---------------------------------------------------------------------------

import onnxruntime
import onnx
import numpy

print(onnxruntime.__version__)

# ---------------------------------------------------------------------------

""" ONNX Add operator: test two empty tensors as scalar values """

# ---------------------------------------------------------------------------

a_val = onnx.helper.make_tensor_value_info('a', onnx.helper.TensorProto.FLOAT, [])
b_val = onnx.helper.make_tensor_value_info('b', onnx.helper.TensorProto.FLOAT, [])


# Create a node (Concat) with input/outputs
node_def = onnx.helper.make_node(
    'Add', 
    ['a', 'b'], 
    ['c']
)


graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add-scalars',
    [a_val, b_val], 
    [onnx.helper.make_tensor_value_info('c', onnx.helper.TensorProto.FLOAT, [])], 
)

onnx_model = onnx.helper.make_model(graph_def, producer_name='onnx-scalar-add-test')

# Set Opset
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15 
onnx_model.ir_version = 8 

# Verify the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"ONNX model invalid: {e}")


print("\nTest with scalar inputs for Add operator:\n")
print(onnx.helper.printable_graph(onnx_model.graph))

# Initialize tensors
a_np = numpy.array(1.0, dtype=numpy.float32) 
b_np = numpy.array(2.0, dtype=numpy.float32) 


# Do inference with try-except block
try:
    print("\nAttempting to create InferenceSession and run...")
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    
    c_result = sess.run(None, {'a': a_np, 'b': b_np})[0]

    print("\nInference successful (if this line is reached).")
    print("Output C shape:", c_result.shape)
    print("Output C value:", c_result)

   
except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
    print("\n--- ONNX Runtime Execution Error Caught ---")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
 
    
except Exception as e:
    print("\n--- An Unexpected Error Occurred ---")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")

print("\nScript execution continued after inference attempt.")


# ---------------------------------------------------------------------------

""" ONNX Add operator: test with vector inputs """

# ---------------------------------------------------------------------------

# Create inputs
a = onnx.helper.make_tensor_value_info('a', onnx.helper.TensorProto.FLOAT, [1])
b = onnx.helper.make_tensor_value_info('b', onnx.helper.TensorProto.FLOAT, [1])



# Create a node (Add) with input/outputs
node_def = onnx.helper.make_node(
    'Add', # node name
    ['a', 'b'], # inputs
    ['c']
)

# Create the graph
graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add',
    [a, b],
    [onnx.helper.make_tensor_value_info('c', onnx.helper.TensorProto.FLOAT, [1])],
)

onnx_model = onnx.helper.make_model(graph_def)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Verify the model
onnx.checker.check_model(onnx_model)
print("\n Add test with scalar inputs: \n")
# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))

# Do inference
sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])

# Initialize tensors
a_val = 1* numpy.ones((1), dtype=numpy.float32)
b_val = 2* numpy.ones((1), dtype=numpy.float32)



c = sess.run(None, {'a': a_val, 'b':b_val})[0]

print("C shape:",
      c.shape)
print("C:", c)


# ---------------------------------------------------------------------------

""" ONNX Add operator: test with input matrixes """

# ---------------------------------------------------------------------------


# Create inputs
a = onnx.helper.make_tensor_value_info('a', onnx.helper.TensorProto.FLOAT, [4, 3])
b = onnx.helper.make_tensor_value_info('b', onnx.helper.TensorProto.FLOAT, [4, 3])


# Create a node (Add) with input/outputs
node_def = onnx.helper.make_node(
    'Add', # node name
    ['a', 'b'], # inputs
    ['c'] # output
)

# Create the graph
graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add',
    [a, b],
    [onnx.helper.make_tensor_value_info('c', onnx.helper.TensorProto.FLOAT, [4, 3])],
)

onnx_model = onnx.helper.make_model(graph_def)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Verify the model
onnx.checker.check_model(onnx_model)

print("\n Test with input matrixes: \n")
# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))

# Do inference
sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])


a = numpy.array([[1, 2, 3],
                 [4, 5, 6],
                 [7,  8,  9],
                 [10, 11, 12]], dtype=numpy.float32)


b = numpy.array([[13, 14, 15],
                 [16, 17, 18],
                 [19, 20, 21],
                 [22, 23, 24]], dtype=numpy.float32)


c = sess.run(None, {'a': a, 'b':b})[0]


print("C shape:", c.shape)
print("C:", c)

# ---------------------------------------------------------------------------

""" ONNX Add operator: test with input matrixes with different size to illustrate one case of broadcast property """

# ---------------------------------------------------------------------------


a = onnx.helper.make_tensor_value_info('a', onnx.helper.TensorProto.FLOAT, [4, 3])
b = onnx.helper.make_tensor_value_info('b', onnx.helper.TensorProto.FLOAT, [3])


# Create a node (Add) with input/outputs
node_def = onnx.helper.make_node(
    'Add',
    ['a', 'b'],
    ['c']
)

graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add-broadcast',
    [a, b],
    [onnx.helper.make_tensor_value_info('c', onnx.helper.TensorProto.FLOAT, [4, 3])],
)

onnx_model = onnx.helper.make_model(graph_def)


del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Verify the model
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
except onnx.checker.ValidationError as e:
    print(f"ONNX model is invalid: {e}")


print("\nTest with broadcastable input matrices:\n")
# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))

# Do inference
sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])



a_np = numpy.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12]], dtype=numpy.float32)


b_np = numpy.array([100, 200, 300], dtype=numpy.float32)


# Run inference
c_result = sess.run(None, {'a': a_np, 'b': b_np})[0]


print("\nC shape:", c_result.shape)
print("C:", c_result)

# ---------------------------------------------------------------------------

""" ONNX Add operator: test with 3D input tensors """

# ---------------------------------------------------------------------------


# Create inputs
a = onnx.helper.make_tensor_value_info('a', onnx.helper.TensorProto.FLOAT, [2, 3, 4])
b = onnx.helper.make_tensor_value_info('b', onnx.helper.TensorProto.FLOAT, [2, 3, 4])


# Create a node (Add) with input/outputs
node_def = onnx.helper.make_node(
    'Add', # node name
    ['a', 'b'], # inputs
    ['c'] # output
)

# Create the graph
graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add',
    [a, b],
    [onnx.helper.make_tensor_value_info('c', onnx.helper.TensorProto.FLOAT, [2, 3, 4])],
)

onnx_model = onnx.helper.make_model(graph_def)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Verify the model
onnx.checker.check_model(onnx_model)

print("\n Test with input matrixes: \n")
# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))

# Do inference
sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])


a = numpy.array([[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]], 
                 [[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]] ], dtype=numpy.float32)


b = numpy.array([[[100, 101, 102, 103],
                  [104, 105, 106, 107],
                  [108, 109, 110, 111]],
                 [[112, 113, 114, 115],
                  [116, 117, 118, 119],
                  [120, 121, 122, 123]]], dtype=numpy.float32)



c = sess.run(None, {'a': a, 'b':b})[0]


print("C shape:", c.shape)
print("C:", c)

# ---------------------------------------------------------------------------

""" ONNX Add operator: test with 3D input tensors with different types"""

# ---------------------------------------------------------------------------


# Create inputs
a = onnx.helper.make_tensor_value_info('a', onnx.helper.TensorProto.FLOAT, [2, 3, 4])
b = onnx.helper.make_tensor_value_info('b', onnx.helper.TensorProto.INT32, [2, 3, 4])


# Create a node (Add) with input/outputs
node_def = onnx.helper.make_node(
    'Add', # node name
    ['a', 'b'], # inputs
    ['c'] # output
)

# Create the graph
graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add',
    [a, b],
    [onnx.helper.make_tensor_value_info('c', onnx.helper.TensorProto.FLOAT, [2, 3, 4])],
)

onnx_model = onnx.helper.make_model(graph_def)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Verify the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"ONNX model invalid: {e}")

print("\n Test with input matrixes: \n")
# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))


a = numpy.array([[[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]], 
                 [[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]] ], dtype=numpy.float32)


b = numpy.array([[[100, 101, 102, 103],
                  [104, 105, 106, 107],
                  [108, 109, 110, 111]],
                 [[112, 113, 114, 115],
                  [116, 117, 118, 119],
                  [120, 121, 122, 123]]], dtype=numpy.int32)

# Do inference with try-except block
try:
    print("\nAttempting to create InferenceSession and run...")
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    
    c_result = sess.run(None, {'a': a_np, 'b': b_np})[0]

    print("\nInference successful (if this line is reached).")
    print("Output C shape:", c_result.shape)
    print("Output C value:", c_result)

   
except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
    print("\n--- ONNX Runtime Execution Error Caught ---")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
 
    
except Exception as e:
    print("\n--- An Unexpected Error Occurred ---")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")

print("\nScript execution continued after inference attempt.")

# ---------------------------------------------------------------------------

""" ONNX Add operator: test overflow int32 """

# ---------------------------------------------------------------------------

x_val_info = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.INT32, [])
y_val_info = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.INT32, [])

node_def = onnx.helper.make_node(
 'Add',
 ['x', 'y'],
 ['z']
)
graph_def = onnx.helper.make_graph(
 [node_def],
 'test-add-scalars',
 [x_val_info, y_val_info],
 [onnx.helper.make_tensor_value_info('z', onnx.TensorProto.INT32, [])]
)

onnx_model = onnx.helper.make_model(graph_def, producer_name='onnx-scalar-add-test')
# Set opset
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Verify
onnx.checker.check_model(onnx_model)
print("\nTest 1: Scalar Add\n")
print(onnx.helper.printable_graph(onnx_model.graph))

# Run
x_np = numpy.array(2**31 - 1, dtype=numpy.int32) # scalar, shape = []
y_np = numpy.array(1, dtype=numpy.int32)
sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
providers=["CPUExecutionProvider"])
z = sess.run(None, {'x': x_np, 'y': y_np})[0]

print("x:", x_np)
print("y:", y_np)
print("z = x + y :", z)

# -------------------------------
#

""" ONNX Add operator: test overflow float32 """

#
# -------------------------------

# Create model inputs and outputs
x_val_info = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [])
y_val_info = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [])
z_val_info = onnx.helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT, [])

# Create Add node
node_def = onnx.helper.make_node(
    'Add',
    ['x', 'y'],
    ['z']
)

# Create Graph
graph_def = onnx.helper.make_graph(
    [node_def],
    'test-add-scalars',
    [x_val_info, y_val_info],
    [z_val_info]
)

# Create Model
onnx_model = onnx.helper.make_model(graph_def, producer_name='onnx-scalar-add-test')


del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8


onnx.checker.check_model(onnx_model)

print("\nTest: Scalar Add Overflow Float32\n")
print(onnx.helper.printable_graph(onnx_model.graph))

# -------------------------------
# Overflow Test
# -------------------------------

# Define test inputs close to float32 max
float32_max = numpy.finfo(numpy.float32).max  # 3.4028235e+38

x_ov = numpy.array(float32_max, dtype=numpy.float32)
y_ov = numpy.array(1e+38, dtype=numpy.float32)  

# Create Inference Session
sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(),
                            providers=["CPUExecutionProvider"])

# Run inference
z = sess.run(None, {'x': x_ov, 'y': y_ov})[0]


print("x:", x_ov)
print("y:", y_ov)
print("z = x + y :", z)

