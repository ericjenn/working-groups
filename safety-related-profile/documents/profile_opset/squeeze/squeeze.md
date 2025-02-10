# Preliminary remarks

# `Squeeze` operator (real)

### Restrictions
The following restrictions apply to the `Squeeze` operator for the SONNX profile:
- The axes input is optional but if it is provided it must be constant. 
- Axes must not contain duplicate entries.
Otherwise, the operator reshapes the input tensor to a dynamic shape determined at runtime.`[R1]`

### Signature
`Y = Squeeze(X, axes)`
where
- `X`: input tensor to reshape
- `axes`: optional list of integer values indicating the dimensions to squeeze. All elements in `axes` must refer to single-dimension shape entries.
For example, if `X` has shape `[1, 2, 1, 3, 1, 1]`, then `axes` could be `[0, 2, 4, 5]` to squeeze the tensor to the shape `[2, 3]`.


### Remarks
In a pure C implementation of a neural network that operates on plain one-dimensional C arrays, 
or an equivalent implementation in another language, 
the `Squeeze` operator does not require any implementation at all.
It does not change the data layout of the underlying one-dimensional array.