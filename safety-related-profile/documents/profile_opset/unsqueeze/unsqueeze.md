# Preliminary remarks

# `Unsqueeze` operator (real)

### Restrictions
The following restrictions apply to the `Unsqueeze` operator for the SONNX profile:
- The axes input must be constant. 
Otherwise, the operator reshapes the input tensor to a dynamic shape determined at runtime.`[R1]`

### Signature
`Y = Unsqueeze(X, axes)`
where
- `X`: input tensor to reshape
- `axes`: list of integer values indicating where to insert one-dimensional entries into the shape of `X`.

## SB: 
Not clear how this works. For example, if shape = `(2, 3, 4)` and `axes=[0, 1, 3]`, what is the output shape?
I assume from the documentation that it will be `(1, 1, 2, 1, 3, 4)`, but it is not completely clear.

### Remarks
In a pure C implementation of a neural network that operates on plain one-dimensional C arrays, 
or an equivalent implementation in another language, 
the `Unsqueeze` operator does not require any implementation at all.
It does not change the data layout of the underlying one-dimensional array.