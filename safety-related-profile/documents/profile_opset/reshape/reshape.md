# Convention
## Notations
- The notation $N(X)$ denotes the _number of elements_ in tensor $X$.
- Notations $h(X)$ and $w(X)$ respectively denote the _height_ and the _width_ of tensor $X$.


## Usage of fonts
- Inputs, outputs, and attributes are represented using a non-serif font. For instance, the "condition" attribute is represented by `condition`.

## Tags
- Restrictions with respect to the ONNX standard are indicated in the text with the tag `[Ri]` where `i` is a number. A synthesis of all restrictions is given in section "Restrictions".

# `reshape` operator (real)

### Restrictions


### Signature
`Y = reshape(X, shape)`
where
- `shape`: target shape of the tensor `Y`. Must be compatible with the shape of `X`, i.e., `X` and `Y` must have the same number of elements.
- `X`: input tensor

#### Informal specification

The `reshape` operator generates a new tensor `Y` that contains the same input data as the input tensor `X` but with a different shape.


#### Inputs and outputs

##### `shape`

Tensor `shape` is a tensor of integral values.
The product of the components of `shape` (number of elements of `Y`) must be equal to the product of the components of the shape of `X` (number of elements of `X`).
One component of `shape` can optionally be -1. This value is then derived from the other components and the shape of `X`.

##### SB 
Should this feature be allowed?

#### Attributes


###### Constraints

- (C1) Consistency in number of elements
    - Statement: The `shape` must be specified such that `Y` has the same number of elements as `X`.

#### Outputs

##### `Y`

The output tensor `Y` contains the same data as the input tensor `X`, just interpreted with a new shape.

##### SB 
The documentation of the Reshape operator is problematic and incomplete. It refers to the corresponding documentation of the numpy reshape function.
This, however, is also incomplete. In particular, this statement is problematic: `It is not always possible to change the shape of an array without copying the data.`
However, it is not explained under which circumstances this could happen.
This has major implications for the implementation on the target hardware.
For example, in a C implementation of a network, Reshape does not require any implementation at all if all tensors are represented as plain C arrays.

#### Attributes

#### `allowzero (default 0)`
Must be 0 `[R1]`.
The ONNX standard allows components of the `shape` input to be 0. 
This signifies that the corresponding component of the dimension value of the input tensor is copied dynamically.
SB: Should this feature be allowed?
