# Preliminary remarks

## Types


# `Softmax` operator

## SB: `Softmax` can overflow.
Normal softmax implementations compute the largest value in the input tensor `X` and subtract this value from all components of `X` before applying the exponential function.
This ensures that `Softmax` cannot overflow.
This, however, implies that the safety of this operator is dependent on its specific implementation.