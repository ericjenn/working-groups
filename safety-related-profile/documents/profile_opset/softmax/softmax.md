# Preliminary remarks

## Types


# `Exp` operator

## SB: `Exp` can overflow. 
The precise range for which it produces finite results depends on the datatype (float16, float32, float64).
It is strictly only safe if the input domain is ensured to be within this range.
