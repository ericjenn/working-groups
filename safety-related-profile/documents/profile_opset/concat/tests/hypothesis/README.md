# Explanations of Concat corner and edge cases generation

## Assumptions
First of all let us explain what we mean by corner and edge cases.
- **Corner cases** are the extreme values of a given parameter, such as the minimum and maximum values that the parameter can take.

- **Edge cases** are values that are just above or below the minimum and maximum values, respectively.

## Concat tests
For the concat operator we just **checked both the corner and edge cases** for the **number_of_input_tensors** and the **shape_size_input**.

We consider both (**number of input tensors and shape of input tensors**) as **lines** since they are independent and we checked the corner and edge cases for both of them.

We **didn't check** the corner and edges cases for the **concatenation_axis** and for **input_tensors_shapes** because the way we generate them is not independent (actualy **concatenation_axis** depends on the number of axis generated while **input_tensor_shapes** depends on the number of input tensors).

For example, imagine that my number of input tensors is 1, if i try to check the corner cases and the edge cases for this example it will fail, because i will just have one tensor. It would be impossible to check the corner and edges cases in this situation.

The same happens for the **concatenation_axis**.