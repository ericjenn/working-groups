# ScalarType implementations in whyML

This folder *scalar_type* contains two distinct implementations of a `Scalar` type and an associated `Add` operator. The purpose is to explore two different modeling approaches and verify that they produce identical results.

---

## Folder Descriptions

### `scalar_&_add_generic/`

This folder represents the final, intended design for the scalar type.

-   **Abstract Module Approach**: `Scalar` is implemented as an abstract module.
-   **Concrete Instances**: Different modules, such as `ScalarInt32` and `ScalarFloat32`, represent the specific instances of the abstract `ScalarType`.
-   **Operator**: The `Add` operator is implemented to be compatible with this generic, modular structure.
-   **Verification**: The implementation includes reference code generated in OCaml and a test suite. The tests are the same as those used for the "simple" implementation to ensure coherence.

### `scalar_&_add_simple/`

This folder contains a "naive" or more direct implementation of the scalar type.

-   **Pattern-Matching Approach**: It defines `ScalarType` using the whyML keyword `match value with ...`.
-   **Operator**: The `Add` operator is specifically implemented to work with this pattern-matching structure.
-   **Verification**: The same test suite as the "generic" version is used to verify the implementation.
-   **Test Cases**: Tests cover trivial cases (e.g., `1+2`), `int32` overflow, and `float32` overflow.
-   **Code Generation**: Reference OCaml code is also generated for this version.

---

## Results

 **Both implementations produce the same results** on the shared test suite. This confirms that both the abstract modular approach and the direct pattern-matching approach are coherent.