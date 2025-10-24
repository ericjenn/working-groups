# Contents

- **slice** operator for type [real](#real)
- **slice** operator for types [INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, DOUBLE, BFLOAT16, BOOL, STRING](#types)

Based on ONNX documentation version 13.

<a id="real"></a>
# **slice** (real)

## Signature
$Y = \text{slice}(X,S,E,A,K)$

where:
- `X`: input tensor
- `A`: 1-D tensor of axes to slice
- `S`: 1-D tensor with starting indices of corresponding axis in `A`
- `E`: 1-D tensor with ending indices of corresponding axis in `A`
- `K`: 1-D tensor of steps for each axis in `A`
- `Y`: output tensor


## Restrictions
The following restrictions apply to the **slice** operator for the SONNX profile:

| Restriction | Statement                                                   | Origin                                                                                      |
|-------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `[R1]`  <a id="R1"></a>   | Input `A` must be set                           | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R2]`  <a id="R2"></a>   | All axes must be specified for input `A`        | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R3]`  <a id="R3"></a>   | Input `K` must be set                           | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R4]`   <a id="R4"></a>  | Sparse tensors are not supported              | General restriction [GR1](../general_restrictions.md#GR1) |
| `[R5]`     <a id="R5"></a>     | Shape of tensors shall be explicit          | General restriction [GR2](../general_restrictions.md#GR2) |
| `[R6]`     <a id="R6"></a>     | Positive steps must have starting positions lower or equal than ending positions | Transient |
| `[R7]`     <a id="R7"></a>     | Negative steps must have starting positions greater or equal than ending positions | Transient |

## Informal specification


Operator **slice** extracts from `X` a subtensor defined by the axes listed in `A`. 

For each axis, `i` slicing: 
- starts at $S[A[i]]$ 
- slides with a step of $K[A[i]]$ 
- stops strictly before $E[A[i]]$

The result is stored in output tensor `Y`.

$$
  \begin{align*}
Y[a, b, \ldots, z] = X[&\text{S}[A[0]] + a \cdot \text{K}[A[0]], \\
                       &\text{S}[A[1]] + b \cdot \text{K}[A[1]], \\
                       &\vdots \\
                       &\text{S}[A[r-1]] + z \cdot \text{K}[A[r-1]]]
\end{align*}
$$

Where:
- $r$ is the rank of tensor `X`
- $a \in [0, dY_0-1]$ is the index along the first dimension of output tensor `Y`
- $b \in [0, dY_1-1]$ is the index along the second dimension of output tensor `Y`
- $\vdots$
- $z \in [0, dY_{r-1}-1]$ is the index along the last dimension of output tensor `Y`
- $i \in \{0, ...\ r-1\}$ is the index of the current axis


- $space_{i} = E[A[i]] - S[A[i]]$ is the available space along axis `i` for slicing
- <a id="f"></a> $f = \begin{cases} 0 & \text{if}\quad (space_{i} \bmod K[A[i]] = 0) \\ 1 & \text{otherwise} \end{cases}$ 
- $dY_{i} = \left\lfloor \frac{\text{space}_{i}}{K[A[i]]} \right\rfloor + f$ is the dimension of the output tensor `Y` along axis `i` 

is a flag indicating whether there is a remainder when dividing the available space by the step size along axis `i`.

The effect of the operator is illustrated on the following figure. In this example
- shape of `X` is ($5, 6$)
- shape of `A` is ($2$) with values `[0, 1]`
- shape of `S` is ($2$) with values `[0, 1]`
- shape of `E` is ($2$) with values `[4, 6]`
- shape of `K` is ($2$) with values `[1, 2]`

Finally, the following figure illustrates operator Slice applied on input `X` with the above parameters, resulting in output `Y` with shape ($4, 3$).

<img src="./imgs/example.png" alt="drawing" width="100%"/>

In order to understand how the shape of output `Y` is computed, the following figure illustrates the slicing process along each axis.

For the horizontal axis and starting from the black square we can do two steps and the third one is incomplete. Once there is an incomplete step we have to add one [<b><span style="font-family: 'Courier New', monospace">f</span></b>](#f) (ensuring that the dimension is equal to the total number of steps, completed or not). In this case it is clear that the dimension should be three yet only 2 steps were completed.

For the vertical axes there is no incomplete steps and the dimension is the same as the number of steps completed.
<img src="./imgs/m_explanation.png" alt="drawing" width="100%"/>

## Error conditions
No error condition

## Inputs

### $X$: real
Tensor `X` is the input tensor from which a subtensor will be extracted.

### Constraints

- `[C1]` <a id="C1ra"></a> Dimensional consistency
    - Statement:
        - $rank(X) = d{S_0} = d{E_0} = d{A_0} = d{K_0}$
    - Rationale: This ensures that the slicing parameters are well-defined for each axis of `X`. [<b><span style="font-family: 'Courier New', monospace">[R2]</span></b>](#R2)
-  `[C2]`<a id="C2ra"></a> Rank consistency
   - Statement: The rank of tensor `X` and `Y` must be the same.
   - Rationale: Slicing does not change the rank of the tensor.

### $S$: real
Tensor `S` is a tensor containing the starting indices for each axis specified in `A`.

Tensor `S` must be a 1-D tensor.

### Constraints
- `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ra) on tensor `X`.

- `[C2]` <a id="C3ra"></a> Value Domain
    - Statement: The adjusted starting indices must be clamped to valid ranges based on the stepping direction.
       $$
       \forall i \in [0, r-1], \quad
       S[A[i]] \in 
       \begin{cases} 
       [-d{X_i}, d{X_i}] & \text{if } K[A[i]] > 0 \\
       [-d{X_i}, d{X_i} -1] & \text{if } K[A[i]] < 0 
       \end{cases}
       $$
        Where
        - $r$ is the rank of tensor `X`.
        - $i$ is the axis index.

- `[C3]`<a id="C8ra"></a> Steps and starting/ending indices consistency
   - Statement: Ensuring that output dimensions are non negative and follow this [<b><span style="font-family: 'Courier New', monospace">formula</span></b>](#dY). [<b><span style="font-family: 'Courier New', monospace">[R5]</span></b>](#R5) [<b><span style="font-family: 'Courier New', monospace">[R6]</span></b>](#R6)


         
### $E$: real
Tensor `E` is a tensor containing the ending indices (exclusive) for each axis specified in `A`.

Tensor `E` must be a 1-D tensor.

### Constraints
- `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ra) on tensor `X`.

- `[C2]` <a id="C4ra"></a> Value Domain
    - Statement: The adjusted ending indices must be clamped to valid ranges based on the stepping direction.
       $$
       \forall i \in [0, r-1], \quad
       E[A[i]] \in 
       \begin{cases} 
       [-d{X_i}, d{X_i}] & \text{if } K[A[i]] > 0 \\
       [-d{X_i}-1, d{X_i} -1] & \text{if } K[A[i]] < 0 
       \end{cases}
       $$
        Where
        - $r$ is the rank of tensor `X`.
        - $i$ is the axis index.

  - `[C3]` Steps and starting/ending indices consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C3]</span></b>](#C8ra) on tensor `S`. [<b><span style="font-family: 'Courier New', monospace">[R5]</span></b>](#R5) [<b><span style="font-family: 'Courier New', monospace">[R6]</span></b>](#R6)


### $A$: real
Tensor `A` is a tensor containing the axes along which to slice.

Tensor `A` must be a 1-D tensor.
### Constraints
 - `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ra) on tensor `X`.   

 - `[C2]` <a id="C5ra"></a> Value Domain
   - Statement: Each axis specified in `A` must be a valid axis index for tensor `X`.
       $$
       \forall i \in [0, r-1], \quad
       A[i] \in [-r, r-1]
       $$
        Where
        - $r$ is the rank of tensor `X`.
        - $i$ is the axis index.
    - Rationale: This ensures that the slicing axes are valid for the input tensor.
 - `[C3]` <a id="C6ra"></a> Uniqueness
   - Statement: After normalizing negative indices, all axes in `A` must be unique: 
     $$\forall i, j \in [0, r-1], \; (A[i] + r) \bmod r = ((A[j] + r) \bmod r)\implies (i = j)$$
   - Rationale: Prevents ambiguity when the same axis is specified multiple times using different representations (negative and positive).

### $K$: real
Tensor `K` is a tensor containing the steps for each axis specified in `A`.

Tensor `K` must be a 1-D tensor.

### Constraints
 - `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ra) on tensor `X`.

 - `[C2]` <a id="C7ra"></a> Value Domain
   - Statement: Each step in `K` must be a valid step size for the corresponding axis in `A`.
       $$
       \forall i \in [0, r-1], \quad
       K[i] \in \mathbb{Z} \setminus \{0\}
       $$
       Where
       - $r$ is the rank of tensor `X`.
       - $i$ is the axis index.
   - Rationale: This ensures that the steps are well-defined for each axis of `X`.
  
  - `[C3]` Steps and starting/ending indices consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C3]</span></b>](#C8ra) on tensor `S`. [<b><span style="font-family: 'Courier New', monospace">[R5]</span></b>](#R5) [<b><span style="font-family: 'Courier New', monospace">[R6]</span></b>](#R6)


## Outputs

### $Y$: real
Tensor `Y` is the output tensor containing the sliced subtensor from `X`.
### Constraints
 - `[C1]` Rank consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C2]</span></b>](#C2ra) on tensor `X`.

 - `[C2]` Consistency between the shape of tensors `S`, `E`, `A`, `K`, and `Y`  
   - Statement: 
        - $\forall i \in [0, r-1], \quad dY_{i} \ge 0$

        Where
    - $i \in \{0, ...\ r-1\}$ is the index of the current axis

    - $space_{i} = E[A[i]] - S[A[i]]$ is the available space along axis `i` for slicing
    - $f = \begin{cases} 0 & \text{if}\quad (space_{i} \bmod K[A[i]] = 0) \\ 1 & \text{otherwise} \end{cases}$ 
    - <a id="dY"></a> $dY_{i} = \left\lfloor \frac{\text{space}_{i}}{K[A[i]]} \right\rfloor + f$ is the dimension of the output tensor `Y` along axis `i` 

   - Rationale: The size of the output tensor `Y` is determined by the slicing parameters applied to tensor `X`.

## Attributes

Operator **slice** has no attribute.

## Formal specification
 
See the Why3 specification.

## Numerical Accuracy

Operator **slice** does not introduce any numerical error. Hence, for all valid indexes $i$,

<a id="types"></a>
# **slice** (type, itype, itype, itype, itype)

Where type is in {INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, DOUBLE, BFLOAT16, BOOL, STRING}

Where itype is in {INT32, INT64}

## Signature
$Y = \text{slice}(X,S,E,A,K)$

where:
- `X`: input tensor
- `A`: 1-D tensor of axes to slice
- `S`: 1-D tensor with starting indices of corresponding axis in `A`
- `E`: 1-D tensor with ending indices of corresponding axis in `A`
- `K`: 1-D tensor of steps for each axis in `A`
- `Y`: output tensor


## Restrictions
The following restrictions apply to the **slice** operator for the SONNX profile:

| Restriction | Statement                                                   | Origin                                                                                      |
|-------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `[R1]`  <a id="R1t"></a>   | Input `A` must be set                           | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R2]`  <a id="R2t"></a>   | All axes must be specified for input `A`        | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R3]`  <a id="R3t"></a>   | Input `K` must be set                           | [No default values](../../../deliverables/reqs/reqs.md#no_default_value) |
| `[R4]`   <a id="R4t"></a>  | Sparse tensors are not supported              | General restriction [GR1](../general_restrictions.md#GR1) |
| `[R5]`     <a id="R5t"></a>     | Shape of tensors shall be explicit          | General restriction [GR2](../general_restrictions.md#GR2) |
| `[R6]`     <a id="R6t"></a>     | Positive steps must have starting positions lower or equal than ending positions | Transient |
| `[R7]`     <a id="R7t"></a>     | Negative steps must have starting positions greater or equal than ending positions | Transient |

## Informal specification


Operator **slice** extracts from `X` a subtensor defined by the axes listed in `A`. 

For each axis, `i` slicing: 
- starts at $S[A[i]]$ 
- slides with a step of $K[A[i]]$ 
- stops strictly before $E[A[i]]$

The result is stored in output tensor `Y`.

$$
  \begin{align*}
Y[a, b, \ldots, z] = X[&\text{S}[A[0]] + a \cdot \text{K}[A[0]], \\
                       &\text{S}[A[1]] + b \cdot \text{K}[A[1]], \\
                       &\vdots \\
                       &\text{S}[A[r-1]] + z \cdot \text{K}[A[r-1]]]
\end{align*}
$$

Where:
- $r$ is the rank of tensor `X`
- $a \in [0, dY_0-1]$ is the index along the first dimension of output tensor `Y`
- $b \in [0, dY_1-1]$ is the index along the second dimension of output tensor `Y`
- $\vdots$
- $z \in [0, dY_{r-1}-1]$ is the index along the last dimension of output tensor `Y`
- $i \in \{0, ...\ r-1\}$ is the index of the current axis


- $space_{i} = E[A[i]] - S[A[i]]$ is the available space along axis `i` for slicing
- <a id="f"></a> $f = \begin{cases} 0 & \text{if}\quad (space_{i} \bmod K[A[i]] = 0) \\ 1 & \text{otherwise} \end{cases}$ 
- $dY_{i} = \left\lfloor \frac{\text{space}_{i}}{K[A[i]]} \right\rfloor + f$ is the dimension of the output tensor `Y` along axis `i` 

is a flag indicating whether there is a remainder when dividing the available space by the step size along axis `i`.

The effect of the operator is illustrated on the following figure. In this example
- shape of `X` is ($5, 6$)
- shape of `A` is ($2$) with values `[0, 1]`
- shape of `S` is ($2$) with values `[0, 1]`
- shape of `E` is ($2$) with values `[4, 6]`
- shape of `K` is ($2$) with values `[1, 2]`

Finally, the following figure illustrates operator Slice applied on input `X` with the above parameters, resulting in output `Y` with shape ($4, 3$).

<img src="./imgs/example.png" alt="drawing" width="100%"/>

In order to understand how the shape of output `Y` is computed, the following figure illustrates the slicing process along each axis.

For the horizontal axis and starting from the black square we can do two steps and the third one is incomplete. Once there is an incomplete step we have to add one [<b><span style="font-family: 'Courier New', monospace">f</span></b>](#f) (ensuring that the dimension is equal to the total number of steps, completed or not). In this case it is clear that the dimension should be three yet only 2 steps were completed.

For the vertical axes there is no incomplete steps and the dimension is the same as the number of steps completed.
<img src="./imgs/m_explanation.png" alt="drawing" width="100%"/>

## Error conditions
No error condition

## Inputs

### $X$: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, DOUBLE, BFLOAT16, BOOL, STRING
Tensor `X` is the input tensor from which a subtensor will be extracted.

### Constraints

- `[C1]` <a id="C1ta"></a> Dimensional consistency
    - Statement:
        - $rank(X) = d{S_0} = d{E_0} = d{A_0} = d{K_0}$
    - Rationale: This ensures that the slicing parameters are well-defined for each axis of `X`. [<b><span style="font-family: 'Courier New', monospace">[R2]</span></b>](#R2t)
-  `[C2]`<a id="C2ta"></a> Rank consistency
   - Statement: The rank of tensor `X` and `Y` must be the same.
   - Rationale: Slicing does not change the rank of the tensor.

### $S$: INT32, INT64
Tensor `S` is a tensor containing the starting indices for each axis specified in `A`.

Tensor `S` must be a 1-D tensor.

### Constraints
- `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ta) on tensor `X`.

- `[C2]` <a id="C3ta"></a> Value Domain
    - Statement: The adjusted starting indices must be clamped to valid ranges based on the stepping direction.
       $$
       \forall i \in [0, r-1], \quad
       S[A[i]] \in 
       \begin{cases} 
       [-d{X_i}, d{X_i}] & \text{if } K[A[i]] > 0 \\
       [-d{X_i}, d{X_i} -1] & \text{if } K[A[i]] < 0 
       \end{cases}
       $$
        Where
        - $r$ is the rank of tensor `X`.
        - $i$ is the axis index.

- `[C3]`<a id="C8ta"></a> Steps and starting/ending indices consistency
   - Statement: Ensuring that output dimensions are non negative and follow this [<b><span style="font-family: 'Courier New', monospace">formula</span></b>](#dYt). [<b><span style="font-family: 'Courier New', monospace">[R5]</span></b>](#R5t) [<b><span style="font-family: 'Courier New', monospace">[R6]</span></b>](#R6t)


         
### $E$: INT32, INT64
Tensor `E` is a tensor containing the ending indices (exclusive) for each axis specified in `A`.

Tensor `E` must be a 1-D tensor.

### Constraints
- `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ta) on tensor `X`.

- `[C2]` <a id="C4ta"></a> Value Domain
    - Statement: The adjusted ending indices must be clamped to valid ranges based on the stepping direction.
       $$
       \forall i \in [0, r-1], \quad
       E[A[i]] \in 
       \begin{cases} 
       [-d{X_i}, d{X_i}] & \text{if } K[A[i]] > 0 \\
       [-d{X_i}-1, d{X_i} -1] & \text{if } K[A[i]] < 0 
       \end{cases}
       $$
        Where
        - $r$ is the rank of tensor `X`.
        - $i$ is the axis index.

- `[C3]` Steps and starting/ending indices consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C3]</span></b>](#C8ta) on tensor `S`. [<b><span style="font-family: 'Courier New', monospace">[R5]</span></b>](#R5t) [<b><span style="font-family: 'Courier New', monospace">[R6]</span></b>](#R6t)



### $A$: INT32, INT64
Tensor `A` is a tensor containing the axes along which to slice.

Tensor `A` must be a 1-D tensor.
### Constraints
 - `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ta) on tensor `X`.   

 - `[C2]` <a id="C5ta"></a> Value Domain
   - Statement: Each axis specified in `A` must be a valid axis index for tensor `X`.
       $$
       \forall i \in [0, r-1], \quad
       A[i] \in [-r, r-1]
       $$
        Where
        - $r$ is the rank of tensor `X`.
        - $i$ is the axis index.
    - Rationale: This ensures that the slicing axes are valid for the input tensor.
 - `[C3]` <a id="C6ta"></a> Uniqueness
   - Statement: After normalizing negative indices, all axes in `A` must be unique: 
     $$\forall i, j \in [0, r-1], \; (A[i] + r) \bmod r = ((A[j] + r) \bmod r)\implies (i = j)$$
   - Rationale: Prevents ambiguity when the same axis is specified multiple times using different representations (negative and positive).

### $K$: INT32, INT64
Tensor `K` is a tensor containing the steps for each axis specified in `A`.

Tensor `K` must be a 1-D tensor.

### Constraints
- `[C1]` Dimensional consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C1]</span></b>](#C1ta) on tensor `X`.

- `[C2]` <a id="C7ta"></a> Value Domain
   - Statement: Each step in `K` must be a valid step size for the corresponding axis in `A`.
       $$
       \forall i \in [0, r-1], \quad
       K[i] \in \mathbb{Z} \setminus \{0\}
       $$
       Where
       - $r$ is the rank of tensor `X`.
       - $i$ is the axis index.
   - Rationale: This ensures that the steps are well-defined for each axis of `X`.
  
- `[C3]` Steps and starting/ending indices consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C3]</span></b>](#C8ta) on tensor `S`. [<b><span style="font-family: 'Courier New', monospace">[R5]</span></b>](#R5t) [<b><span style="font-family: 'Courier New', monospace">[R6]</span></b>](#R6t)


## Outputs

### $Y$: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FP16, FP32, DOUBLE, BFLOAT16, BOOL, STRING
Tensor `Y` is the output tensor containing the sliced subtensor from `X`.
### Constraints
 - `[C1]` Rank consistency
   - Statement: see constraint [<b><span style="font-family: 'Courier New', monospace">[C2]</span></b>](#C2ta) on tensor `X`.

 - `[C2]` Consistency between the shape of tensors `S`, `E`, `A`, `K`, and `Y`  
   - Statement: 
        - $\forall i \in [0, r-1], \quad dY_{i} \ge 0$

        Where
    - $i \in \{0, ...\ r-1\}$ is the index of the current axis

    - $space_{i} = E[A[i]] - S[A[i]]$ is the available space along axis `i` for slicing
    - $f = \begin{cases} 0 & \text{if}\quad (space_{i} \bmod K[A[i]] = 0) \\ 1 & \text{otherwise} \end{cases}$ 
    - <a id="dYt"></a> $dY_{i} = \left\lfloor \frac{\text{space}_{i}}{K[A[i]]} \right\rfloor + f$ is the dimension of the output tensor `Y` along axis `i` 

   - Rationale: The size of the output tensor `Y` is determined by the slicing parameters applied to tensor `X`.

## Attributes

Operator **slice** has no attribute.

## Formal specification
 
See the Why3 specification.

## Numerical Accuracy

Operator **slice** does not introduce any numerical error. Hence, for all valid indexes $i$,
