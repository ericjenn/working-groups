- (C1) Shape consistency
    - Statement: The shapes of `A` and `B` must be the same. $N(A)=N(B)$ `[R1]` Broadcastable tensor is forbidden. `[R4]`
- (C2) No zero elements for defined division
    - Statement: All elements of tensor `B` must be non-zero to avoid division by zero. $\forall B[i] \neq 0$ else the result will be inf `[R5]`

##SB: C2 is equivalent to enforcing the tensor `B` to be constant or to originate from a dynamic operation that ensures that its values cannot be zero, which might be hard.

