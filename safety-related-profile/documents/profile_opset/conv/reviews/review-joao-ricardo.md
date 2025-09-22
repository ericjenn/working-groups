# Doubts and Suggested Corrections

### Wrong Image Representation
- According to our calculations (and the ones presented in the python script), the image does not reflect the correct convolution for the output tensor. The error occurs for the entries: $Y[3, 1]$ and $Y[3, 2]$. Instead of $5$ we believe the value should be $4$.
- We validated it afterwards with the ONNX (see the cript provided).

### Pads
- When referred for the first time, `pads` is said to be: $\text{pads} = [\text{left}, \text{right}, \text{top}, \text{bottom}]$, although, the image suggests otherwise, namely $\text{pads} = [\text{left}, \text{top}, \text{right}, \text{bottom}]$. 
- Additionally, on the first image, under the arrow it states $\text{pads} = (1, 3, 2, 2)$. Shouldn't it be $\text{pads} = [1, 2, 2, 2]$?

### Strides
- For the strides definition, we intend that it represents the magnitude of the shift of the kernel during the convolution. According to the image, strides is defined: $\text{strides} = [\text{shift between columns}, \text{shift between lines}]$. Shouldn't it be $[\text{shift between lines}, \text{shift between columns}]$?

### Convolution Formula
- **Assuming this definition of strides:** $\text{strides} = [\text{shift between columns}, \text{shift between lines}]$,  presented in the image is correct, then the output tensor definition is not correct

$$
\begin{gathered}
   Y[b, c, m, n] = \sum_{i=0}^{dW_1-1} \sum_{j=0}^{dW_2-1} \sum_{z=0}^{dW_3-1} \\
   (X_p[b,i,m \cdot \text{strides}[0]+ j , n \cdot \text{strides}[1]+ z ] \cdot W_d[c, i, j, z]) \\
   + B_b[c]
\end{gathered}
$$

- According to this definition, we are stating the iterations over the lines should take into account the stride of the columns (and the same for the other axis). Either the strides arrangement  or the formula is wrong. If we switch the strides arrangement , then the formula is correct. 
Actually, when we tested the operator using our script, we couldn't even run the code as it said "index out of bound". 
When we switched the strides for $[\text{lines}, \text{columns}]$ everything worked out.


### dY2 and dY3 Calculations
- In the constraints section, more precisely for the C3 constraints, something might be wrong since those formulas do not provide the correct method for calculating $dY2$ and $dY3$.

---

#### $dY2$ Calculation

According to this formula:

$$
dY_2 = \left\lfloor \frac{\alpha - \left(dilation[0] \cdot dW_2 - 1\right)}{strides[0]} \right\rfloor + 1
\quad \text{where}\quad \alpha = dX_2 + pads[0] + pads[2]
$$
- $\alpha = 8 + 1 + 2 = 11$
- $\text{dilations}[0] \times dW2 - 1 = 2 \times 2 - 1 = 3$
- $(11 - 3) / \text{strides}[0] = 4$
- $dY2 = 4 + 1 = 5$ (**it should be 4**)

### Error Causes
- Using the wrong indexes of padding to calculate the number of lines (should use indexes 1 and 3)
- Not taking into account that for dilations greater than 2 it is not enough to subtract only one unit

### Suggested Correction

As the rationale states: The size of the output is determined by the number of times the kernel can be applied on a given spatial axis.

Therefore, we have to calculate how many times the difference between the size of the input and the size of the kernel allows a stride to be done.

**Proposal**

**Assuming this definition of strides:** $\text{strides} = [\text{shift between columns}, \text{shift between lines}]$
$$
dY2 = \left\lfloor \frac{\alpha - \theta}{strides[1]} \right\rfloor + 1 \quad 
\text{where}$$
$$ 
\alpha = dX2 + \text{pads}[1] + \text{pads}[3]$$
$$\quad
\theta = (\text{dilations}[0] * (dW - 1)) + 1
$$

For a more concrete description of the theta calculation we can think of this as a dilation of the matrix by the factor $x$ is a given axis. 
- $x$ = 1: Not dilated (same number of elements in the axis)
- $x$ = 2: Each element in that axis is separated with one 0 between
- $x$ = 3: Each element  in that axis is separated with two 0s 
- $x$ = d: Each element  in that axis is separated with (d - 1) 0s
- We are adding 0s **only between** the elements already present on the matrix. Therefore:
$$
\theta = (\text{dilations}[0] * (dW - 1)) + 1
$$ 

The meaning of this calculation is as follows:
- Every element of the initial matrix, except the last, will have $(\text{dilations}[0] - 1)$ zeros added to its right.
- The last element does not receive any zeros after it, so we add $+1$ to account for its position.

According to the proposed formula:
- $\alpha = 8 + 2 + 2 = 12$
- $\theta = 2 * (2 - 1) + 1 = 3$
- $\left\lfloor \frac{12 - 3}{strides[0]} \right\rfloor = \left\lfloor \frac{9}{3} \right\rfloor$ = 3
- $dY2 = 3 + 1 = 4$

For instance this formula was verified for multiple inputs. You can check that in the script provided.
The same should be done concerning the **$dY3$** formula.
For this case, we propose:
$$
dY3 = \left\lfloor \frac{\beta - \gamma}{strides[0]} \right\rfloor + 1 \quad 
\text{where}$$
$$ 
\beta = dX3 + \text{pads}[0] + \text{pads}[2]$$
$$\quad
\gamma = (\text{dilations}[1] * (dW - 1)) + 1
$$

**Note: If the strides structure is redesigned  its effects should be considered in all these definitions**