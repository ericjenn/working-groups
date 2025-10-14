# Explanations of Conv corner and edge cases generation

## Assumptions
First of all let us explain what we mean by corner and edge cases.
- **Corner cases** are the extreme values of a given parameter, such as the minimum and maximum values that the parameter can take.

- **Edge cases** are values that are just above or below the minimum and maximum values, respectively.

## Conv tests
For concat operator we checked the **corner and edge cases** for: **dx0, dx1, dx2, dx3, dw0, strides, auto_pad and pads**.
We think its **irrelevant** to check the corner and edge cases for **db0** because of **B [C1]** and **kernel shape** because of **W [C3]**.
We didn't check the corner and edges cases for **dw2, dw3 and dilations** because the way we generate them is not independent. 

We consider that **dx0, dx1, and dw0** as lines since they are independent and we check the corner and edge cases for them.

For **dx2** and **dx3** we consider them as a **rectangle**. We check each **corner** of the rectangle, a **middle point of each rectangle edge** and a **middle point of the rectangle**.

For **pads** (when **auto_pad == NOTSET**) we consider a **4D cube** and we check each corner of the "4D cube". All **16 corners**, example, (min,min,min,min), (min,min,min,max), ... (min,max,max,max), (max,max,max,max) where min is the minimum value of pads and max is the maximum value of pads.
Also we check a middle point of each edge of the "4D cube" and a middle point of the "4D cube".

For **auto_pad** we check if all the possible values are being generated (Except **VALID**)

For **strides** we consider them as a **rectangle***. We check each **corner** of the rectangle, a **middle point of each rectangle edge** and a **middle point of the rectangle**.