# Hypothesis Tests Doubts for Conv operator

## Conv

### Inputs/attributes generation
What would be the best approach for generating inputs that satisfy the **X[C3]** constraint?

We know that ideally Hypothesis should generate input values that already satisfied this equation. We have been searching for ways to do that and we found no viable way to do so. 

Therefore, our aproach was to generate two unconstrained variables, and then derive the remaining two based on those, ensuring the final inputs satisfy the required condition. Another possibility was to generate only unconstrainted values and then filter by those who satisfy the equation (this would be computationally expensive and would probably defeat the purpose of using Hypothesis — many edge cases would be lost).

### Auto_Pad = "Valid"
There is no documentation stating how this type of configuration works.
Therefore we are unable to calculate dY dimensions (useful for the generation phase) and we cannot reason about the results it produces.
How should we proceed?
Should we ignore this configuration for Auto_Pad or is there any documentation that we haven't looked yet.
