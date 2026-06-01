# solve_arcopt_linear_program

## Description

Solve an ARCOPT linear programming problem using scipy's linear programming solver.

## Usage

```python
from solve_arcopt_linear_program import solve_arcopt_linear_program
result = solve_arcopt_linear_program(<spec>)
print(result)
```

## Inputs
spec (str): ARCOPT specification string in format 
               "ARCOPT:v1;VARS:...;OBJ:linear:...;CONSTRS:...;BOUNDS:..."
