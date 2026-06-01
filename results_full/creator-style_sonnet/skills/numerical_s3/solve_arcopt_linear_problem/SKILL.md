# solve_arcopt_linear_problem

## Description

Solves an ARCOPT linear programming optimization problem by parsing the specification
and using scipy's linear programming solver.

## Usage

```python
from solve_arcopt_linear_problem import solve_arcopt_linear_problem
result = solve_arcopt_linear_problem(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCOPT specification string containing variables, objective, 
                      constraints, and bounds in the format:
                      ARCOPT:v1;VARS:...;OBJ:linear:...;CONSTRS:...;BOUNDS:...
