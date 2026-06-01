# solve_arcopt_quadratic_problem

## Description

Parse and solve an ARCOPT quadratic optimization problem, then analyze boundary points.

## Usage

```python
from solve_arcopt_quadratic_problem import solve_arcopt_quadratic_problem
result = solve_arcopt_quadratic_problem(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCOPT specification string containing variables, objective, 
                      constraints, and bounds
