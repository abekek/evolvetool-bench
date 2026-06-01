# evaluate_arcfit_exponential_decay

## Description

Evaluates an ARCFIT exponential decay model at specified x values with robust error handling.

## Usage

```python
from evaluate_arcfit_exponential_decay import evaluate_arcfit_exponential_decay
result = evaluate_arcfit_exponential_decay(<spec_string>)
print(result)
```

## Inputs
spec_string (str): Specification in format "FITTED:model_type;PARAMS:param_assignments;QUERY:x_values"
                      Example: "FITTED:exp_decay;PARAMS:a=1.0,b=1.0,c=0.0;QUERY:0.0,-1.0,100.0"
