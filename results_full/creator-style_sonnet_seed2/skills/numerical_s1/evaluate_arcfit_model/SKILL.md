# evaluate_arcfit_model

## Description

Evaluate an ARCFIT fitted model on query points.

## Usage

```python
from evaluate_arcfit_model import evaluate_arcfit_model
result = evaluate_arcfit_model(<spec_string>)
print(result)
```

## Inputs
spec_string (str): ARCFIT specification in format 
                      "FITTED:model_type;PARAMS:param=value,...;QUERY:x1,x2,..."
