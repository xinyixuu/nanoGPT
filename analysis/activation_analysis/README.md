# Softmax Comparison

This script evaluates different softmax alternatives for how resistant they are
to softmax collapse (where scaling causes a value to collapse to a single
entry), as well as general distribution shifts (measured by dot product), as
well as overall magnitude changes for the output softmax vectors.

## Usage

```python
python3 softmax_comparison.py 1 1 1 2
```

Then results for softmaxes of 1 1 1 2 and scaled (10 10 10 20) are shown and
compared.

