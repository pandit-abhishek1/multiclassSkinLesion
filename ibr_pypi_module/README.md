# ibr-defused-algorithms

Independent PyPI module for:

- `IBR5Net` (5-stage from-scratch CNN)
- `IBR6Net` (6-stage from-scratch CNN)
- `DefusedIBR5IBR6` (late fusion of IBR5 + IBR6 features)

This package is fully separated from your personal skin-lesion project logic.

## Install Locally

```bash
pip install -e .
```

## Usage

```python
import torch
from ibr_defused_algo import IBR5Net, IBR6Net, DefusedIBR5IBR6

x = torch.randn(4, 3, 224, 224)

ibr5 = IBR5Net(num_classes=7)
ibr6 = IBR6Net(num_classes=7)
defused = DefusedIBR5IBR6(num_classes=7)

print(ibr5(x).shape)     # [4, 7]
print(ibr6(x).shape)     # [4, 7]
print(defused(x).shape)  # [4, 7]
```

## Publish to PyPI

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```
