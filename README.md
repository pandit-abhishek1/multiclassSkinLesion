# IBR Algorithm Module

This repo now keeps only the algorithm module (no PyPI package setup).

Independent publishable package path: [ibr_pypi_module](ibr_pypi_module)

Implemented in [Algos/ibr.py](Algos/ibr.py):

- `IBR5Net`: exactly 5 convolution stages
- `IBR6Net`: exactly 6 convolution stages
- `DefusedIBR5IBR6`: late fusion over IBR5 and IBR6 features

Both `IBR5Net` and `IBR6Net` return class logits by default.

## Direct Usage

```python
import torch
from Algos.ibr import IBR5Net, IBR6Net, DefusedIBR5IBR6

x = torch.randn(4, 3, 224, 224)

ibr5 = IBR5Net(num_classes=7)
y5 = ibr5(x)  # [4, 7]

ibr6 = IBR6Net(num_classes=7)
y6 = ibr6(x)  # [4, 7]

defused = DefusedIBR5IBR6(num_classes=7)
yf = defused(x)  # [4, 7]
```

## Colab Links

1. [Data Process](https://colab.research.google.com/drive/1vpqEfQ_SgCArX6mtgXwH9M5p8G7CYrrZ?usp=sharing)
2. [Algos](https://colab.research.google.com/drive/10uuzHEatXZwZL0o7qAvEnn3q8KJqY_tE?usp=sharing)
3. [Model Training](https://colab.research.google.com/drive/1N0Jvu7mi2I8vkhAtBfnyRgIcX_9Spxzi?usp=sharing)
