# ibr-defused-algorithms

Independent PyPI module for:

- `IBR5Net` (5-stage from-scratch CNN)
- `IBR6Net` (6-stage from-scratch CNN)
- `DefusedIBR5IBR6` (late fusion of IBR5 + IBR6 features)

Supports both **PyTorch** and **TensorFlow/Keras** backends.

## Install

```bash
pip install ibr-defused-algorithms
```

For TensorFlow/Keras usage, install the optional TensorFlow extra:

```bash
pip install "ibr-defused-algorithms[tensorflow]"
```

## PyTorch Usage

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

## TensorFlow/Keras Usage

```python
import tensorflow as tf
from ibr_defused_algo.tensorflow_models import IBR5Net, IBR6Net, DefusedIBR5IBR6

# TensorFlow models expect NHWC tensors: [batch, height, width, channels]
x = tf.random.normal((4, 224, 224, 3))

ibr5 = IBR5Net(num_classes=7)
ibr6 = IBR6Net(num_classes=7)
defused = DefusedIBR5IBR6(num_classes=7)

# For TensorFlow models
print(ibr5(x).shape)     # [4, 7]
print(ibr6(x).shape)     # [4, 7]
print(defused(x).shape)  # [4, 7]

# Compile and train:
ibr5.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# ibr5.fit(train_ds, epochs=10)
```

### Framework sanity check (recommended in notebooks)

```python
import tensorflow as tf
from ibr_defused_algo.tensorflow_models import IBR5Net

print("Module:", IBR5Net.__module__)  # should be: ibr_defused_algo.tensorflow_models
model = IBR5Net(num_classes=7)
print("Is tf.keras.Model:", isinstance(model, tf.keras.Model))
```

If that last line prints `False`, the runtime is importing the wrong backend. Reinstall and restart kernel.

## Build & Publish

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```
