import torch

from ibr_defused_algo import DefusedIBR5IBR6, IBR5Net, IBR6Net


def test_ibr5_shape() -> None:
    x = torch.randn(2, 3, 224, 224)
    y = IBR5Net(num_classes=7)(x)
    assert y.shape == (2, 7)


def test_ibr6_shape() -> None:
    x = torch.randn(2, 3, 224, 224)
    y = IBR6Net(num_classes=7)(x)
    assert y.shape == (2, 7)


def test_defused_shape() -> None:
    x = torch.randn(2, 3, 224, 224)
    y = DefusedIBR5IBR6(num_classes=7)(x)
    assert y.shape == (2, 7)
