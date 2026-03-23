"""TensorFlow/Keras implementations of IBR algorithms."""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConvStage(layers.Layer):
    """Single convolution stage for IBR architectures."""

    def __init__(self, out_channels: int, stride: int, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.stride = stride

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.out_channels, 3, strides=self.stride, padding="same", use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(
            self.out_channels, 3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        return x


class IBRBase(keras.Model):
    """Base class for IBR variants."""

    def __init__(
        self,
        channels: list,
        strides: list,
        num_classes: int = 7,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.strides = strides
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.stages = []
        for out_ch, stride in zip(channels, strides):
            self.stages.append(ConvStage(out_ch, stride, name=f"conv_stage_{len(self.stages)}"))

        self.pool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(num_classes)

    def call(self, x, training=False):
        if x.shape.rank != 4:
            raise ValueError(
                f"IBR TensorFlow models expect rank-4 inputs [B, H, W, C]; got rank {x.shape.rank}."
            )

        # Fail fast for a common mistake when users reuse PyTorch NCHW pipelines.
        if x.shape[-1] not in (1, 3) and x.shape[1] in (1, 3):
            raise ValueError(
                "IBR TensorFlow models expect NHWC inputs [B, H, W, C]. "
                "Received a tensor that looks NCHW; remove channel transpose and keep channels last."
            )

        for stage in self.stages:
            x = stage(x, training=training)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.classifier(x)
        return x

    def extract_features(self, x, training=False):
        for stage in self.stages:
            x = stage(x, training=training)
        x = self.pool(x)
        return x


class IBR5Net(IBRBase):
    """TensorFlow/Keras IBR5 with exactly 5 stages."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.2, **kwargs):
        super().__init__(
            channels=[32, 64, 128, 256, 512],
            strides=[2, 2, 2, 2, 2],
            num_classes=num_classes,
            dropout=dropout,
            **kwargs
        )


class IBR6Net(IBRBase):
    """TensorFlow/Keras IBR6 with exactly 6 stages."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.2, **kwargs):
        super().__init__(
            channels=[32, 64, 128, 256, 384, 512],
            strides=[2, 2, 2, 2, 1, 1],
            num_classes=num_classes,
            dropout=dropout,
            **kwargs
        )


class DefusedIBR5IBR6(keras.Model):
    """TensorFlow/Keras late fusion of IBR5 and IBR6 branches."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.ibr5 = IBR5Net(num_classes=num_classes, dropout=dropout, name="ibr5_branch")
        self.ibr6 = IBR6Net(num_classes=num_classes, dropout=dropout, name="ibr6_branch")

        self.fusion_dense1 = layers.Dense(512, activation="relu", name="fusion_dense1")
        self.fusion_dropout = layers.Dropout(dropout)
        self.fusion_dense2 = layers.Dense(num_classes, name="fusion_dense2")

    def call(self, x, training=False):
        f5 = self.ibr5.extract_features(x, training=training)
        f6 = self.ibr6.extract_features(x, training=training)
        fused = layers.concatenate([f5, f6])
        fused = self.fusion_dense1(fused)
        fused = self.fusion_dropout(fused, training=training)
        out = self.fusion_dense2(fused)
        return out


FusedIBR5IBR6 = DefusedIBR5IBR6
