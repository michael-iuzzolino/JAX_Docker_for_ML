import functools
import numpy as onp
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity, MaxPool,
                                   Relu, LogSoftmax)


def MLP(layers, num_classes):
    return stax.serial(
        *[stax.serial(Dense(layer), Relu) for layer in layers],
        Dense(num_classes),
        LogSoftmax
    )
