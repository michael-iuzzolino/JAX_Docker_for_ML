import functools
import numpy as onp
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity, MaxPool,
                                   Relu, Dropout, LogSoftmax)


def AlexNet(num_classes):
    
    feature_extraction_stack = stax.serial(
        Conv(64, (11,11), strides=(4,4), padding='VALID'),
        Relu, 
        MaxPool((3,3), strides=(2,2)),
        Conv(192, (5,5), strides=(2,2), padding='VALID'),
        Relu,
        MaxPool((3,3), strides=(2,2)),
        Conv(384, (3,3), strides=(1,1), padding='SAME'),
        Relu,
        Conv(256, (3,3), strides=(1,1), padding='SAME'),
        Relu,
        Conv(256, (3,3), strides=(1,1), padding='SAME'),
        Relu,
        MaxPool((3,3), strides=(2,2)),
    )
    
    classification_stack = (
        Dropout(0.75, mode='train'),
        Dense(4096), Relu,
        Dropout(0.75, mode='train'),
        Dense(4096), Relu,
        Dense(num_classes)
    )
    
    return stax.serial(
        feature_extraction_stack,
        AvgPool((6,6)),
        Flatten,
        classification_stack,
        LogSoftmax
    )
    
