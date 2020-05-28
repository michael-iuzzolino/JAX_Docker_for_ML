import functools
import numpy as onp
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity, MaxPool,
                                   Relu, LogSoftmax)

def BasicBlock(planes, strides=(1,1)):
    _expansion = 1
    
    residual = stax.serial(
        Conv(planes, (3,3), strides, padding='SAME'), 
        BatchNorm(), Relu,
        Conv(planes, (3,3), strides=(1,1), padding='SAME'), 
        BatchNorm()
    )
    
    identity = stax.serial(
        Conv(planes, (1, 1), strides), BatchNorm()
    )
    
    out = stax.serial(
        FanOut(2), # Splits input X into two streams; i.e., FanOut(X) -> (X, X)
        stax.parallel(residual, identity),
        FanInSum,
        Relu
    )
    
    return out

def Bottleneck(planes, strides=(1,1)):
    _expansion = 4
    
    residual = stax.serial(
        Conv(planes, (1,1), strides=(1,1), padding='SAME'), 
        BatchNorm(), Relu,
        Conv(planes, (3,3), strides=strides, padding='SAME'), 
        BatchNorm(), Relu,
        Conv(planes*_expansion, (1,1), strides=(1,1), padding='SAME'), 
        BatchNorm()
    )
    
    identity = stax.serial(
        Conv(planes*_expansion, (1, 1), strides), BatchNorm()
    )
    
    out = stax.serial(
        FanOut(2), # Splits input X into two streams; i.e., FanOut(X) -> (X, X)
        stax.parallel(residual, identity),
        FanInSum, # Rejoins streams
        Relu
    )
    
    return out

def make_layer(block_op, planes, blocks, strides=(1,1)):
    init_layer = block_op(planes, strides)
    layers = [init_layer, *[ block_op(planes, strides=(1,1)) for _ in range(1, blocks) ]]
    return stax.serial(*layers)

def compute_pool_dim(img_dim, n_downsteps=5):
    return tuple(onp.array(img_dim) // (2**n_downsteps))
    
def ResNet(block_op, blocks, planes, num_classes, img_dim=(224,224),
           img_fmt="NCHW", kernel_fmt="OIHW", output_fmt="NHWC"):
    strides = [(1,1)] + [(2,2) for _ in range(len(blocks)-1)]
    
    global_pool_spatial_dim = compute_pool_dim(img_dim)
    
    # First block
    _format = (img_fmt, kernel_fmt, output_fmt)
    first_layer = stax.serial(
        GeneralConv(_format, 64, (7, 7), strides=(2,2), padding='SAME'),
        BatchNorm(), Relu,
        MaxPool((3, 3), strides=(2, 2), padding='SAME')
    )
    
    # Res Blocks
    n_blocks = len(blocks)
    res_block_layers = stax.serial(
        *[make_layer(block_op, planes[i], blocks[i], strides[i]) 
              for i in range(n_blocks)]
    )
    
    # Final Layer
    final_layer = stax.serial(
        AvgPool(global_pool_spatial_dim),
        Flatten, 
        Dense(num_classes), 
        LogSoftmax
    )
    
    return stax.serial(
        first_layer,
        res_block_layers,
        final_layer
    )

def ResNet18(*args, **kwargs):
    block_op = BasicBlock
    blocks = [2, 2, 2, 2]
    planes = [64, 128, 256, 512]
    return functools.partial(ResNet, block_op, blocks, planes)(*args, **kwargs)

def ResNet34(*args, **kwargs):
    block_op = BasicBlock
    blocks = [3, 4, 6, 3]
    planes = [64, 128, 256, 512]
    return functools.partial(ResNet, block_op, blocks, planes)(*args, **kwargs)

def ResNet50(*args, **kwargs):
    block_op = Bottleneck
    blocks = [3, 4, 6, 3]
    planes = [64, 128, 256, 512]
    return functools.partial(ResNet, block_op, blocks, planes)(*args, **kwargs)

def ResNet101(*args, **kwargs):
    block_op = Bottleneck
    blocks = [3, 4, 23, 3]
    planes = [64, 128, 256, 512]
    return functools.partial(ResNet, block_op, blocks, planes)(*args, **kwargs)

def ResNet152(*args, **kwargs):
    block_op = Bottleneck
    blocks = [3, 8, 36, 3]
    planes = [64, 128, 256, 512]
    return functools.partial(ResNet, block_op, blocks, planes)(*args, **kwargs)