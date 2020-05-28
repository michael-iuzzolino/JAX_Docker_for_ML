import numpy as onp
from jax import jit, grad, random
import modules.jax_models

img_format = 'NCHW'

rng = onp.random.RandomState(0)
rng_key = random.PRNGKey(0)
batch_size = 2
pool_dim = 1
input_shape = (batch_size, 3, 32, 32)
images = rng.rand(*input_shape).astype('float32')
print("Images: ", images.shape)

print("Initializing model...")
init, apply = modules.jax_models.ResNet18(num_classes=10, img_fmt=img_format, pool_dim=pool_dim)
_, params = init(rng_key, input_shape)
print("Complete.")

print("Running model")
out = apply(params, images)
print("out.shape: ", out.shape)