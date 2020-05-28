import numpy as onp
import torch
from torchvision import datasets, transforms
import tensorflow_datasets as tfds

import modules.utils
import modules.TensorflowDatasets
import modules.PyTorchDatasets

_NORMALIZATIONS = {
    "cifar10"  : ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar100" : ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "celeba"   : ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "mnist"    : ((0.1307,), (0.3081,))
}

def BuildDataHandler(handler_key, download_dir, dataset_name, batch_size, 
                     flatten_img=False, onehot_label=True, image_resize=None):
    assert handler_key in ['tensorflow', 'pytorch'], f'Invalid handler key: {handler_key} -- valid options: tensorflow, pytorch'
    assert dataset_name.lower() in _NORMALIZATIONS.keys(), f"{dataset_name} is not supported."
    
    params = {
        "download_dir" : download_dir, 
        "dataset_name" : dataset_name, 
        "norm_params"  : _NORMALIZATIONS[dataset_name.lower()], 
        "batch_size"   : batch_size, 
        "flatten_img"  : flatten_img, 
        "onehot_label" : onehot_label,
        "image_resize" : image_resize
    }
    
    if 'pytorch' in handler_key.lower():
        data_handler = modules.PyTorchDatasets.Handler(**params)
    elif 'tensorflow' in handler_key.lower():
        data_handler = modules.TensorflowDatasets.Handler(**params)

    return data_handler