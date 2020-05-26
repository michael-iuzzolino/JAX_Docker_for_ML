import numpy as onp
import torch
from torchvision import datasets, transforms
import tensorflow_datasets as tfds

import modules.utils

_NORMALIZATIONS = {
    "cifar10" : ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "mnist"   : ((0.1307,), (0.3081,))
}

class FlattenImg:
    def __init__(self, active):
        self.active = active
        
    def __call__(self, x):
        if self.active:
            x = onp.reshape(x, (x.shape[0], -1)).squeeze()
        return x
            
class PyTorchDataHandler:
    def __init__(self, download_dir, dataset_name, batch_size, flatten_img=False, onehot_label=True):
        assert dataset_name.lower() in ['mnist', 'cifar10'], f"{dataset_name} is not supported."
        self.flatten_img = flatten_img
        self.onehot_label = onehot_label
        self.batch_size = batch_size
        
        data_xform = self._setup_transform(dataset_name)
        self.loaders = self._setup_dataset(download_dir, data_xform, dataset_name, batch_size)
 
    @property
    def num_classes(self):
        return len(self.datasets['train'].classes)
    
    @property
    def classes(self):
        return self.datasets['train'].classes
    
    @property
    def img_dim(self):
        img_dim = self.datasets['train'].data.shape[1:]
        if len(img_dim) == 2:
            img_dim = [1, *img_dim]
        return img_dim[::-1]
    
    def num_batches(self, dataset_key):
        R = int(self.size(dataset_key) % self.batch_size > 0)
        return self.size(dataset_key) // self.batch_size + R
        
    def size(self, dataset_key):
        return self.datasets[dataset_key].data.shape[0]
        
    def _setup_transform(self, dataset_name):
        data_xform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*_NORMALIZATIONS[dataset_name.lower()]),
            FlattenImg(self.flatten_img)
        ])
        return data_xform

    def _setup_dataset(self, download_dir, data_xform, dataset_name, batch_size):
        dataset_obj = datasets.__dict__[dataset_name]
        
        self.datasets = {
            "train" : dataset_obj(download_dir, train=True, download=True, transform=data_xform),
            "test"  : dataset_obj(download_dir, train=False, transform=data_xform)
        }
        
        
        loaders = { key : torch.utils.data.DataLoader(dataset, 
                                                      batch_size=batch_size, 
                                                      shuffle=key=='train')
                       for key, dataset in self.datasets.items() }
        return loaders
            
    def __call__(self, dataset_key):
        for x, y in self.loaders[dataset_key]:
            x = x.numpy()
            y = y.numpy()
            if self.onehot_label:
                y = modules.utils.label_2_onehot(y, self.num_classes)
            yield x, y
    
class TensorflowDataHandler:
    def __init__(self, download_dir, dataset_name, batch_size, flatten_img=False, onehot_label=True):
        assert dataset_name.lower() in ['mnist', 'cifar10'], f"{dataset_name} is not supported."
        self.download_dir = download_dir
        self.dataset_name = dataset_name.lower()
        self.flatten_img = flatten_img
        self.batch_size = batch_size
        self.onehot_label = onehot_label

        self._get_info()
        self.norm_params = onp.array(_NORMALIZATIONS[dataset_name.lower()])
    
    @property
    def num_classes(self):
        return self._info.features['label'].num_classes
    
    @property
    def classes(self):
        return self._info.features["label"].names
    
    @property
    def img_dim(self):
        h, w, c = self._info.features['image'].shape
        return (c, h, w)
    
    def num_batches(self, dataset_key):
        R = int(self.size(dataset_key) % self.batch_size > 0)
        return self.size(dataset_key) // self.batch_size + R
        
    def size(self, dataset_key):
        return self._info.splits[dataset_key].num_examples
    
    def _get_info(self):
        _, self._info = tfds.load(name=self.dataset_name, batch_size=-1, data_dir=self.download_dir, with_info=True)
    
    def _normalize(self, imgs):
        expand = lambda x : x[onp.newaxis, :, onp.newaxis, onp.newaxis]
        imgs = imgs / 255.0
        means, stds = self.norm_params
        imgs_normed = (imgs - means) / stds
        return imgs_normed
    
    def __call__(self, dataset_key):
        ds = tfds.load(name=self.dataset_name, split=dataset_key, as_supervised=True, data_dir=self.download_dir)
        ds = ds.batch(self.batch_size).prefetch(1)
        ds = tfds.as_numpy(ds)
        
        for x, y in ds:
            # Normalize
            x = self._normalize(x)
            
            # Swap to NCHW format
            x = onp.transpose(x, (0,3,1,2))
            if self.flatten_img:
                x = onp.reshape(x, (x.shape[0], -1)).squeeze()
               
            if self.onehot_label:
                y = modules.utils.label_2_onehot(y, self.num_classes)
                
            yield x, y
            
def BuildDataHandler(handler_key, download_dir, dataset_name, batch_size, flatten_img=False, onehot_label=True):
    if 'pytorch' in handler_key.lower():
        data_handler = PyTorchDataHandler(download_dir, dataset_name, batch_size, flatten_img, onehot_label)
    elif 'tensorflow' in handler_key.lower():
        data_handler = TensorflowDataHandler(download_dir, dataset_name, batch_size, flatten_img, onehot_label)
    else:
        assert False, f'Invalid handler key: {handler_key} -- valid options: tensorflow, pytorch'

    return data_handler