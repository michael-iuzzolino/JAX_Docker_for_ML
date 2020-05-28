import numpy as onp
import torch
from torchvision import datasets, transforms

import modules.utils

class FlattenImg:
    def __init__(self, active):
        self.active = active
        
    def __call__(self, x):
        if self.active:
            x = onp.reshape(x, (x.shape[0], -1)).squeeze()
        return x
            
class Handler:
    def __init__(self, download_dir, dataset_name, norm_params, batch_size, 
                 flatten_img=False, onehot_label=True, image_resize=None):
        if dataset_name.lower() == 'celeba':
            dataset_name = 'CelebA'
        
        self._download_dir = download_dir
        self._dataset_name = dataset_name
        self._norm_params = norm_params
        self._batch_size = batch_size
        self._flatten_img = flatten_img
        self._onehot_label = onehot_label
        self._image_resize = image_resize
        
        data_xform = self._setup_transform()
        self.loaders = self._setup_dataset(download_dir, data_xform)
 
    @property
    def num_classes(self):
        if self._dataset_name == 'CelebA':
            _num_classes = len(self.datasets['train'].attr_names)
        else:
            _num_classes = len(self.datasets['train'].classes)
        return _num_classes
    
    @property
    def classes(self):
        if self._dataset_name == 'CelebA':
            _class_labels = self.datasets['train'].attr_names
        else:
            _class_labels = self.datasets['train'].classes
        return _class_labels
    
    @property
    def img_dim(self):
        if self._dataset_name == 'CelebA':
            img_dim = (3, self._image_resize, self._image_resize)
        else:
            img_dim = self.datasets['train'].data.shape[1:]
            c = 1 if len(img_dim) == 2 else img_dim[-1]
            if self._image_resize:
                img_dim = (c, self._image_resize, self._image_resize)
            else:
                img_dim = self.datasets['train'].data.shape[1:]
                if len(img_dim) == 2:
                    img_dim = (1, *img_dim)
                else:
                    img_dim = img_dim[::-1]
            
        return img_dim
    
    def num_batches(self, dataset_key):
        R = int(self.size(dataset_key) % self._batch_size > 0)
        return self.size(dataset_key) // self._batch_size + R
        
    def size(self, dataset_key):
        return self.datasets[dataset_key].data.shape[0]
        
    def _setup_transform(self):
        xform_list = []
        if self._dataset_name == 'CelebA':
            xform_list += [
                transforms.Resize(self._image_resize),
                transforms.CenterCrop(self._image_resize)
            ]
        
        xform_list += [
            transforms.ToTensor(),
            transforms.Normalize(*self._norm_params),
            FlattenImg(self._flatten_img)
        ]
        
        data_xform = transforms.Compose(xform_list)
        return data_xform

    def _setup_dataset(self, download_dir, data_xform):
        dataset_obj = datasets.__dict__[self._dataset_name]
        if self._dataset_name == 'CelebA':
            train_dataset = dataset_obj(download_dir, split='train', download=True, transform=data_xform)
            test_data = dataset_obj(download_dir, split='test', transform=data_xform)
        else:
            train_dataset = dataset_obj(download_dir, train=True, download=True, transform=data_xform)
            test_data = dataset_obj(download_dir, train=False, transform=data_xform)
        self.datasets = {
            "train" : train_dataset,
            "test"  : test_data
        }
        
        
        loaders = { key : torch.utils.data.DataLoader(dataset, 
                                                      batch_size=self._batch_size, 
                                                      shuffle=key=='train')
                       for key, dataset in self.datasets.items() }
        return loaders
            
    def __call__(self, dataset_key):
        for x, y in self.loaders[dataset_key]:
            x = x.numpy()
            y = y.numpy()
            if self._onehot_label:
                y = modules.utils.label_2_onehot(y, self.num_classes)
            yield x, y