import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds

import modules.utils

class Handler:
    def __init__(self, download_dir, dataset_name, norm_params, batch_size,
                 flatten_img=False, onehot_label=True, image_resize=None):
        if dataset_name.lower() == 'celeba':
            dataset_name = 'celeb_a'
        
        self._download_dir = download_dir
        self._dataset_name = dataset_name.lower()
        self._norm_params = norm_params
        self._batch_size = batch_size
        self._flatten_img = flatten_img
        self._onehot_label = onehot_label
        self._image_resize = image_resize

        self._get_info()
        self.norm_params = onp.array(self._norm_params)
    
    @property
    def num_classes(self):
        if self._dataset_name == 'celeb_a':
            _num_classes = len(list(dict(self._info.features['attributes']).keys()))
        else:
            _num_classes = self._info.features['label'].num_classes
        return _num_classes
    
    @property
    def classes(self):
        if self._dataset_name == 'celeb_a':
            _class_labels = list(dict(self._info.features['attributes']).keys())
        else:
            _class_labels = self._info.features["label"].names
        return _class_labels
    
    @property
    def img_dim(self):
        if self._dataset_name == 'celeb_a':
            img_dim = (3, self._image_resize, self._image_resize)
        else:
            h, w, c = self._info.features['image'].shape
            if self._image_resize:
                img_dim = (c, self._image_resize, self._image_resize)
            else:
                img_dim = (c, h, w)
            
        return img_dim
    
    def num_batches(self, dataset_key):
        R = int(self.size(dataset_key) % self._batch_size > 0)
        return self.size(dataset_key) // self._batch_size + R
        
    def size(self, dataset_key):
        return self._info.splits[dataset_key].num_examples
    
    def _get_info(self):
        print(f"Generating info for {self._dataset_name}...")
        _, self._info = tfds.load(name=self._dataset_name, batch_size=1, data_dir=self._download_dir, with_info=True)
        print("Complete.")
        
    def _normalize(self, imgs):
        expand = lambda x : x[onp.newaxis, :, onp.newaxis, onp.newaxis]
        imgs = imgs / 255.0
        means, stds = self.norm_params
        imgs_normed = (imgs - means) / stds
        return imgs_normed
    
    def _resize_img(self, ele):
        if self._image_resize:
            ele = {
                "image" : tf.image.resize(ele['image'], (self._image_resize, self._image_resize)), 
                **{key : val for key, val in ele.items() if key != "image"}    
            }
        
        return ele
    
    def _transform_img(self, ele):
        ele = {
            "image" : tf.image.random_flip_left_right(ele['image']), 
            **{key : val for key, val in ele.items() if key != "image"}
        }
        
        return ele
        
    def _celebA_iterator(self, dataset_key):
        """ See these discussion for help
            - https://stackoverflow.com/questions/55141076/how-to-apply-data-augmentation-in-tensorflow-2-0-after-tfds-load
            - https://stackoverflow.com/questions/56875027/tensorflow-dataset-image-transform-with-dataset-map
            - https://stackoverflow.com/questions/60187560/how-to-seperate-a-tensorflow-dataset-object-in-features-and-labels
        
        """
        ds = tfds.load(name=self._dataset_name, split=dataset_key, data_dir=self._download_dir)
        ds = ds.map(
                lambda ele: self._resize_img(ele)
            ).cache().map(
                lambda ele: self._transform_img(ele)
            ).batch(
                self._batch_size
            ).prefetch(1)
                
        ds = tfds.as_numpy(ds)
        for d in ds:
            x = d['image']
            y = onp.array(list(d['attributes'].values())).astype(onp.int)
            
            # Normalize
            x = self._normalize(x)

            # Swap to NCHW format
            x = onp.transpose(x, (0,3,1,2))
            if self._flatten_img:
                x = onp.reshape(x, (x.shape[0], -1)).squeeze()

            if self._onehot_label:
                y = modules.utils.label_2_onehot(y, self.num_classes)
            
            y = onp.transpose(y, (1, 0, 2))
            
            yield x, y

    def __call__(self, dataset_key):
        if self._dataset_name == 'celeb_a':
            yield from self._celebA_iterator(dataset_key)
            
        else:
            ds = tfds.load(name=self._dataset_name, split=dataset_key, as_supervised=True, data_dir=self._download_dir)
            ds = ds.batch(self._batch_size).prefetch(1)
            ds = tfds.as_numpy(ds)

            for x, y in ds:
                # Normalize
                x = self._normalize(x)

                # Swap to NCHW format
                x = onp.transpose(x, (0,3,1,2))
                if self._flatten_img:
                    x = onp.reshape(x, (x.shape[0], -1)).squeeze()

                if self._onehot_label:
                    y = modules.utils.label_2_onehot(y, self.num_classes)

                yield x, y