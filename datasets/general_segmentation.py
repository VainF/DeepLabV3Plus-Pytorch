import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url


def gen_cmap(N=256, normalized=False):
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    cmap[1:6] = np.array([
        [166, 166, 166],
        [254, 149, 115],
        [90, 123, 255],
        [57, 248, 248],
        [255, 247, 62]
    ])
    cmap = cmap/255 if normalized else cmap
    return cmap

class GenSegmentation(data.Dataset):
    """ GenSegmentation Dataset
    Args:
        root (string): Root directory of the general segmentation Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = gen_cmap()
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        
        self.image_set = image_set
        base_dir = ''
        gen_root = os.path.join(self.root, base_dir)

        image_dir = os.path.join(gen_root, f'{image_set}/images')
        mask_dir = os.path.join(gen_root, f'{image_set}/masks')
        
        self.images = [os.path.join(image_dir, x.split('.')[0]+'.jpg') for x in os.listdir(image_dir)]
        self.masks = [os.path.join(mask_dir, x.split('.')[0]+'.png') for x in os.listdir(image_dir)]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)