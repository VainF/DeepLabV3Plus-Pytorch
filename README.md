# DeepLabv3-plus.pytorch

Pytorch implementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611).

## Backend
* ResNet50, ResNet101  

## Supported Datsets
* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Cityscapes](https://www.cityscapes-dataset.com) 

## Quick Start
#### 1. Requirements
* Python 3
* Pytorch 1.0.0+
* Torchvision
* Numpy
* Pillow
* tqdm

#### 2. Prepare Datasets

The default path of datasets is *./datasets/data*. The scripts will automatically download and extract all data for you. Also You can manually put the donwloaded files (e.g. VOCtrainval_11-May-2012.tar) under any directory. Just remember to specify your path:
```bash
python train.py --data_root /path/to/your/datasets
```

After extraction, The directory may be like this:  
```
/data
    /VOCdevkit  
        ...
    /VOCtrainval_11-May-2012.tar
    ...
```

#### 3. Train
##### Train without visualization
```bash
python train.py --gpu_id 0 --backbone resnet50 --lr 5e-4 --batch_size 4
```

##### Train with visualization
To enable [visdom]((https://github.com/facebookresearch/visdom)) for visualization, run the following commands separately.
```bash
# Run visdom server on port 13500
visdom -port 13500

# Train
python train.py --gpu_id 0 --backbone resnet50  --lr 5e-4 --batch_size 4 --enable_vis --vis_env deeplab --vis_port 13500
```
Please see train.py for more options.


## Some Details

* In this repo, images are used for validation without any cropping, which is different from 4.1 of [1].

        We thus employ crop size to be 513 during both training and test on PASCAL VOC 2012 dataset.

* The init learning rate is different from original paper. I use 3e-4 for voc when the author uses 7e-3.

* **4G GPU RAM** is required for batch size of 4. If GPU memory is limited, try to reduce batch size or change crop size. Note that batchnorm usually needs large bacth size. As an alternative, you can use [group normalization (GN)](https://arxiv.org/abs/1803.08494).

* Multi-Grid are not introduced in this repo according to the paper. see 4.3 of [2].

        Note that we do not employ the multi-grid method [77,78,23], which we found does not improve the performance.

* Use small momentum for batchnorm of backbone. see part 4 of [1].
  
        In short, we employ the same learning rate schedule (i.e., “poly” policy [52] and same initial learning rate 0.007), crop size 513 × 513, fine-tuning batch normalization parameters [75] when output stride = 16, and random scale data augmentation during training. Note that we also include batch normalization parameters in the proposed decoder module.

* About Data augmentation. see 4.1 of [1]
  
        Data augmentation: We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0) and randomly left-right flipping during training.
## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation]([https://arxiv.org/pdf/1706.05587.pdf](https://arxiv.org/abs/1706.05587))

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)