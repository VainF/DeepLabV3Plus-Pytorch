# DeepLabv3-plus.pytorch

Pytorch implementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611).

## Datsets
* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Cityscapes](https://www.cityscapes-dataset.com) 

## Results
### VOC2012Aug
| Backbone   | Output Stride     | Overall Acc   | Mean IoU    |  Iters    |
| :--------: | :-------------:   | :-----------: | :--------:  |  :-----:  |
| ResNet50   | 16                |               |             |   20k     |
| ResNet101  | 16                | In Progress   | In Progress |   20k     |


## Quick Start
### 1. Requirements
* Python 3
* Pytorch 1.0.0+
* Torchvision
* Numpy
* Pillow
* tqdm

### 2. Prepare Datasets

Data will be automatically downloaded and extracted. The default path is ./datasets/data. Or you can manually put the downloaded files *(e.g. VOCtrainval_11-May-2012.tar)* under any directory. Please specify your data root with **--data_root PATH/TO/DATA_DIR** if necessary. 

The data dir may be like this:  
```
/DATA_DIR
    /VOCdevkit  
        /SegmentationClass
        /JPEGImages
        ...
    /VOCtrainval_11-May-2012.tar
    ...
```

#### Use VOC2012 trainaug (Optional, Recommended)

See 4 of [2]

         The original dataset contains 1, 464 (train), 1, 449 (val), and 1, 456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10, 582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

*./datasets/data/train_aug.txt* includes names of 10582 trainaug images (val images are excluded). You need to **download extra annatations** from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). **Please extract the SegmentationClassAug files and run scripts with --train_aug_path PATH/TO/SegmentationClassAug**

See [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet) for more information about SegmentationClassAug.

### 3. Train
#### Train on Standard PASCAL VOC2012

```bash
python train.py --backbone resnet50 --dataset voc --data_root ./datasets/data --lr 3e-4 --epochs 60 --batch_size 10 
```

##### Enable Visdom
To enable [visdom]((https://github.com/facebookresearch/visdom)) for visualization, run the following commands separately.
```bash
# Run visdom server on port 13500
visdom -port 13500

# Train
python train.py --backbone resnet50 --dataset voc --data_root ./datasets/data --lr 3e-4 --epochs 60  --batch_size 10 --enable_vis --vis_env deeplab --vis_port 13500
```


## More Details

* run with --crop_val to use cropped image for validation.

* **8G GPU RAM** is required for batch size of 10. If GPU memory is limited, try to reduce batch size or change crop size. Note that batchnorm usually needs large bacth size. As an alternative, you can use [group normalization (GN)](https://arxiv.org/abs/1803.08494).

* Multi-Grid are **not introduced** in this repo according to the paper. see 4.3 of [2].

        Note that we do not employ the multi-grid method [77,78,23], which we found does not improve the performance.

* About Data augmentation. see 4.1 of [1]
  
        Data augmentation: We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0) and randomly left-right flipping during training.


## Train your own model
Please modify line 251 and construct yout own model. Your model should accept a batch of images of Nx3xHxW and return preditions of NxCxHxW logits (no softmax). Here C means number of classes. Remember to modify the datasets transform if input normalization is not needed.

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation]([https://arxiv.org/pdf/1706.05587.pdf](https://arxiv.org/abs/1706.05587))

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)