# DeepLabv3-plus.pytorch

Pytorch implementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611).

## Datsets
* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Cityscapes](https://www.cityscapes-dataset.com) 

## Metrics
Pixel Accuracy and Mean IoU are used as metrics. Note that mIoU and Acc are calculated using confusion matrix.

## Results
### VOC2012 trainaug 

All models are evaluated on cropped images.

| Backbone          | OS (Train/Val)     | Overall Acc   | Mean IoU        |  Fix BN   | Separable Conv  |       
| :--------:        | :-------------:    | :-----------: | :--------:      |  :-----:  | :--------:      |
| ResNet101         | 16/16              |  93.57%       |  75.31%         |    Yes    |     No          |
| ResNet101         | 16/16              |               |                 |    Yes    |     Yes          |
| ResNet101 (Paper) | 16/16              |    -          |  78.85%         |    No     |     Yes         |

## Quick Start
### 1. Requirements
* Pytorch
* Torchvision
* Numpy
* Pillow
* scikit-learn
* tqdm
* matplotlib

### 2. Prepare Datasets

Data will be automatically downloaded and extracted into ./datasets/data with **"--donwload"** options. Or you can change the data root with **--data_root PATH/TO/YOUR/DATA_DIR**. 

The data dir may be like this:  
```
/DATA_DIR
    /VOCdevkit 
        /VOC2012 
            /SegmentationClass
            /JPEGImages
            ...
        ...
    /VOCtrainval_11-May-2012.tar
    ...
```

### 3. Train

#### Train on PASCAL VOC2012 Aug (Recommended)

See chapter 4 of [2]

         The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

*./datasets/data/train_aug.txt* includes names of 10582 trainaug images (val images are excluded). You need to download annatations from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those annotations come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

**Please extract trainaug files (SegmentationClassAug) to the VOC2012 directory.**

```
/DATA_DIR
    /VOCdevkit  
        /VOC2012
            /SegmentationClass
            /SegmentationClassAug
            /JPEGImages
            ...
        ...
    /VOCtrainval_11-May-2012.tar
    ...
```

Then run train.py with *"--year 2012_aug"*
```bash
python train.py --backbone resnet50 --dataset voc --year 2012_aug --data_root ./datasets/data  --lr 3e-4 --epochs 20  --batch_size 12 --use_seperable_conv --fix_bn --enable_vis --vis_env deeplab --vis_port 13500 
```


#### Train on Standard PASCAL VOC2012
```bash
python train.py --backbone resnet101 --dataset voc --year 2012 --data_root ./datasets/data --lr 7e-4 --epochs 30 --batch_size 10
```

##### Enable Visdom
To enable [visdom]((https://github.com/facebookresearch/visdom)) for visualization, run the following commands separately.
```bash
# Run visdom server on port 13500
visdom -port 13500

# Train
python train.py --backbone resnet50 --dataset voc --year 2012 --data_root ./datasets/data --lr 3e-4 --epochs 60  --batch_size 12 --enable_vis --vis_env deeplab --vis_port 13500
```

## More Details

* If GPU memory is limited, try to reduce crop size or batch size. Note that batchnorm needs large bacth size. As an alternative, you can try [group normalization (GN)](https://arxiv.org/abs/1803.08494).

* Multi-Grid are **not introduced** in this repo according to the paper. see 4.3 of [2].

        Note that we do not employ the multi-grid method [77,78,23], which we found does not improve the performance.

* About Data augmentation. see 4.1 of [1]
  
        Data augmentation: We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0) and randomly left-right flipping during training.


## Train your own model
Please modify line 251 and construct yout own model. Your model should accept a batch of images of Nx3xHxW and return preditions of NxCxHxW logits (no softmax). Here C means number of classes. Remember to modify the datasets transform if input normalization is not needed.

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation]([https://arxiv.org/pdf/1706.05587.pdf](https://arxiv.org/abs/1706.05587))

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)