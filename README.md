# DeepLabv3-plus.pytorch

Pytorch implementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611).

## Datsets
* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Cityscapes](https://www.cityscapes-dataset.com) 

## Results

### VOC2012 trainaug (In Pregress)
| Backbone   | OS (Train/Val)    | Overall Acc   | Mean IoU    |  Iters    |
| :--------: | :-------------:   | :-----------: | :--------:  |  :-----:  |
| ResNet50   | 16/8              |               |             |   20k     |
| ResNet101  | 16/8              |               |             |   20k     |


## Quick Start
### 1. Requirements
* Pytorch 1.0
* Torchvision
* Numpy
* Pillow
* scikit-learn
* tqdm

### 2. Prepare Datasets

Data will be automatically downloaded and extracted. The default path is ./datasets/data. Or you can manually put the downloaded files *(e.g. VOCtrainval_11-May-2012.tar)* under any directory. Please specify your data root with **--data_root PATH/TO/DATA_DIR** if necessary. 

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

#### Train on Standard PASCAL VOC2012
You can run the scripts directly without any preparing. training will start after downloading and extraction.

```bash
python train.py --backbone resnet50 --dataset voc --year 2012 --data_root ./datasets/data --lr 3e-4 --epochs 60 --batch_size 10 
```

##### Enable Visdom
To enable [visdom]((https://github.com/facebookresearch/visdom)) for visualization, run the following commands separately.
```bash
# Run visdom server on port 13500
visdom -port 13500

# Train
python train.py --backbone resnet50 --dataset voc --year 2012 --data_root ./datasets/data --lr 3e-4 --epochs 60  --batch_size 12 --enable_vis --vis_env deeplab --vis_port 13500
```


#### Train on PASCAL VOC2012 Aug (Recommended)

See chapter 4 of [2]

         The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

*./datasets/data/train_aug.txt* includes names of 10582 trainaug images (val images are excluded). You need to **download extra annatations** from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those annotations come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

**Please extract the SegmentationClassAug files to directory of VOC2012. And the directory should be like this:**

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
python train.py --backbone resnet50 --dataset voc --year 2012_aug --data_root ./datasets/data  --lr 3e-4 --epochs 20  --batch_size 12 --enable_vis --vis_env deeplab --vis_port 13500
```


## More Details

* run with --crop_val to use cropped image for validation.

* **9GB GPU Memory** is required for batch size of 12 and ResNet50. If GPU memory is limited, try to reduce crop size or batch size. Note that batchnorm needs large bacth size. As an alternative, you can try [group normalization (GN)](https://arxiv.org/abs/1803.08494).

* Multi-Grid are **not introduced** in this repo according to the paper. see 4.3 of [2].

        Note that we do not employ the multi-grid method [77,78,23], which we found does not improve the performance.

* About Data augmentation. see 4.1 of [1]
  
        Data augmentation: We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0) and randomly left-right flipping during training.


## Train your own model
Please modify line 251 and construct yout own model. Your model should accept a batch of images of Nx3xHxW and return preditions of NxCxHxW logits (no softmax). Here C means number of classes. Remember to modify the datasets transform if input normalization is not needed.

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation]([https://arxiv.org/pdf/1706.05587.pdf](https://arxiv.org/abs/1706.05587))

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)