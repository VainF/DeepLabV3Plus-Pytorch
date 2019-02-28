# DeepLabv3-plus.pytorch

Pytorch implementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611).

## Backend
:white_check_mark: ResNet50, ResNet101  

:x: Xception (In Progress)

## Supported Datsets
* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Cityscapes](https://www.cityscapes-dataset.com) 

## Quick Start
#### 1. Requirements
* Python 3
* Pytorch 0.4.0+
* Torchvision
* Numpy
* Pillow
* tqdm

#### 2. Prepare Datasets

The default dataset path is *./datasets/data*. The scripts will automatically prepare all data. Also You can manually put those donwloaded files (e.g. tar file) under any directory. Just remember to specify your path:

```bash
python train.py --data_root /path/to/your/datasets
```

The directory may be like this:  
```
/data
    /VOCdevkit  
        ...
    /VOCtrainval_11-May-2012.tar
    ...
```

#### 3. run scripts
```bash
python train.py --lr 7e-4 --backbone resnet50 --gpu_id 0 --dataset voc --do_crop
```
If you want to use visdom for visualization, run the following commands separately:
```bash
visdom -port 13500

python train.py --lr 7e-4  --backbone resnet50 --gpu_id 0  --dataset voc --do_crop --enable_vis --vis_env main --vis_port 13500 
```
visit [visdom (github repo)](https://github.com/facebookresearch/visdom) for more information.

