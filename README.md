# DeepLabv3-plus.pytorch

Unofficial Pytorch implementation of [DeepLabv3+](https://arxiv.org/abs/1802.02611).

## Backend
* ResNet50, ResNet101  


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
python train.py --gpu_id 0 --lr 5e-4 --batch_size 6 
```

##### Train with visualization
To enable [visdom]((https://github.com/facebookresearch/visdom)) for visualization, run the following commands separately.
```bash
# Run visdom server on port 13500
visdom -port 13500

# Train
python train.py --gpu_id 0  --lr 5e-4 --batch_size 6 --enable_vis --vis_env deeplab --vis_port 13500
```
Please see train.py for more options.

