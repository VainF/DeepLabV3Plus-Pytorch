# DeepLabv3-plus.pytorch

DeepLabV3 and DeepLabV3+ for Pytorch.

Supported Backbones: ResNet101, ResNet50, MobileNetV2

## Datsets
* [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

## Results

#### Results on PASCAL VOC2012 Aug (In Progress)

|  Model          | Batch Size  | FLOPs  | Overall Acc   | Mean IoU        | Checkpoint  |
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: | 
| DeepLabV3Plus-MobileNetV2   | 16     |  3.103G      |  0.9259       |  0.7183        |  [Download](https://www.dropbox.com/s/7rox8qvfncz9m2i/best_deeplabv3plus_mobilenet_voc.pth?dl=0) |
| DeepLabV3-MobileNetV2       | -      |  2.187G      |  -       |  -         |    - |
| DeepLabV3Plus-ResNet101     | -      |  25.91G      |    -     |  -         |    - |
| DeepLabV3-ResNet101         | -      |  24.97G      |    -     |  -         |    - |


#### Segmentation Results (DeepLabv3Plus-MobileNet)

<div>
<img src="samples/120_image.png"   width="20%">
<img src="samples/120_target.png"  width="20%">
<img src="samples/120_pred.png"    width="20%">
<img src="samples/120_overlay.png" width="20%">
</div>

<div>
<img src="samples/417_image.png"   width="20%">
<img src="samples/417_target.png"  width="20%">
<img src="samples/417_pred.png"    width="20%">
<img src="samples/417_overlay.png" width="20%">
</div>

<div>
<img src="samples/474_image.png"   width="20%">
<img src="samples/474_target.png"  width="20%">
<img src="samples/474_pred.png"    width="20%">
<img src="samples/474_overlay.png" width="20%">
</div>


#### Vsualization of training

![trainvis](samples/visdom-screenshoot.png)


## Quick Start

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### pascal voc
You can run train.py with "--download" option to download and extract pascal voc dataset. The defaut path is './datasets/data':

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

#### trainaug

See chapter 4 of [2]

        The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10582 (trainaug) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

*./datasets/data/train_aug.txt* includes names of 10582 trainaug images (val images are excluded). You need to download extra annatations from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those annotations come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

**Please extract trainaug files (SegmentationClassAug) to the VOC2012 directory.**

```
/datasets
    /data
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

### 3. Train

#### Visualize training (Optional)

Start visdom sever for visualization. Please remove '--enable_vis' if visualization is not needed. 

```bash
# Run visdom server on port 28333
visdom -port 28333
```

#### Train on PASCAL VOC2012 Aug (Recommended)

Run train.py with *"--year 2012_aug"*

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16
```

#### Train on Standard PASCAL VOC2012

Run train.py with *"--year 2012"*

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012 --crop_val --lr 0.01 --crop_size 513 --batch_size 16
```

### 4. Test

Results and images will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --test_only --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc.pkl --save_test_results
```

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
