from models import DeepLabv3
import utils
from tqdm import tqdm
import argparse
import os

import numpy as np
import random
from torch.utils import data

from datasets import VOCSegmentation, Cityscapes
from utils.ext_transforms import *
from metrics import StreamSegMetrics

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.visualizer import Visualizer



def modify_command_options(opts):
    if opts.dataset=='voc':
        opts.num_classes = 21
    elif opts.dataset=='cityscapes':
        opts.num_classes = 20
    return opts

def get_argparser():
    parser = argparse.ArgumentParser()


    parser.add_argument("--save_path", type=str, default=None, 
                        help="Path to save results (default: None)")
    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data', 
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset' )
    parser.add_argument("--num_classes", type=int, default=None, 
                        help="num classes (default: None)")
    
    # Model Options
    parser.add_argument("--bn_mom", type=float, default=3e-4,
                        help='momentum for batchnorm of backbone  (default: 3e-4)')
    parser.add_argument("--output_stride", type=int, default=16,
                        help="output stride for deeplabv3+")
    parser.add_argument("--use_separable_conv", action='store_true', default=False,
                        help="Use separable conv in ASPP and Decoder")
    parser.add_argument("--use_gn", action='store_true', default=False,
                        help='use group normalization')
                        
    # Train Options
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='do crop for validation (default: False)')
    parser.add_argument("--download", action='store_true', default=False,
                        help='download datasets (default: False)')
    parser.add_argument("--batch_size", type=int, default=12,
                        help='batch size (default: 12)')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--gpu_id", type=str, default='0', 
                        help="GPU ID")
    parser.add_argument("--crop_size", type=int, default=513,
                        help="crop size (default: 513)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument("--random_seed", type=int, default=23333,
                        help="random seed (default: 23333)")
    
    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC' )
    
    # Deeplab Options
    parser.add_argument("--backbone", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet'], help='backbone for deeplab' )
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset=='voc':
        train_transform = ExtCompose( [ 
            ExtRandomScale((0.5, 2.0)),
            ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] ),
        ])

        if opts.crop_val:
            val_transform = ExtCompose([
                ExtResize(size=opts.crop_size),
                ExtCenterCrop(size=opts.crop_size),
                ExtToTensor(),
                ExtNormalize( mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] ),
            ])
        else:
            # no crop, batch size = 1
            val_transform = ExtCompose([
                ExtToTensor(),
                ExtNormalize( mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] ),
            ])
    
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='val', download=False, transform=val_transform)
        
    if opts.dataset=='cityscapes':
        train_transform = ExtCompose( [ 
            ExtScale(0.5),
            ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] ),
        ] )

        val_transform = ExtCompose( [
            ExtScale(0.5),
            ExtToTensor(),
            ExtNormalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] ),
        ] )

        train_dst = Cityscapes(root=opts.data_root, split='train', download=opts.download, target_type='semantic',  transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root, split='test', target_type='semantic', download=False, transform=val_transform)
    return train_dst, val_dst


def main():
    opts = get_argparser().parse_args()
    opts = modify_command_options(opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print("Device: %s"%device)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up dataloader
    _, val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1 , shuffle=False, num_workers=opts.num_workers)
    print("Dataset: %s, Val set: %d"%(opts.dataset, len(val_dst)))
    
    # Set up model
    print("Backbone: %s"%opts.backbone)
    model = DeepLabv3(num_classes=opts.num_classes, backbone=opts.backbone, pretrained=True, momentum=opts.bn_mom, output_stride=opts.output_stride, use_separable_conv=opts.use_separable_conv)
    if opts.use_gn==True:
        print("[!] Replace BatchNorm with GroupNorm!")
        model = utils.convert_bn2gn(model)

    if torch.cuda.device_count()>1: # Parallel
        print("%d GPU parallel"%(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model_ref = model.module # for ckpt
    else:
        model_ref = model
    model = model.to(device)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.save_path is not None:
        utils.mkdir(opts.save_path)

    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt)
        model_ref.load_state_dict(checkpoint["model_state"])
        print("Model restored from %s"%opts.ckpt)
    else:
        print("[!] Retrain")
    
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset)) # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],  
                               std=[0.229, 0.224, 0.225])  # denormalization for ori images
    model.eval()
    metrics.reset()
    idx = 0

    if opts.save_path is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        

    with torch.no_grad():
        for i, (images, labels) in tqdm( enumerate( val_loader ) ):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
            if opts.save_path is not None:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1,2,0).astype(np.uint8)
                    target = label2color(target).astype(np.uint8)
                    pred = label2color(pred).astype(np.uint8)

                    Image.fromarray(image).save(os.path.join(opts.save_path, '%d_image.png'%idx) )
                    Image.fromarray(target).save(os.path.join(opts.save_path, '%d_target.png'%idx) )
                    Image.fromarray(pred).save(os.path.join(opts.save_path, '%d_pred.png'%idx) )
                    
                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(os.path.join(opts.save_path, '%d_overlay.png'%idx), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    idx+=1
                
    score = metrics.get_results()
    print(metrics.to_str(score))
    if opts.save_path is not None:
        with open(os.path.join(opts.save_path, 'score.txt'), mode='w') as f:
            f.write(metrics.to_str(score))

if __name__=='__main__':
    main()


    



