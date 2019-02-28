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
from utils.visualizer import Visualizer


def modify_command_options(opts):
    if opts.do_crop==False and opts.dataset=='voc':
        opts.batch_size=1 # for different image size
        opts.fix_bn = True # Fix bn for batch 1
    return opts

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='./datasets/data', 
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset' )
    parser.add_argument("--num_classes", type=int, default=21, 
                        help="num classes (default: 21)")
    parser.add_argument("--random_seed", type=int, default=23333,
                        help="random seed (default: 23333)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch Size (default: 4)')
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')

    parser.add_argument("--epochs", type=int, default=200,
                        help="epoch number (default: 200)")
    parser.add_argument("--fix_bn", action='store_true', default=False ,
                        help="fix batchnorm layer")
    parser.add_argument("--gpu_id", type=str, default='0', 
                        help="GPU ID")
    parser.add_argument("--num_workers", type=int, default=4,
                        help='number of workers (default: 4)')

    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale images (default: 1.0, no scale)")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")

    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--do_crop", action='store_true', default=False,
                        help="do crop for train set. If false, batch size will be set to 1 automatically")
    parser.add_argument("--output_stride", type=int, default=16,
                        help="output stride for deeplabv3+ ")

    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="saving interval (default: 1)")
    
    parser.add_argument("--seed", type=int, default=23333,
                        help="random seed (default: 23333)")
    parser.add_argument("--val_on_trainset", action='store_true', default=False ,
                        help="enable validation on train set (default: False)")
    
    # data set options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012', '2011', '2009', '2008', '2007'], help='year of VOC' )
    # Deeplab options
    parser.add_argument("--backbone", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet'], help='backbone for deeplab' )
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    return parser

def train( cur_epoch, model, optim, train_loader, device, print_interval=10, vis=None):
    print("Epoch %d, lr = %f"%(cur_epoch, optim.param_groups[0]['lr']))
    
    model.train()
    epoch_loss = 0.0
    for cur_step, (images, labels) in enumerate( train_loader ):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # N, C, H, W
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction='mean', ignore_index=255)

        optim.zero_grad()
        loss.backward()
        optim.step()

        np_loss = loss.detach().cpu().numpy()
        epoch_loss+=np_loss

        if (cur_step+1)%print_interval==0:
            print("Epoch %d, Batch %d/%d, Loss=%f"%(cur_epoch, cur_step+1, len(train_loader), np_loss))
        
        # visualization
        if vis is not None:
            x = cur_epoch*len(train_loader) + cur_step + 1
            vis.vis_scalar('Loss', x, np_loss )
    
    return epoch_loss / len(train_loader)

def validate( model, loader, device, metrics, ret_samples_ids=None):
    model.eval()
    metrics.reset()
    ret_samples = []
    with torch.no_grad():
        for i, (images, labels) in tqdm( enumerate( loader ) ):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            #print(np.unique(preds), np.unique(targets))
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids: # get vis samples
                ret_samples.append( (images[0].detach().cpu().numpy(), targets[0], preds[0]) )
        
        score = metrics.get_results()
    return score, ret_samples

def get_dataset(opts):
    if opts.dataset=='voc':
        train_transform = ExtCompose( [ 
            #ExtResize(size=512),
            ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            ExtRandomHorizontalFlip(),
            ExtToTensor(),
            ExtNormalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] ),
        ])

        # batch size 1
        val_transform = ExtCompose([
            ExtToTensor(),
            ExtNormalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] ),
        ])
    
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='train', download=True, transform=train_transform)
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

        # batch size 1
        val_transform = ExtCompose( [
            ExtToTensor(),
            ExtNormalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] ),
        ] )

        train_dst = Cityscapes(root=opts.data_root, split='train', download=True, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, split='val', download=False, transform=val_transform)
    return train_dst, val_dst

def main():
    opts = get_argparser().parse_args()
    opts = modify_command_options(opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    #print( torch.cuda.is_available() )
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    print("Device: %s"%device)
    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up visualization
    if opts.enable_vis==False:
        vis=None
    else:
        vis = Visualizer(port=opts.vis_port, env=opts.vis_env)

    # Set up dataloader
    train_dst, val_dst = get_dataset(opts)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_loader = data.DataLoader(val_dst, batch_size=1 if opts.dataset=='voc' else opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    print("Dataset: %s, Train set: %d, Val set: %d"%(opts.dataset, len(train_dst), len(val_dst)))
    
    # Set up model
    model = DeepLabv3(num_classes=opts.num_classes, backbone=opts.backbone, pretrained=True)
    if torch.cuda.device_count()>1: # Parallel
        print("%d GPU parallel"%(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model_ref = model.module # for ckpt
    else:
        model_ref = model
    model = model.to(device)

    # fix bn if needed
    if opts.fix_bn==True:
        print("[!] Fix BN")
        utils.fix_bn(model)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[ 
        {"params": model.group_params_1x(),  'lr': opts.lr },
        {"params": model.group_params_10x(), 'lr': opts.lr*10 },
    ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    print("Optimizer:\n%s"%(optimizer))
    
    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(args.ckpt)
        model_ref.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]
        best_score = checkpoint['best_score']
        print("Model restored from %s"%opts.ckpt)
    else:
        print("[!] Retrain")
    
    def save_ckpt(path):
        state = {
                    "epoch": cur_epoch,
                    "model_state": model_ref.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score,
        }
        #path = 'checkpoints/deeplab_%s_%s_epoch%d.pkl'%(opts.backbone, opts.dataset, cur_epoch)
        torch.save(state, path)
        print( "Model saved as %s"%path )

    #
    #==========   Train Loop   ========#
    #

    vis_sample_id = np.random.randint(0, len(train_loader), 4, np.int32) if opts.enable_vis else None # samples for visualization
    label2color = utils.Label2Color(cmap=utils.color_map()) # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],  
                               std=[0.229, 0.224, 0.225])  # denormalization

    while cur_epoch < opts.epochs:
        # Train Steps
        epoch_loss = train(cur_epoch=cur_epoch, model=model, optim=optimizer, train_loader=train_loader, device=device, vis=vis)
        print("End of Epoch %d, Average Loss=%f"%(cur_epoch, epoch_loss))

        # save
        if (cur_epoch+1)%opts.ckpt_interval==0:
            save_ckpt( 'latest_%s_%s.pkl'%(opts.backbone, opts.dataset) )

        # validation
        if (cur_epoch+1)%opts.val_interval==0:
            print("validate on val set...")
            val_score, ret_samples = validate(model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))

            if val_score['Mean IoU']>best_score: # save best model
                best_score = val_score['Mean IoU']
                save_ckpt( 'best_%s_%s.pkl'%(opts.backbone, opts.dataset) )

            if vis is not None: # visualize score and samples
                vis.vis_scalar("[Val] Overall Acc", cur_epoch, val_score['Overall Acc'] )
                vis.vis_scalar("[Val] Mean IoU", cur_epoch, val_score['Mean IoU'] )
                for k, (img, target, lbl) in enumerate( ret_samples ):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = label2color(target).transpose(2,0,1).astype(np.uint8)
                    lbl = label2color(lbl).transpose(2,0,1).astype(np.uint8)

                    concat_img = np.concatenate( (img, target, lbl), axis=2 ) # concat along width
                    vis.vis_image('Sample %d'%k, concat_img)

            if opts.val_on_trainset==True: # do validation on train set
                print("validate on train set...")
                train_score, _ = validate(model=model, loader=train_loader, device=device, metrics=metrics)
                print(metrics.to_str(train_score))
                if vis is not None:
                    vis.vis_scalar("[Train] Overall Acc", cur_epoch, train_score['Overall Acc'] )
                    vis.vis_scalar("[Train] Mean IoU", cur_epoch, train_score['Mean IoU'] )
        cur_epoch+=1

if __name__=='__main__':
    main()


    



