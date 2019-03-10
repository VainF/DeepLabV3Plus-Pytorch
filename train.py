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

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="learning rate (default: 7e-4)")
    parser.add_argument("--fix_bn", action='store_true', default=False,
                        help='fix batch normalization during training (default: False)')
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='do crop for validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=12,
                        help='batch size (default: 12)')
    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step'], help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=2000,
                        help="decay step for stepLR (default: 2000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0', 
                        help="GPU ID")
    parser.add_argument("--no_nesterov", action='store_true', default=False,
                        help="Enable nesterov (default: False)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--crop_size", type=int, default=513,
                        help="crop size (default: 513)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument("--val_on_trainset", action='store_true', default=False ,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--random_seed", type=int, default=23333,
                        help="random seed (default: 23333)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="saving interval (default: 1)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC' )
    
    # Deeplab Options
    parser.add_argument("--backbone", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet'], help='backbone for deeplab' )

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_sample_num", type=int, default=8,
                        help='number of samples for visualization (default: 6)')
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

        train_dst = Cityscapes(root=opts.data_root, split='train', download=opts.download, transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root, split='val', download=False, transform=val_transform)
    return train_dst, val_dst


def train( cur_epoch, criterion, model, optim, train_loader, device, scheduler=None, print_interval=10, vis=None):
    """Train and return epoch loss"""
    print("Epoch %d, lr = %f"%(cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0
    for cur_step, (images, labels) in enumerate( train_loader ):
        if scheduler is not None:
            scheduler.step()
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # N, C, H, W
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
    
        loss.backward()
        optim.step()

        np_loss = loss.detach().cpu().numpy()
        epoch_loss+=np_loss
        interval_loss+=np_loss

        if (cur_step+1)%print_interval==0:
            interval_loss = interval_loss/print_interval
            print("Epoch %d, Batch %d/%d, Loss=%f"%(cur_epoch, cur_step+1, len(train_loader), interval_loss))
            # visualization
            if vis is not None:
                x = cur_epoch*len(train_loader) + cur_step + 1
                vis.vis_scalar('Loss', x, interval_loss )
            interval_loss=0.0
        
    return epoch_loss / len(train_loader)


def validate( model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    with torch.no_grad():
        for i, (images, labels) in tqdm( enumerate( loader ) ):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids: # get vis samples
                ret_samples.append( (images[0].detach().cpu().numpy(), targets[0], preds[0]) )
        
        score = metrics.get_results()
    return score, ret_samples

def main():
    opts = get_argparser().parse_args()
    opts = modify_command_options(opts)

    # Set up visualization
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None

    if vis is not None: # display options
        vis.vis_table( "Options", vars(opts) )

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print("Device: %s"%device)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1 , shuffle=False, num_workers=opts.num_workers)
    print("Dataset: %s, Train set: %d, Val set: %d"%(opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    print("Backbone: %s"%opts.backbone)
    model = DeepLabv3(num_classes=opts.num_classes, backbone=opts.backbone, pretrained=True, momentum=opts.bn_mom, output_stride=opts.output_stride, use_separable_conv=opts.use_separable_conv)
    if opts.fix_bn==True:
        model.fix_bn()

    if torch.cuda.device_count()>1: # Parallel
        print("%d GPU parallel"%(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model_ref = model.module # for ckpt
    else:
        model_ref = model
    model = model.to(device)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    
    # Set up optimizer
    decay_1x, no_decay_1x = model_ref.group_params_1x()
    decay_10x, no_decay_10x = model_ref.group_params_10x()
    optimizer = torch.optim.SGD(params=[ 
        {"params": decay_1x, 'lr': opts.lr, 'weight_decay':opts.weight_decay},
        {"params": no_decay_1x, 'lr': opts.lr},
        {"params": decay_10x,  'lr': opts.lr*10, 'weight_decay':opts.weight_decay },
        {"params": no_decay_10x,  'lr': opts.lr*10},
    ], lr=opts.lr, momentum=opts.momentum, nesterov=not opts.no_nesterov)
    del decay_1x, no_decay_1x, decay_10x, no_decay_10x
    
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs*len(train_loader), power=opts.lr_power)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    print("Optimizer:\n%s"%(optimizer))
    


    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt)
        model_ref.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]+1
        best_score = checkpoint['best_score']
        print("Model restored from %s"%opts.ckpt)
        del checkpoint # free memory
    else:
        print("[!] Retrain")
    
    def save_ckpt(path):
        """ save current model
        """
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

    # Set up criterion
    criterion = utils.get_loss(opts.loss_type)
    #==========   Train Loop   ==========#
    
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_sample_num, np.int32) if opts.enable_vis else None # sample idxs for visualization
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset)) # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],  
                               std=[0.229, 0.224, 0.225])  # denormalization for ori images
    while cur_epoch < opts.epochs:
        # =====  Train  =====
        model.train()
        if opts.fix_bn==True:
            model_ref.fix_bn()

        epoch_loss = train(cur_epoch=cur_epoch, criterion=criterion, model=model, optim=optimizer, train_loader=train_loader, device=device, scheduler=scheduler, vis=vis)
        print("End of Epoch %d/%d, Average Loss=%f"%(cur_epoch, opts.epochs, epoch_loss))
        if opts.enable_vis:
            vis.vis_scalar("Epoch Loss", cur_epoch, epoch_loss )

        # =====  Save Latest Model  =====
        if (cur_epoch+1)%opts.ckpt_interval==0:
            save_ckpt( 'checkpoints/latest_%s_%s.pkl'%(opts.backbone, opts.dataset) )

        # =====  Validation  =====
        if (cur_epoch+1)%opts.val_interval==0:
            print("validate on val set...")
            model.eval()
            val_score, ret_samples = validate(model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if val_score['Mean IoU']>best_score: # save best model
                best_score = val_score['Mean IoU']
                save_ckpt( 'checkpoints/best_%s_%s.pkl'%(opts.backbone, opts.dataset) )
            
            if vis is not None: # visualize validation score and samples
                vis.vis_scalar("[Val] Overall Acc", cur_epoch, val_score['Overall Acc'] )
                vis.vis_scalar("[Val] Mean IoU", cur_epoch, val_score['Mean IoU'] )
                vis.vis_table("[Val] Class IoU", val_score['Class IoU'] )

                for k, (img, target, lbl) in enumerate( ret_samples ):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = label2color(target).transpose(2,0,1).astype(np.uint8)
                    lbl = label2color(lbl).transpose(2,0,1).astype(np.uint8)

                    concat_img = np.concatenate( (img, target, lbl), axis=2 ) # concat along width
                    vis.vis_image('Sample %d'%k, concat_img)

            if opts.val_on_trainset==True: # validate on train set
                print("validate on train set...")
                train_score, _ = validate(model=model, loader=train_loader, device=device, metrics=metrics)
                print(metrics.to_str(train_score))
                if vis is not None:
                    vis.vis_scalar("[Train] Overall Acc", cur_epoch, train_score['Overall Acc'] )
                    vis.vis_scalar("[Train] Mean IoU", cur_epoch, train_score['Mean IoU'] )
                    
        cur_epoch+=1

if __name__=='__main__':
    main()


    



