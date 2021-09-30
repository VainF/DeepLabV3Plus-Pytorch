import argparse
import os
import pprint
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils import data
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import network
import utils
from datasets import VOCSegmentation, Cityscapes
from datasets.cityscapes import split_data
from losses import semi_loss_zoo
from losses.compose import LossCompose
from losses.mp import MultiCoreKLwithIgnoreIndex
from meters.averagemeter import AverageValueMeter
from metrics import StreamSegMetrics
from utils import ext_transforms as et, save_ckpt, get_lrs_from_optimizer, grouper
from utils.logger import logger
from utils.pretty_print import item2str
from utils.sampler import InfiniteRandomSampler
from utils.visualizer import Visualizer

torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    dataset_parser = parser.add_argument_group("dataset")
    # Dataset Options
    dataset_parser.add_argument("--data_root", type=str, default='./datasets/data',
                                help="path to Dataset")
    dataset_parser.add_argument("--dataset", type=str, default='voc',
                                choices=['voc', 'cityscapes'], help='Name of dataset')

    # PASCAL VOC Options
    dataset_parser.add_argument("--year", type=str, default='2012',
                                choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    model_parser = parser.add_argument_group("model")

    # Deeplab Options
    model_parser.add_argument("--model", type=str, default='deeplabv3_resnet50',
                              choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                       'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                       'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    model_parser.add_argument("--separable_conv", action='store_true', default=False,
                              help="apply separable conv to decoder and aspp")
    model_parser.add_argument("--num_classes", type=int, help="num classes for output classes", required=True)
    model_parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    tra_parser = parser.add_argument_group("train parser")
    tra_parser.add_argument("--test_only", action='store_true', default=False)
    tra_parser.add_argument("--save_val_results", action='store_true', default=False,
                            help="save segmentation results to \"./results\"")
    tra_parser.add_argument("--total_itrs", type=int, default=50e3,
                            help="epoch number")
    tra_parser.add_argument("--lr", type=float, default=0.01,
                            help="learning rate (default: 0.01)")
    tra_parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                            help="learning rate scheduler policy")
    tra_parser.add_argument("--step_size", type=int, default=10000)
    tra_parser.add_argument("--crop_val", action='store_true', default=False,
                            help='crop validation (default: False)')
    tra_parser.add_argument("--batch_size", type=int, default=8,
                            help='batch size (default: 16)')
    tra_parser.add_argument("--val_batch_size", type=int, default=4,
                            help='batch size for validation (default: 4)')
    tra_parser.add_argument("--crop_size", type=int, default=513)

    tra_parser.add_argument("--ckpt", default=None, type=str,
                            help="restore from checkpoint")
    tra_parser.add_argument("--continue_training", action='store_true', default=False)

    tra_parser.add_argument("--gpu_id", type=str, default='0',
                            help="GPU ID")
    tra_parser.add_argument("--weight_decay", type=float, default=1e-4,
                            help='weight decay (default: 1e-4)')
    tra_parser.add_argument("--random_seed", type=int, default=1,
                            help="random seed (default: 1)")
    tra_parser.add_argument("--val_interval", type=int, default=100,
                            help="epoch interval for eval (default: 100)")
    tra_parser.add_argument("--download", action='store_true', default=False,
                            help="download datasets")

    # LOSS part
    loss_parser = parser.add_argument_group("loss")
    loss_parser.add_argument("--loss_type", type=str, default='cross_entropy',
                             choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    loss_parser.add_argument("--semi_loss_type", type=str, nargs="+", default=['null'],
                             choices=['null', 'orth', "mi"], help="semi supervised loss type")
    loss_parser.add_argument("--semi_weights", type=float, nargs="+", default=[0.0, ],
                             help="semi-supervised loss weight")

    visdom_parser = parser.add_argument_group("visdom")

    # Visdom options
    visdom_parser.add_argument("--enable_vis", action='store_true', default=False,
                               help="use visdom for visualization")
    visdom_parser.add_argument("--vis_port", type=str, default='13570',
                               help='port for visdom')
    visdom_parser.add_argument("--vis_env", type=str, default='main',
                               help='env for visdom')
    visdom_parser.add_argument("--vis_num_samples", type=int, default=8,
                               help='number of samples for visualization (default: 8)')

    parser.add_argument("--enable-scale", action="store_true", help="enable amp scale")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    elif opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    else:
        raise NotImplementedError(opts.dataset)
    return train_dst, val_dst


@torch.no_grad()
def validate(opts, model, loader, device, metrics, ret_samples_ids=None, *, auto_cast, group_func=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    de_norm = None
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        de_norm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        img_id = 0
    assert isinstance(loader, DataLoader), type(loader)
    for i, (images, labels) in enumerate(loader):
        with auto_cast:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            if group_func:
                outputs = group_func(outputs)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()

        metrics.update(targets, preds)
        if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
            ret_samples.append(
                (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        if de_norm:
            for i in range(len(images)):
                image = images[i].detach().cpu().numpy()
                target = targets[i]
                pred = preds[i]

                image = (de_norm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = loader.dataset.decode_target(target).astype(np.uint8)
                pred = loader.dataset.decode_target(pred).astype(np.uint8)

                Image.fromarray(image).save('results/%d_image.png' % img_id)
                Image.fromarray(target).save('results/%d_target.png' % img_id)
                Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                plt.imshow(image)
                plt.axis('off')
                plt.imshow(pred, alpha=0.7)
                ax = plt.gca()
                ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                plt.close()
                img_id += 1

    score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_args().parse_args()
    logger.info(pprint.pformat(opts))
    if opts.dataset.lower() == 'voc':
        opts.true_num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.true_num_classes = 19
    else:
        raise NotImplementedError(opts.dataset.lower())

    # Setup visualization
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis \
        else None

    if vis:
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    labeled_set, unlabeled_set = split_data(train_dst, labeled_ratio=0.1)
    del train_dst
    tra_sampler = InfiniteRandomSampler(labeled_set, shuffle=True)
    train_loader = data.DataLoader(
        labeled_set, batch_size=opts.batch_size, sampler=tra_sampler, num_workers=12)
    unlabeled_sampler = InfiniteRandomSampler(unlabeled_set, shuffle=True)

    unlabeled_loader = data.DataLoader(
        unlabeled_set, batch_size=opts.batch_size, sampler=unlabeled_sampler, num_workers=12)

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    logger.info("Dataset: %s, Labeled set: %d, Unlabeled set: %d, Val set: %d" %
                (opts.dataset, len(labeled_set), len(unlabeled_set), len(val_dst)))

    model = network.model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    logger.trace("\n" + str(summary(model, input_size=(1, 3, 1024, 512), verbose=0)))

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes, device="cuda")

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError(opts.lr_policy)

    # Set up criterion
    # if opts.loss_type == 'focal_loss':
    #     criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    if opts.loss_type == 'cross_entropy':
        assert opts.num_classes % opts.true_num_classes == 0
        criterion = MultiCoreKLwithIgnoreIndex(ignore_index=255,
                                               groups=list(grouper(range(
                                                   opts.true_num_classes * (opts.num_classes // opts.true_num_classes)),
                                                   opts.true_num_classes)))
    else:
        raise NotImplementedError(opts.loss_type)

    # set up semi-supervised criterion
    semi_criteria = LossCompose()
    for semi_name, semi_weight in zip(opts.semi_loss_type, opts.semi_weights):
        _criterion = semi_loss_zoo[semi_name](prototypes=model.classifier.classifier[4].weight, )

        semi_criteria.register_loss(_criterion, semi_weight)
    logger.info(f"Semi-supervised losses : {semi_criteria}")

    scaler = GradScaler(enabled=opts.enable_scale)
    auto_cast = autocast(enabled=opts.enable_scale)
    if opts.enable_scale:
        logger.debug("Enable amp scale.")

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_iter = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_iter = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            scaler = checkpoint["scale"]
            logger.trace("Training state restored from %s" % opts.ckpt)
        logger.trace("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        logger.trace("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id,
            auto_cast=auto_cast, group_func=criterion.reduced_simplex)
        logger.info(metrics.to_str(val_score))
        return

    # training part
    # ==============================================================================================================
    model.train()
    cur_epochs += 1
    loss_meter = AverageValueMeter()
    train_loader_iter = tqdm(train_loader, total=int(opts.total_itrs))
    for (images, labels), (unlabeled_images, _) in zip(train_loader_iter, unlabeled_loader):
        cur_iter += 1
        optimizer.zero_grad()

        with auto_cast:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            unlabeled_images = unlabeled_images.to(device, dtype=torch.float)

            outputs_simplex = model(images).softmax(1)
            loss = criterion(outputs_simplex, labels)
            unlabeled_simplex = model(unlabeled_images).softmax(1)
            unlabeled_loss = semi_criteria(unlabeled_simplex)
            total_loss = loss + unlabeled_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_meter.add(loss.item())
        if vis:
            vis.vis_scalar('Loss', cur_iter, loss.item())
        tqdm_post_fix_dict = {"lrs": get_lrs_from_optimizer(optimizer), "loss": loss_meter.summary()}
        train_loader_iter.set_postfix_str(item2str(tqdm_post_fix_dict))
        scheduler.step()

        if cur_iter % opts.val_interval == 0:
            save_ckpt(cur_iter, 'checkpoints/latest_%s_%s_os%d.pth' %
                      (opts.model, opts.dataset, opts.output_stride), model=model, optimizer=optimizer,
                      scheduler=scheduler, best_score=best_score, scaler=scaler)
            logger.info(f"Training iters @ {cur_iter:03d} / {opts.total_itrs}: " + train_loader_iter.postfix)
            logger.info(f"validation @ {cur_iter:03d} / {opts.total_itrs} ")
            model.eval()
            val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                ret_samples_ids=vis_sample_id, auto_cast=auto_cast, group_func=criterion.reduced_simplex)
            logger.info(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                save_ckpt(cur_iter, 'checkpoints/best_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride), model=model, optimizer=optimizer,
                          scheduler=scheduler, best_score=best_score, scaler=scaler)
            if vis:  # visualize validation score and samples
                vis.vis_scalar("[Val] Overall Acc", cur_iter, val_score['Overall Acc'])
                vis.vis_scalar("[Val] Mean IoU", cur_iter, val_score['Mean IoU'])
                vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                for k, (img, target, lbl) in enumerate(ret_samples):
                    img = (denorm(img) * 255).astype(np.uint8)
                    target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                    lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                    concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                    vis.vis_image('Sample %d' % k, concat_img)
            model.train()

        if cur_iter >= opts.total_itrs:
            logger.info(f"Training iters @ {cur_iter:03d} / {opts.total_itrs}: " + train_loader_iter.postfix)
            logger.info(f"validation @ {cur_iter:03d} / {opts.total_itrs} ")
            model.eval()
            val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                ret_samples_ids=vis_sample_id, auto_cast=auto_cast, group_func=criterion.reduced_simplex)
            logger.info(metrics.to_str(val_score))
            logger.info("Training reaches its end.")
            model.train()
            return


if __name__ == '__main__':
    main()
