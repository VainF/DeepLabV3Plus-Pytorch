import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from .atrous_resnet import build_resnet
from .aspp import build_aspp
from .decoder import build_decoder

class DeepLabv3(nn.Module):
    """DeepLab v3+
    """
    def __init__(self, num_classes=21, backbone='resnet50', pretrained=True, output_stride=16, momentum=0.1, use_separable_conv=False):
        super(DeepLabv3, self).__init__()
        if 'resnet' in backbone:
            low_level_channels = 256
            features_channels = 2048
            self.backbone = build_resnet(backbone, pretrained=pretrained, output_stride=output_stride, momentum=momentum)
        else:
            raise "[!] Backbone %s not supported yet!"%backbone
        
        self.aspp = build_aspp(inplanes=features_channels, output_stride=output_stride, momentum=momentum, use_separable_conv=use_separable_conv)
        self.decoder =  build_decoder(num_classes=num_classes, low_level_channels=low_level_channels, momentum=momentum, use_separable_conv=use_separable_conv)

    def forward(self, x):
        in_size = x.shape[2:]
        x, low_level_features = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        return F.interpolate(x, size=in_size, mode='bilinear', align_corners=False)

    def group_params_1x(self):
        group_decay = []
        group_no_decay = []

        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    group_decay.append(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    group_no_decay.append(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    if p.requires_grad:
                        group_no_decay.append(p)
        return group_decay, group_no_decay


    def group_params_10x(self):
        group_decay = []
        group_no_decay = []
        for module in [self.aspp, self.decoder]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.weight.requires_grad:
                        group_decay.append(m.weight)
                    if m.bias is not None and m.bias.requires_grad:
                        group_no_decay.append(m.bias)

                elif isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        if p.requires_grad:
                            group_no_decay.append(p)
        return group_decay, group_no_decay