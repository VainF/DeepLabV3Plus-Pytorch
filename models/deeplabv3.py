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
    def __init__(self, num_classes=21, backbone='resnet50', pretrained=True, output_stride=16, momentum=3e-4, use_separable_conv=False):
        super(DeepLabv3, self).__init__()
        if 'resnet' in backbone:
            low_level_channels = 256
            features_channels = 2048
            self.backbone = build_resnet(backbone, pretrained=pretrained, output_stride=output_stride, momentum=momentum)
        
        self.aspp = build_aspp(inplanes=features_channels, output_stride=output_stride, momentum=0.1, use_separable_conv=use_separable_conv)
        self.decoder =  build_decoder(num_classes=num_classes, low_level_channels=low_level_channels, momentum=0.1, use_separable_conv=use_separable_conv)

    def forward(self, x):
        in_size = x.shape[2:]
        x, low_level_features = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        return F.interpolate(x, size=in_size, mode='bilinear', align_corners=False)

    def group_params_1x(self):
        for m in self.backbone.modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p
    
    def group_params_10x(self):
        for module in [self.aspp, self.decoder]:
            for m in module.modules():
                for p in m.parameters():
                    if p.requires_grad:
                        yield p
        