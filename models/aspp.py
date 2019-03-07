import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AtrousSeparableConvolution

class _ASPP(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, momentum, use_separable_conv=False):
        super(_ASPP, self).__init__()
        
        if use_separable_conv:
            self.atrous_conv_bn_relu = AtrousSeparableConvolution(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False, momentum=momentum)
        else:
            self.atrous_conv_bn_relu = nn.Sequential( nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
                                                      nn.BatchNorm2d(planes, momentum=momentum),
                                                      nn.ReLU(inplace=True))
        self._init_weight()

    def forward(self, x):
        return self.atrous_conv_bn_relu(x)
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, momentum=0.1,use_separable_conv=False):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = _ASPP(inplanes, 256, 1, padding=0, dilation=dilations[0], momentum=momentum, use_separable_conv=use_separable_conv)
        self.aspp2 = _ASPP(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], momentum=momentum, use_separable_conv=use_separable_conv)
        self.aspp3 = _ASPP(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], momentum=momentum, use_separable_conv=use_separable_conv)
        self.aspp4 = _ASPP(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], momentum=momentum, use_separable_conv=use_separable_conv)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256,momentum=momentum),
                                             nn.ReLU(inplace=True))
        
        self.reduce = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(inplace=True)
        )
        
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.reduce(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_aspp(**kargs):
    return ASPP(**kargs)