import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import AtrousSeparableConvolution
class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_channels=2048, momentum=0.1, use_separable_conv=False):
        super(Decoder, self).__init__()

        self.reduce_low_level = nn.Sequential( 
                nn.Conv2d(low_level_channels, 48, 1, bias=False),
                nn.BatchNorm2d(48, momentum=momentum),
                nn.ReLU(inplace=True),
        )
        if use_separable_conv:
            self.last_conv = nn.Sequential( AtrousSeparableConvolution(304, 256, kernel_size=3, stride=1, padding=1, bias=False, dilation=1, momentum=momentum ),
                                            AtrousSeparableConvolution(256, 256, kernel_size=3, stride=1, padding=1, bias=False, dilation=1, momentum=momentum ),
                                            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        else:
            self.last_conv = nn.Sequential( nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256, momentum=momentum),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256, momentum=momentum),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()
    

    def forward(self, x, low_level_features):
        low_level_features = self.reduce_low_level(low_level_features)
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(**kargs):
    return Decoder(**kargs)