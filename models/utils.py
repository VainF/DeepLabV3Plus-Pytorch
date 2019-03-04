import torch
import torch.nn as nn
import torch.nn.functional as F

class AtrousSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True, groups=1):
        super(AtrousSeparableConvolution, self).__init__()
        self.separable_conv = nn.Conv2d( in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels )
        self.pointwise_conv = nn.Conv2d( out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.pointwise_conv( self.separable_conv(x) )