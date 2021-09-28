from ._deeplab import convert_to_separable_conv
from .modeling import *

# Set up model
model_map = {
    'deeplabv3_resnet50': deeplabv3_resnet50,
    'deeplabv3plus_resnet50': deeplabv3plus_resnet50,
    'deeplabv3_resnet101': deeplabv3_resnet101,
    'deeplabv3plus_resnet101': deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': deeplabv3plus_mobilenet
}
