import torch
from torch import nn
import torch.nn.functional as F
import os

__all__ = ['HRNet', 'hrnetv2_48', 'hrnetv2_32']

# Checkpoint path of pre-trained backbone (edit to your path). Download backbone pretrained model hrnetv2-32 @
# https://drive.google.com/file/d/1NxCK7Zgn5PmeS7W1jYLt5J9E0RRZ2oyF/view?usp=sharing .Personally, I added the backbone
# weights to the folder /checkpoints
try:
    CKPT_PATH = './checkpoints/hrnetv2_32_model_best_epoch96.pth'
    print(f"Backbone HRNet Pretrained weights at: {CKPT_PATH}, only usable for HRNetv2-32")
except:
    print("No backbone checkpoint found for HRNetv2, please set pretrained=False when calling model")

# HRNetv2-48 not available yet, but you can train the whole model from scratch.

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()

        self.number_of_branches = stage  # number of branches is equivalent to the stage configuration.
        self.output_branches = output_branches

        self.branches = nn.ModuleList()

        # Note: Resolution + Number of channels maintains the same throughout respective branch.
        for i in range(self.number_of_branches):  # Stage scales with the number of branches. Ex: Stage 2 -> 2 branch
            channels = c * (2 ** i)  # Scale channels by 2x for branch with lower resolution,

            # Paper does x4 basic block for each forward sequence in each branch (x4 basic block considered as a block)
            branch = nn.Sequential(*[BasicBlock(channels, channels) for _ in range(4)])

            self.branches.append(branch)  # list containing all forward sequence of individual branches.

        # For each branch requires repeated fusion with all other branches after passing through x4 basic blocks.
        self.fuse_layers = nn.ModuleList()

        for branch_output_number in range(self.output_branches):

            self.fuse_layers.append(nn.ModuleList())

            for branch_number in range(self.number_of_branches):
                if branch_number == branch_output_number:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif branch_number > branch_output_number:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** branch_number), c * (2 ** branch_output_number), kernel_size=1, stride=1,
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** branch_output_number), eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (branch_number - branch_output_number)), mode='nearest'),
                    ))
                elif branch_number < branch_output_number:
                    downsampling_fusion = []
                    for _ in range(branch_output_number - branch_number - 1):
                        downsampling_fusion.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** branch_number), c * (2 ** branch_number), kernel_size=3, stride=2,
                                      padding=1,
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** branch_number), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    downsampling_fusion.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** branch_number), c * (2 ** branch_output_number), kernel_size=3,
                                  stride=2, padding=1,
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** branch_output_number), eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*downsampling_fusion))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # input to each stage is a list of inputs for each branch
        x = [branch(branch_input) for branch, branch_input in zip(self.branches, x)]

        x_fused = []
        for branch_output_index in range(
                self.output_branches):  # Amount of output branches == total length of fusion layers
            for input_index in range(self.number_of_branches):  # The inputs of other branches to be fused.
                if input_index == 0:
                    x_fused.append(self.fuse_layers[branch_output_index][input_index](x[input_index]))
                else:
                    x_fused[branch_output_index] = x_fused[branch_output_index] + self.fuse_layers[branch_output_index][
                        input_index](x[input_index])

        # After fusing all streams together, you will need to pass the fused layers
        for i in range(self.output_branches):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused  # returning a list of fused outputs


class HRNet(nn.Module):
    def __init__(self, c=48, num_blocks=[1, 4, 3], num_classes=1000):
        super(HRNet, self).__init__()

        # Stem:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1:
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, affine=True, track_running_stats=True),
        )
        # Note that bottleneck module will expand the output channels according to the output channels*block.expansion
        bn_expansion = Bottleneck.expansion  # The channel expansion is set in the bottleneck class.
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),  # Input is 64 for first module connection
            Bottleneck(bn_expansion * 64, 64),
            Bottleneck(bn_expansion * 64, 64),
            Bottleneck(bn_expansion * 64, 64),
        )

        # Transition 1 - Creation of the first two branches (one full and one half resolution)
        # Need to transition into high resolution stream and mid resolution stream
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c, eps=1e-05, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 2, eps=1e-05, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2:
        number_blocks_stage2 = num_blocks[0]
        self.stage2 = nn.Sequential(
            *[StageModule(stage=2, output_branches=2, c=c) for _ in range(number_blocks_stage2)])

        # Transition 2  - Creation of the third branch (1/4 resolution)
        self.transition2 = self._make_transition_layers(c, transition_number=2)

        # Stage 3:
        number_blocks_stage3 = num_blocks[1]  # number blocks you want to create before fusion
        self.stage3 = nn.Sequential(
            *[StageModule(stage=3, output_branches=3, c=c) for _ in range(number_blocks_stage3)])

        # Transition  - Creation of the fourth branch (1/8 resolution)
        self.transition3 = self._make_transition_layers(c, transition_number=3)

        # Stage 4:
        number_blocks_stage4 = num_blocks[2]  # number blocks you want to create before fusion
        self.stage4 = nn.Sequential(
            *[StageModule(stage=4, output_branches=4, c=c) for _ in range(number_blocks_stage4)])

        # Classifier (extra module if want to use for classification):
        # pool, reduce dimensionality, flatten, connect to linear layer for classification:
        out_channels = sum([c * 2 ** i for i in range(len(num_blocks)+1)])  # total output channels of HRNetV2
        pool_feature_map = 8
        self.bn_classifier = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4, eps=1e-05, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(pool_feature_map),
            nn.Flatten(),
            nn.Linear(pool_feature_map * pool_feature_map * (out_channels // 4), num_classes),
        )

    @staticmethod
    def _make_transition_layers(c, transition_number):
        return nn.Sequential(
            nn.Conv2d(c * (2 ** (transition_number - 1)), c * (2 ** transition_number), kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(c * (2 ** transition_number), eps=1e-05, affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Stem:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # split to 2 branches, form a list.

        # Stage 2
        x = self.stage2(x)
        x.append(self.transition2(x[-1]))

        # Stage 3
        x = self.stage3(x)
        x.append(self.transition3(x[-1]))

        # Stage 4
        x = self.stage4(x)

        # HRNetV2 Example: (follow paper, upsample via bilinear interpolation and to highest resolution size)
        output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
        x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)

        # Upsampling all the other resolution streams and then concatenate all (rather than adding/fusing like HRNetV1)
        x = torch.cat([x[0], x1, x2, x3], dim=1)
        x = self.bn_classifier(x)
        return x


def _hrnet(arch, channels, num_blocks, pretrained, progress, **kwargs):
    model = HRNet(channels, num_blocks, **kwargs)
    if pretrained:
        print("Loading pretrained backbone HRNetV2 model .....")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
    return model


def hrnetv2_48(pretrained=False, progress=True, number_blocks=[1, 4, 3], **kwargs):
    w_channels = 48
    return _hrnet('hrnetv2_48', w_channels, number_blocks, pretrained, progress,
                  **kwargs)


def hrnetv2_32(pretrained=False, progress=True, number_blocks=[1, 4, 3], **kwargs):
    w_channels = 32
    return _hrnet('hrnetv2_32', w_channels, number_blocks, pretrained, progress,
                  **kwargs)


if __name__ == '__main__':

    try:
        CKPT_PATH = os.path.join(os.path.abspath("."), '../../checkpoints/hrnetv2_32_model_best_epoch96.pth')
        print("--- Running file as MAIN ---")
        print(f"Backbone HRNET Pretrained weights as __main__ at: {CKPT_PATH}")
    except:
        print("No backbone checkpoint found for HRNetv2, please set pretrained=False when calling model")

    # Models
    model = hrnetv2_32(pretrained=True)
    #model = hrnetv2_48(pretrained=False)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    in_ = torch.ones(1, 3, 768, 768).to(device)
    y = model(in_)
    print(y.shape)

    # Calculate total number of parameters:
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)






