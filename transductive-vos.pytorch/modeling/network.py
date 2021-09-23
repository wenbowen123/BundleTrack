import torch.nn as nn

from modeling.backbone.resnet import resnet18, resnet50, resnet101
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d        # additional codes


class VOSNet(nn.Module):

    def __init__(self,
                 model='resnet18', sync_bn=False):

        super(VOSNet, self).__init__()
        self.model = model
        
        # additional codes
        if sync_bn:
            print("Using SynchronizedBatchNorm2d.")
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if model == 'resnet18':
            # resnet = resnet18(pretrained=True)
            resnet = resnet18(pretrained=True, BatchNorm=BatchNorm)      # additional codes
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
        elif model == 'resnet50':
            # resnet = resnet50(pretrained=True)
            resnet = resnet50(pretrained=True, BatchNorm=BatchNorm)    # additional codes
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
            self.adjust_dim = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn256 = nn.BatchNorm2d(256)
        elif model == 'resnet101':
            # resnet = resnet101(pretrained=True)
            resnet = resnet101(pretrained=True, BatchNorm=BatchNorm)    # additional codes
            self.backbone = nn.Sequential(*list(resnet.children())[0:8])
            self.adjust_dim = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn256 = nn.BatchNorm2d(256)
        else:
            raise NotImplementedError

    def forward(self, x):

        if self.model == 'resnet18':
            x = self.backbone(x)
        elif self.model == 'resnet50' or self.model == 'resnet101':
            x = self.backbone(x)
            x = self.adjust_dim(x)
            x = self.bn256(x)

        return x
