import torch
import torch.nn as nn
from copy import deepcopy

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, score:bool=False, non_linearity=nn.ReLU(), num_classes: int=10):
        super(VGG, self).__init__()
        self.non_linearity=non_linearity
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.score = score

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if not self.score:
            out = self.softmax(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           deepcopy(self.non_linearity)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def medium_cnn(checkpoint_path: str = "", num_classes: int=10, score: bool=False, non_linearity=nn.ReLU(), vgg_name='VGG11') -> nn.Module:
    net = VGG(vgg_name=vgg_name, non_linearity=non_linearity, score=score, num_classes=num_classes)
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    return net