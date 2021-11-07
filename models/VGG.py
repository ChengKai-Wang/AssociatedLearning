import torch
import torch.nn as nn
from .utils import conv_layer_bn

class VGG(nn.Module):
    def __init__(self, num_class):
        super(VGG, self).__init__()
        neurons = 300
        num_class = num_class
        self.activation = nn.ReLU()
        layer_cfg = [128, 256, "M", 256, 512, "M", 512, "M", 512, "M"]
        self.in_channel = 3
        self.creterion = nn.CrossEntropyLoss()
        self.layer1 = self._make_conv_layer(layer_cfg[0:3])
        self.layer2 = self._make_conv_layer(layer_cfg[3:6])
        self.layer3 = self._make_conv_layer(layer_cfg[6:8])
        self.layer4 = self._make_conv_layer(layer_cfg[8:10])

        self.fc1 = self._make_linear_layer(2048, 5*neurons)
        self.fc2 = self._make_linear_layer(5*neurons, neurons)
        self.fc3 = self._make_linear_layer(neurons, neurons)
        self.fc4 = self._make_linear_layer(neurons, neurons)
        self.fc5 = self._make_linear_layer(neurons, num_class)
    
    def _make_conv_layer(self, channel_size: list):
        layers = []
        for dim in channel_size:
            if dim == 'M':
                layers.append(nn.MaxPool2d(2, stride=2))
            else:
                layers.append(conv_layer_bn(self.in_channel, dim, self.activation))
                self.in_channel = dim
        return nn.Sequential(*layers)

    def _make_linear_layer(self, in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features, bias=True), nn.BatchNorm1d(out_features), nn.ReLU())
    
    def forward(self, x, y):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        loss = self.creterion(out, y)
        if self.training:
            return loss
        else:
            return out.detach()