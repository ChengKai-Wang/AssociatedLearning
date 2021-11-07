import torch
import torch.nn as nn

# for convolutional neural networks
def conv_layer_bn(in_channels: int, out_channels: int, activation: nn.Module, stride: int=1, bias: bool=False) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, bias = bias, padding = 1)
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(conv, bn, activation)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)