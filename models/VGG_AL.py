import torch
import torch.nn as nn
from .utils import conv_layer_bn, Flatten
from .ALComponent import ALComponent


# Single layer of AL block
class VGG_AL(ALComponent):
    def __init__(self, in_size: int, in_channel: int, channel_size: list, hidden_size: int, out_features: int):
        self.size = in_size
        self.features = in_channel
        f_function = nn.ELU()
        g_function = nn.Sigmoid()
        b_function = nn.Sigmoid()
        f = self._make_conv_layer(channel_size, f_function)
        g = nn.Sequential(nn.Linear(out_features, hidden_size), g_function)
        flatten_size = int(self.size * self.size * self.features)
        b = nn.Sequential(Flatten(), nn.Linear(flatten_size, 5*hidden_size), nn.BatchNorm1d(5*hidden_size), b_function, nn.Linear(5*hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), b_function)
        inv = nn.Sequential(nn.Linear(hidden_size, out_features),  nn.BatchNorm1d(out_features), g_function)
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        super(VGG_AL, self).__init__(f, g, b, inv, cb, ca)
    def get_out_size(self):
        return self.size, self.features
    def _make_conv_layer(self, channel_size, activation):
        layers = []
        for dim in channel_size:
            if dim == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
                self.size/=2
            else:
                layers.append(conv_layer_bn(self.features, dim, activation))
                self.features = dim
        return nn.Sequential(*layers)


class VGGALClassifier(nn.Module):
    def __init__(self, num_classes = 10):
        super(VGGALClassifier, self).__init__()
        layers = []
        layer_cfg = {0:[128, 256, "M"], 1:[256, 512, "M"], 2:[512, "M"], 3:[512, "M"]}
        neuron_size = 100
        in_size = 32
        in_channel = 3
        self.num_layers = 4
        self.num_classes = num_classes
        for i in range(self.num_layers):
            if i == 0:
                layer = VGG_AL(in_size, in_channel, layer_cfg[i], neuron_size, num_classes)
                in_size, in_channel = layer.get_out_size()
                layers.append(layer)
            else:
                layer = VGG_AL(in_size, in_channel, layer_cfg[i], neuron_size, neuron_size)
                in_size, in_channel = layer.get_out_size()
                layers.append(layer)
        self.network = nn.ModuleList(layers)
    
    def forward(self, x, y):
        if self.training:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            y_onehot = torch.zeros([len(y), self.num_classes]).to(device)
            for i in range(len(y)):
                #print(y[i])
                y_onehot[i][y[i]] = 1.
            #print(y_onehot)
            _s = x
            _t = y_onehot
            total_loss = {'b':[],'ae':[]}
            for layer in range(self.num_layers):
                _s, _t, loss_b, loss_ae = self.network[layer](_s, _t)
                total_loss['b'].append(loss_b)
                total_loss['ae'].append(loss_ae)
            return total_loss
        else:
            _s = x
            for layer in range(self.num_layers):
                if layer == (self.num_layers - 1):
                    _t0 = self.network[layer].bridge_forward(_s)
                else:
                    _s = self.network[layer](_s, None)
            for layer in range(self.num_layers - 2, -1, -1):
                _t0 = self.network[layer](None, _t0)
            return _t0