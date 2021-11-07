import torch
import torch.nn as nn
from .ALComponent import ALComponent
from typing import Tuple

class LinearAL(ALComponent):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: Tuple[int, int],
        bias: bool = False,
    )->None:
        f = nn.Sequential(nn.Linear(in_features, hidden_size[0], bias=bias), nn.ELU())
        g = nn.Sequential(nn.Linear(out_features, hidden_size[1], bias=bias), nn.Sigmoid())
        # bridge function
        b = nn.Sequential(nn.Linear(hidden_size[0], 5*hidden_size[0], bias=bias), nn.Sigmoid(), nn.Linear(5*hidden_size[0], hidden_size[1], bias=bias), nn.Sigmoid())
        # inv function
        inv = nn.Sequential(nn.Linear(hidden_size[1], out_features, bias=bias), nn.Sigmoid())
        # loss function
        cb = nn.MSELoss()
        ca = nn.MSELoss()
        
        super(LinearAL, self).__init__(f, g, b, inv, cb, ca)

class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_shape: int,
        neurons: int,
        out_shape: int,
        bias: bool = False,
        num_layers: int = 3,
    )->None:
        super(MLPClassifier, self).__init__()
        self.X_shape = input_shape
        self.out_shape = out_shape
        self.num_layers = num_layers
        self.mlp_blocks = []
        for layer in range(self.num_layers):
            if layer == 0:
                self.mlp_blocks.append(LinearAL(input_shape, out_shape, (neurons, neurons), bias))
            else:
                self.mlp_blocks.append(LinearAL(neurons, neurons, (neurons, neurons), bias))
        self.mlp_blocks = nn.ModuleList(self.mlp_blocks)
    
    def forward(self, x, y):
        x = x.view(-1, self.X_shape)
        if self.training:
            y_onehot = torch.zeros([len(y), self.out_shape]).to(device)
            for i in range(len(y)):
                y_onehot[i][y[i]] = 1.
            _s = x
            _t = y_onehot
            total_loss = 0
            for layer in range(self.num_layers):
                _s, _t, local_loss = self.mlp_blocks[layer](_s, _t)
                total_loss += local_loss
            return total_loss
        else:
            _s = x
            for layer in range(self.num_layers):
                if layer == (self.num_layers - 1):
                    _t0 = self.mlp_blocks[layer].bridge_forward(_s)
                else:
                    _s = self.mlp_blocks[layer](_s, None)
            for layer in range(self.num_layers - 2, -1, -1):
                _t0 = self.mlp_blocks[layer](None, _t0)
            return _t0

    def short_cut(self, x, n_layer=3):
        x = x.view(-1, self.X_shape)
        _s = x
        for layer in range(n_layer):
            if layer == (n_layer - 1):
                _t0 = self.mlp_blocks[layer].bridge_forward(_s)
            else:
                _s = self.mlp_blocks[layer](_s, None)
        if n_layer != 1:
            for layer in range(n_layer - 2, -1, -1):
                _t0 = self.mlp_blocks[layer](None, _t0)
        return _t0