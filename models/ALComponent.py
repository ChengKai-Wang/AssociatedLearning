import torch
import torch.nn as nn

class ALComponent(nn.Module):
    """
        Base class of a single associated learning block
        
        f: forward function
        g: autoencoder function
        b: bridge function
        inv: inverse function of autoencoder
        cb: creterion of bridge function
        ca: creterion of autoencoder
    """
    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        b: nn.Module,
        inv: nn.Module,
        cb: nn.Module,
        ca: nn.Module
    )->None:
        super(ALComponent, self).__init__()
        self.f = f
        self.g = g
        self.b = b
        self.inv = inv
        self.cb = cb
        self.ca = ca
    
    def forward(self, x=None, y=None):
        if self.training:
            s = self.f(x)
            s0 = self.b(s)
            t = self.g(y)
            t0 = self.inv(t)
            loss_b = self.cb(s0, t.detach())
            loss_ae = self.ca(t0, y)
            return s.detach(), t.detach(), loss_b, loss_ae
        else:
            if y == None:
                s = self.f(x)
                return s.detach()
            else:
                t0 = self.inv(y)
                return t0.detach()
        
    # for bridge block inference
    def bridge_forward(self, x):
        s = self.f(x)
        s0 = self.b(s)
        t0 = self.inv(s0)
        return t0.detach()