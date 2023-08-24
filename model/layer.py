import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, dilation=1, type='spa_sconv'):
    if type == 'spa_sconv':
        return SPA_SConv(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, bias=False)
    elif type == 'conv2d':
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation, bias=False)
    else:
        assert False


def get_padding_sph_map(x, m):
    n = x.size(2)

    x11 = x[:,:,0:m,0:n//2].contiguous()
    x12 = x[:,:,0:m,n//2:n].contiguous()
    x13 = x[:,:,(n-m):n,0:n//2].contiguous()
    x14 = x[:,:,(n-m):n,n//2:n].contiguous()

    if m > 1:
        x11 = x11.flip(2)
        x12 = x12.flip(2)
        x13 = x13.flip(2)
        x14 = x14.flip(2)

    x = torch.cat([
        torch.cat([x12, x11], dim=3),
        x,
        torch.cat([x14, x13], dim=3),
    ], dim=2)

    x21 = x[:,:,:,0:m].contiguous()
    x22 = x[:,:,:,(n-m):n].contiguous()
    x = torch.cat([x22, x, x21], dim=3)
    return x


class SPA_SConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SPA_SConv, self).__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, 0, dilation=dilation, bias=bias)

    def forward(self, x):
        x = get_padding_sph_map(x, self.padding)
        x1 = self.conv(x)
        x2 = self.conv(x.flip(2)).flip(2)
        x = torch.stack([x1,x2]).max(0)[0]
        return x


class SPA_SMaxPool(nn.Module):
    def __init__(self, kernel_size=3, stride=2):
        super(SPA_SMaxPool, self).__init__()
        self.padding = kernel_size // 2
        self.pool = nn.MaxPool2d(kernel_size, stride, 0)

    def forward(self, x):
        x = get_padding_sph_map(x, self.padding)
        x = self.pool(x)
        return x


from spherical_utils import sphconv_op, sph_harm_all, DHaj, sph_sample

class SPE_SConv(nn.Module):
    def __init__(self, n, c_in, c_out, real=True, nonlinear='prelu', use_bias=True):
        super(SPE_SConv, self).__init__()

        weight = Parameter(torch.zeros(c_in, 1, n//2, 1, c_out))
        self.register_parameter('weight', weight)
        std = 2./(2 * math.pi * np.sqrt((n // 2) * (c_out)))
        self._init_weight(self.weight, 0, std)

        if use_bias:
            self.register_parameter('bias', Parameter(torch.zeros([1,1,1,c_out])))
        else:
            self.register_parameter('bias', None)

        if nonlinear is None:
            self.nonlinear = None
        elif nonlinear.lower() == 'prelu':
            self.nonlinear = nn.PReLU()
        elif nonlinear.lower() == 'relu':
            self.nonlinear = nn.ReLU()
        else:
            assert False

        harmonics = sph_harm_all(n, as_tfvar=True, real=real)
        aj = DHaj(n)
        self.register_buffer('harmonics', harmonics)
        self.register_buffer('aj', aj)

    def forward(self, x):
        x = x.permute(0,3,2,1).contiguous()
        x = sphconv_op(x, self.weight, self.harmonics, self.aj)
        if self.bias is not None:
            x = x + self.bias
        if self.nonlinear is not None:
            x = self.nonlinear(x)
        x = x.permute(0,3,2,1).contiguous()
        return x

    def _init_weight(self,tensor,mean=0,std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor


class SphWeightedAvgPool(nn.Module):
    def __init__(self, kernel=2, stride=2):
        super(SphWeightedAvgPool, self).__init__()
        self.pool = nn.AvgPool2d(kernel, stride)

    def forward(self, x):
        x = x.permute(0,3,2,1).contiguous()
        x = self.area_weights(x)
        x = self.pool(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.invert_area_weights(x)
        x = x.permute(0,3,2,1).contiguous()
        return x

    def area_weights(self, x):
        n = x.size(1)
        phi, _ = sph_sample(n)
        phi += np.diff(phi)[0]/2
        phi = torch.FloatTensor(np.sin(phi)).to(x.device)
        x = x * phi.reshape(1,1,n,1)
        return x

    def invert_area_weights(self, x):
        n = x.size(1)
        phi, _ = sph_sample(n)
        phi += np.diff(phi)[0]/2
        phi = torch.FloatTensor(np.sin(phi)).to(x.device)
        x = x / phi.reshape(1,1,n,1)

        return x