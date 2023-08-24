from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
from logging import raiseExceptions
from tkinter import YES

import torch
import torch.nn as nn
from torch.autograd import Function
import spherical._ext as _ext

import numpy as np


class SphericalMap(Function):
    @staticmethod
    def forward(ctx, res, phi, rho, dis, feat):
        smap = _ext.sphericalmap(phi, rho, dis, feat, res)
        ctx.mark_non_differentiable(smap)
        return smap

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

spherical_map = SphericalMap.apply


class Feat2Smap(nn.Module):
    def __init__(self, res=64, return_dis_smap=True):
        super(Feat2Smap, self).__init__()
        self.res = res
        self.return_dis_smap = return_dis_smap

    def forward(self, pts, feat=None):
        '''
        pts: B*N*3
        feat: B*N*C

        dis_smap: B*res*res*1
        feat_smap: B*res*res*C
        '''

        r = torch.norm(pts, dim=2)
        if feat is None:
            feat = r.unsqueeze(2)
        else:
            if self.return_dis_smap:
                feat = torch.cat([r.unsqueeze(2), feat], dim=2)

        x = pts[:,:,0]
        y = pts[:,:,1]
        z = pts[:,:,2]

        t = np.pi / float(self.res)
        k = 2*np.pi / float(self.res)

        phi = torch.round(torch.acos(torch.clamp(z / r, -1, 1)) / t).int() % self.res
        rho = torch.atan2(y, x)
        rho = torch.where(y>=0, torch.round(rho / k), rho)
        rho = torch.where(y<0, torch.round((rho+2*np.pi)/k), rho)
        rho = rho.int() % self.res

        smap = spherical_map(self.res, phi, rho, r, feat)

        if self.return_dis_smap:
            dis_smap = smap[:,:,:,0].unsqueeze(3).contiguous()
            feat_smap = smap[:,:,:,1:].contiguous()
            if len(feat_smap.size())==3:
                feat_smap = feat_smap.unsqueeze(3)
            return dis_smap.permute(0,3,2,1).contiguous(), feat_smap.permute(0,3,2,1).contiguous()
        else:
            feat_smap = smap
            return feat_smap.permute(0,3,2,1).contiguous()


if __name__ == "__main__":
    func = Feat2Smap(64)

    pts = torch.rand(2, 2000, 3).cuda().float() - 0.5
    pts[:,:,2] = pts[:,:,2]+2
    feat = torch.rand(2, 2000, 3).cuda().float()
    # feat = torch.norm(pts, dim=2, keepdims=True)

    dis, rgb = func(pts, feat)
    print(dis[0])
    print(rgb[0])
