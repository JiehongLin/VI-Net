import torch
import torch.nn as nn

import math
import numpy as np

from layer import conv3x3, SPA_SMaxPool
from rotation_utils import Ortho6d2Mat
from pointnet2_utils import three_nn, three_interpolate
from pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule


class PointNet2MSG(nn.Module):
    def __init__(self, radii_list, dim_in=6, use_xyz=True):
        super(PointNet2MSG, self).__init__()
        self.SA_modules = nn.ModuleList()
        c_in = dim_in
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=radii_list[0],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 16, 16, 32]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_0 = 32 + 32

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=radii_list[1],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_1 = 64 + 64

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=radii_list[2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_2 = 128 + 128

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=radii_list[3],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 128, 256], [c_in, 128, 128, 256]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_3 = 256 + 256

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + dim_in, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=True))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        _, N, _ = pointcloud.size()

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, type='spa_sconv'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation, type=type)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation, type=type)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, type='spa_sconv'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=1, type=type)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 dim_in=1,
                 layers=(3, 4, 23, 3),
                 type='spa_sconv'
                 ):

        self.current_stride = 4
        self.current_dilation = 1
        self.output_stride = 32

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(dim_in, 64, stride=1, type=type)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if type == 'spa_sconv':
            self.maxpool = SPA_SMaxPool(kernel_size=3, stride=2)
        else:
            self.maxpool = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(block, 64, layers[0], type=type) # 32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, type=type) # 16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=2, type=type) # 8
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1, dilation=4, type=type)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, type='spa_sconv'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation, type=type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation, type=type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x5


class FPN(nn.Module):
    def __init__(self, dim_in=[64,128,256], out_dim=256, mode='nearest', align_corners=True, type='spa_sconv', ds_rate=2):
        super(FPN, self).__init__()
        self.ds_rate = ds_rate
        self.conv1 = conv3x3(dim_in[0], out_dim, stride=1, type=type)
        self.bn1 = nn.BatchNorm2d(out_dim)

        self.conv2 = conv3x3(dim_in[1], out_dim, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv3 = conv3x3(dim_in[2], out_dim, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.relu = nn.ReLU(inplace=True)

        if mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x1, x2, x3):

        x3 = self.up(x3)
        x2 = self.bn2(self.conv2(x2))
        x2 = self.relu(x2+x3)

        if self.ds_rate == 4:
            return x2
        else:
            x2 = self.up(x2)
            x1 = self.bn1(self.conv1(x1))
            x1 = self.relu(x1+x2)

            x1 = self.relu(self.bn3(self.conv3(x1)))

            if self.ds_rate == 1:
                x1 = self.up(x1)

            return x1
        

class SphericalFPN(nn.Module):
    def __init__(self, dim_in1=1, dim_in2=3, type='spa_sconv', ds_rate=2):
        super(SphericalFPN, self).__init__()
        self.ds_rate = ds_rate
        assert ds_rate in [1,2,4]
        self.encoder1 = ResNet(BasicBlock, dim_in1, [2, 2, 2, 2], type=type)
        self.encoder2 = ResNet(BasicBlock, dim_in2, [2, 2, 2, 2], type=type)
        self.FPN = FPN(dim_in=[128,256,256], mode='bilinear', type=type, ds_rate=ds_rate)

        if ds_rate in [1,2]:
            self.conv1 = conv3x3(64*2, 128, stride=1, type=type)
            self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128*2, 256, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = conv3x3(256*2, 256, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        y11,y21,y31 = self.encoder1(x1)
        y12,y22,y32 = self.encoder2(x2)
        if self.ds_rate in [1,2]:
            y1 = self.relu(self.bn1(self.conv1(torch.cat([y11,y12],1))))
        else:
            y1 = None
        y2 = self.relu(self.bn2(self.conv2(torch.cat([y21,y22],1))))
        y3 = self.relu(self.bn3(self.conv3(torch.cat([y31,y32],1))))
        y = self.FPN(y1,y2,y3)
        return y


class V_Branch(nn.Module):
    def __init__(self, in_dim=256, ncls=1, resolution=32):
        super(V_Branch, self).__init__()
        self.ncls = ncls
        self.res = resolution
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.rho_classifier = nn.Sequential(
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, ncls, 1),
        )

        self.phi_classifier = nn.Sequential(
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, ncls, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, l, cls=None):
        emb = self.mlp(x)

        rho_emb = torch.max(emb, dim=2)[0]
        rho_prob = self.rho_classifier(rho_emb)

        phi_emb = torch.max(emb, dim=3)[0]
        phi_prob = self.phi_classifier(phi_emb)

        if self.ncls > 1:
            b,_,n = rho_prob.size()
            index = cls.reshape(b,1,1).expand(b,1,n)

            rho_prob = torch.gather(rho_prob, 1, index)
            phi_prob = torch.gather(phi_prob, 1, index)

        rho_prob = rho_prob.squeeze(1)
        phi_prob = phi_prob.squeeze(1)
        vp_rot = self._get_vp_rotation(rho_prob, phi_prob, l).detach()
        return vp_rot, rho_prob, phi_prob
    
    def _get_vp_rotation(self, rho_prob, phi_prob, l):
        b = rho_prob.size(0)
        n = self.res
        assert n == rho_prob.size(1)
        assert n == phi_prob.size(1)

        if self.training and 'rho_label' in l.keys():
            rho_label = l['rho_label'].reshape(b).long()
            rho_noise = (torch.rand(b)>0.5).long()*torch.randint(-3,3,(b,)).long()
            rho_label = rho_label + rho_noise.to(rho_label.device)
            rho_label = torch.clamp(rho_label, 0, n-1)

        else:
            rho_label = torch.sigmoid(rho_prob)
            rho_label = torch.max(rho_label, dim=1)[1]

        
        if self.training and 'phi_label' in l.keys():
            phi_label = l['phi_label'].reshape(b).long()
            phi_noise = (torch.rand(b)>0.5).long()*torch.randint(-3,3,(b,)).long()
            phi_label = phi_label + phi_noise.to(phi_label.device)
            phi_label = torch.clamp(phi_label, 0, n-1)
        else:
            phi_label = torch.sigmoid(phi_prob)
            phi_label = torch.max(phi_label, dim=1)[1]


        rho_label = rho_label + 0.5
        phi_label = phi_label + 0.5

        init_rho = rho_label.reshape(b).float() * (2*np.pi/float(n))
        init_phi = phi_label.reshape(b).float() * (np.pi/float(n))

        zero = torch.zeros(b,1,1).to(init_rho.device)
        one = torch.ones(b,1,1).to(init_rho.device)

        init_rho = init_rho.reshape(b,1,1)
        m1 = torch.cat([
            torch.cat([torch.cos(init_rho), -torch.sin(init_rho), zero],dim=2),
            torch.cat([torch.sin(init_rho), torch.cos(init_rho), zero],dim=2),
            torch.cat([zero, zero, one],dim=2),
        ],dim=1)

        init_phi = init_phi.reshape(b,1,1)
        m2 = torch.cat([
            torch.cat([torch.cos(init_phi), zero, torch.sin(init_phi)],dim=2),
            torch.cat([zero, one, zero],dim=2),
            torch.cat([-torch.sin(init_phi), zero, torch.cos(init_phi)],dim=2),
        ],dim=1)

        return m1@m2



class I_Branch(nn.Module):
    def __init__(self, in_dim=256, ncls=1, resolution=32):
        super(I_Branch, self).__init__()
        self.ncls = ncls
        self.res = resolution

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, self.res//8, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6*self.ncls),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, vp_rot, cls=None):
        emb = self._get_transformed_feat(x, vp_rot)
        emb = self.conv(emb).squeeze(3).squeeze(2)
        r6d = self.mlp(emb)

        if self.ncls > 1:
            b = r6d.size(0)
            index = cls.reshape(b,1,1).expand(b,6,1)
            r6d = r6d.reshape(b,6,self.ncls)
            r6d = torch.gather(r6d, 2, index).squeeze(2)
        r = Ortho6d2Mat(r6d[:,0:3], r6d[:,3:6])
        return r

    def _get_transformed_feat(self, x, vp_rot):
        b,c,n,_ = x.size()
        assert n == self.res

        grid = torch.arange(n).float().to(x.device) + 0.5
        grid_rho = grid * (2*np.pi/float(n))
        grid_rho = grid_rho.reshape(1,n).repeat(n,1)
        grid_phi = grid * (np.pi/float(n))
        grid_phi = grid_phi.reshape(n,1).repeat(1,n)

        sph_xyz = torch.stack([
            grid_rho.cos() * grid_phi.sin(),
            grid_rho.sin() * grid_phi.sin(),
            grid_phi.cos(),
        ])

        sph_xyz = sph_xyz.reshape(1,3,-1).repeat(b,1,1)
        new_sph_xyz = vp_rot.transpose(1,2) @ sph_xyz

        sph_xyz = sph_xyz.transpose(1,2).contiguous().detach()
        new_sph_xyz = new_sph_xyz.transpose(1,2).contiguous().detach()
        x = x.reshape(b,c,n*n).contiguous()

        dist, idx = three_nn(sph_xyz, new_sph_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        new_x = three_interpolate(
            x, idx.detach(), weight.detach()
        ).reshape(b,c,n,n)
        return new_x

