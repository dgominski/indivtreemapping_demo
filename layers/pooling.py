import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import layers.functional as LF
from layers.normalization import L2N
import torch.nn.functional as F

# --------------------------------------
# Pooling layers
# --------------------------------------

class MAC(nn.Module):

    def __init__(self):
        super(MAC,self).__init__()

    def forward(self, x):
        return LF.mac(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC,self).__init__()

    def forward(self, x):
        return LF.spoc(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'



class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = Parameter(torch.ones(1) * norm)


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=512, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class AmapSD(nn.Module):
    """Layer that keeps an internal average map of features and computes the sum of differences between the map
    and the input when used. The map is updated after each forward pass."""
    def __init__(self, num_features=2048, size=(12, 12)):
        super(AmapSD, self).__init__()
        self.amap = torch.nn.Parameter(torch.zeros((num_features, size[0], size[1])), requires_grad=False)
        self.n = 1

    def forward(self, x):
        output = LF.amapsd(x, self.amap)
        self.amap.data = self.amap.data + (x - self.amap.data) / self.n
        self.n += 1
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n=' + str(self.n) + ')'


class AmapSSD(nn.Module):
    """Layer that keeps an internal average map of features and computes the square sum of differences between the map
    and the input when used. The map is updated after each forward pass."""
    def __init__(self, num_features=2048, size=(12, 12)):
        super(AmapSSD, self).__init__()
        self.amap = torch.nn.Parameter(torch.zeros((1, num_features, size[0], size[1])), requires_grad=False)
        self.n = 1

    def forward(self, x):
        output = LF.amapssd(x, self.amap)
        self.amap.data += (x - self.amap.data) / self.n
        self.n += 1
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n=' + str(self.n) + ')'


class AmapPSD(nn.Module):
    """Layer that keeps an internal average map of features and computes the square sum of differences between the map
    and the input when used. The map is updated after each forward pass."""
    def __init__(self, num_features=2048, size=(12, 12), p=2.0):
        super(AmapPSD, self).__init__()
        self.amap = Parameter(torch.zeros((1, num_features, size[0], size[1])), requires_grad=False)
        self.std = Parameter(torch.ones((1, num_features)), requires_grad=False)
        self.S = Parameter(torch.ones((1, num_features)), requires_grad=False)
        self.p = Parameter(torch.ones(1)*p)
        self.n = 1

    def forward(self, x):
        output = LF.amappsd(x, self.amap, self.p)
        old_amap = self.amap.data
        self.amap.data = self.amap.data + (x - self.amap.data) / self.n
        self.S.data += torch.sum((x - self.amap.data) * (x - old_amap), dim=(2,3))
        self.std.data = torch.sqrt(self.S / self.n)
        self.n += 1
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'n=' + str(self.n) + ')'


class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC,self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class Rpool(nn.Module):

    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super(Rpool,self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = LF.roipool(x, self.rpool, self.L, self.eps) # size: #im, #reg, D, 1, 1

        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0]*s[1], s[2], s[3], s[4]) # size: #im x #reg, D, 1, 1

        # rvecs -> norm
        o = self.norm(o)

        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4]) # size: #im, #reg, D, 1, 1

        # aggregate regions into a single global vector per image
        if aggregate:
            # rvecs -> sumpool -> norm
            o = self.norm(o.sum(1, keepdim=False)) # size: #im, D, 1, 1

        return o

    def __repr__(self):
        return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'


class SpatialAttention2d(torch.nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    '''

    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_c, 512, 1, 1)  # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = torch.nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(num_features=512)
        self.conv2 = torch.nn.Conv2d(512, 1, 1, 1)  # 1x1 conv
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)  # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score
        '''
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class WeightedSum2d(torch.nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()

    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3), \
            'err: h, w of tensors x({}) and weights({}) must be the same.' \
                .format(x.size, weights.size)
        y = x * weights  # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))  # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1

    def __repr__(self):
        return self.__class__.__name__


class SE_ChannelWiseAttention(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=4, scale=1.0):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c // r, bias=True),
            #nn.BatchNorm1d(num_features=c // r),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c)
        return y


if __name__ == '__main__':
    a = torch.randn((4, 1024, 12, 12))
    print(a.mean())
    print(b.mean())
