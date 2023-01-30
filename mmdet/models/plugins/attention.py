import torch
from mmcv.cnn import xavier_init
from torch import nn
from ..utils import ConvModule
import torch.nn.functional as F


class SIM(nn.Module):
    """
    Scale Invariant Module
    """

    def __init__(self, branch, in_channel, out_channel, attention, conv_cfg, norm_cfg, activation):
        super(SIM, self).__init__()
        attention = attention.upper()
        assert attention in ['RCA', 'SCA', 'CA', None], 'only support rca, sca and ca attention module'
        self.branch = branch
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.attention = attention
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation

        convs = []
        for i in range(1, self.branch - 1):
            tmp = []
            tmp.append(nn.Sequential(
                nn.Conv2d(self.in_channel + self.out_channel, self.out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(32, self.out_channel),
                nn.ReLU()
            ))
            for j in range(1, i):
                tmp.append(nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, 3, stride=2, padding=1),
                                         nn.GroupNorm(32, self.out_channel),
                                         nn.ReLU()))
            convs.append(nn.Sequential(*tmp))

        tmp = []
        for j in range(self.branch - 1):
            tmp.append(nn.Sequential(
                nn.Conv2d(self.out_channel, self.out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(32, self.out_channel),
                nn.ReLU())
            )
        convs.append(nn.Sequential(*tmp))

        self.convs = nn.ModuleList(convs)
        if self.attention == 'RCA':
            self.attention = ResidualAttentionModule(self.out_channel, self.conv_cfg, self.norm_cfg)
        elif self.attention == 'CA':
            self.attention = ChannelAttention(self.out_channel)
        elif self.attention == 'SCA':
            self.attention = SpatialChannelAttention(self.out_channel, self.conv_cfg, self.norm_cfg)
        self.reduce = ConvModule(self.out_channel * self.branch, self.out_channel, 1, stride=1, padding=0,
                                 conv_cfg=self.conv_cfg, norm_cfg=norm_cfg, activation=self.activation, inplace=True)


    def forward(self, x):
        if self.branch < 3:
            return x
        res = []
        res.append(x)
        tmp = F.interpolate(self.convs[self.branch - 2](x), x.shape[2:], mode='bilinear', align_corners=True)
        res.append(tmp)
        for i in range(self.branch - 3, -1, -1):
            tmp = F.interpolate(self.convs[i](torch.cat([tmp, x], dim=1)), x.shape[2:], mode='bilinear',
                                align_corners=True)
            res.append(tmp)
        res = torch.cat(res, dim=1)
        res = self.reduce(res)
        if self.attention is None:
            return res
        return self.attention(res)


class SpatialChannelAttention(nn.Module):
    def __init__(self, channel, conv_cfg, norm_cfg):
        super(SpatialChannelAttention, self).__init__()
        self.channel = channel
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.rca = ResidualAttentionModule(channel, conv_cfg, norm_cfg)
        self.refine_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.refine_conv(scale)
        x = x * torch.sigmoid(scale)
        return self.rca(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.channel = channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(channel, channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(channel // 4, channel, bias=False), nn.Sigmoid())
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        n, c, h, w = x.shape
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.channel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        return F.relu(y)


class ResidualAttentionModule(nn.Module):
    def __init__(self, channel, conv_cfg, norm_cfg):
        super(ResidualAttentionModule, self).__init__()
        self.channel = channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(channel, channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(channel // 4, channel, bias=False), nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
                                   nn.GroupNorm(32, channel) )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        n, c, h, w = x.shape
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.channel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        # use x or y?
        return F.relu(y + self.conv2(y))
