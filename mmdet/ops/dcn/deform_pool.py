import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import deform_pool_cuda
from collections import OrderedDict
import torch.nn.functional as F
import math
from mmcv.cnn import xavier_init

COEFF = 12.0


class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


class AWPool(nn.Module):
    def __init__(self, channels):
        super(AWPool, self).__init__()

        self.weight = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                ('bn', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.weight[0].weight.data.fill_(0.0)

    def forward(self, x, kernel=3, stride=2, padding=1):
        weight = self.weight(x).exp()
        return F.avg_pool2d(x * weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)


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
        x = x * self.sigmoid(scale)
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


# TODO: when try to set bias=false in conv2d layer, we need to modify the Linear layer as well
class ResidualAttentionModule(nn.Module):
    def __init__(self, channel, conv_cfg, norm_cfg):
        super(ResidualAttentionModule, self).__init__()
        self.channel = channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(channel, channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(channel // 4, channel, bias=False), nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
                                   nn.GroupNorm(32, channel), )
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


COEFF = 12.0


# normalize the predicted probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()

        self.weight = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
                ('gn', nn.GroupNorm(32, channels)),
                ('sigmoid', nn.Sigmoid()),
            ))
        )

    def init_layer(self):
        self.weight[0].weight.data.fill_(0.0)

    def forward(self, x, kernel=3, stride=2, padding=1):
        weight = self.weight(x)
        return F.avg_pool2d(x * weight, kernel, stride, padding)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels,
                no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=.0):
        # TODO: support unsquare RoIs
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        assert out_h == out_w
        out_size = out_h  # out_h and out_w must be equal

        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std

        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError

        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(
            data, rois, offset, output, output_count, ctx.no_trans,
            ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size,
            ctx.part_size, ctx.sample_per_part, ctx.trans_std)

        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)

        deform_pool_cuda.deform_psroi_pooling_cuda_backward(
            grad_output, data, rois, offset, output_count, grad_input,
            grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels,
            ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part,
            ctx.trans_std)
        return (grad_input, grad_rois, grad_offset, None, None, None, None,
                None, None, None, None)


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None,
                 sample_per_part=4, trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = _pair(out_size)
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels,
                                  self.no_trans, self.group_size, self.part_size, self.sample_per_part,
                                  self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):
    """
    论文中的Discriminative RoI pooling，注意spatial_scale = 1 / featmap_strides
            out_size=7,
            sample_per_part=2,
            out_channels=256,
            no_trans=False,
            group_size=1,
            trans_std=0.1),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
    """

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1,
                 part_size=None, sample_per_part=4, trans_std=.0, num_offset_fcs=3,
                 deform_fc_channels=1024):
        super(DeformRoIPoolingPack,
              self).__init__(spatial_scale, out_size, out_channels, no_trans,
                             group_size, part_size, sample_per_part, trans_std)

        self.num_offset_fcs = num_offset_fcs
        self.deform_fc_channels = deform_fc_channels
        self.offset_size = (3, 3)
        self.awp = AWPool(256)  # 其实相当是一个注意力模块
        self.awp.init_layer()

        if not no_trans:
            seq = []
            ic = self.offset_size[0] * self.offset_size[1] * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1] * 2
                seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        n = rois.shape[0]
        if n == 0:
            return data.new_empty(n, self.out_channels, self.out_size[0],
                                  self.out_size[1])
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                      self.offset_size, self.out_channels,
                                      self.no_trans, self.group_size,
                                      self.offset_size, self.sample_per_part,
                                      self.trans_std)
        else:
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.offset_size, self.out_channels, True,
                                   self.group_size, self.offset_size[0],
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size[0], self.out_size[1])
            offset = F.interpolate(offset, scale_factor=2, mode='nearest')

            roi_feats = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                           (self.out_size[0] * 2, self.out_size[1] * 2), self.out_channels,
                                           self.no_trans, self.group_size,
                                           self.part_size * 2, self.sample_per_part,
                                           self.trans_std)
            roi_feats = self.awp(roi_feats)
            return roi_feats


class DeformRoIPoolingWithAttention(DeformRoIPooling):
    # 与常规的roipooling不同，常规的roipooling采取在一个bin里面采样多个点(sample_per_part)
    # 然后对该bin里面的点求平均，这里我们采用利用注意力机制的办法，为每个点赋予一个权值，所以我们
    # 直接让pooling获取out_size * 2, out_size * 2的特征图，这样我们下一步对该特征图进行处理
    # 求它的权值，相乘然后再平均池化就是我们所需要的答案。这样的话，我们就不需要采样多个点，而是直接每个
    # bin一个点。
    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1,
                 part_size=None, sample_per_part=4, trans_std=.0, num_offset_fcs=3,
                 deform_fc_channels=1024, attention=None):
        super(DeformRoIPoolingWithAttention, self).__init__(spatial_scale, out_size, out_channels, no_trans,
                                                            group_size, part_size, sample_per_part, trans_std)

        self.num_offset_fcs = num_offset_fcs
        self.deform_fc_channels = deform_fc_channels
        self.offset_size = (7, 7)
        self.attention = attention
        self.ca = None
        if self.attention == 'RCA':
            self.sa = SpatialAttention(256)  # 其实相当是一个注意力模块
            self.sa.init_layer()
            self.ca = ResidualAttentionModule(channel=256, conv_cfg=None, norm_cfg='GN')
        elif self.attention == 'CA':
            self.sa = SpatialAttention(256)  # 其实相当是一个注意力模块
            self.sa.init_layer()
            self.ca = ChannelAttention(channel=256)
        elif self.attention == 'SCA':
            self.sa = SpatialChannelAttention(channel=256, conv_cfg=None, norm_cfg='GN')
        else:
            self.sa = SpatialAttention(256)  # 其实相当是一个注意力模块
            self.sa.init_layer()

        if not no_trans:
            seq = []
            ic = self.offset_size[0] * self.offset_size[1] * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1] * 2
                seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        n = rois.shape[0]
        if n == 0:
            return data.new_empty(n, self.out_channels, self.out_size[0],
                                  self.out_size[1])
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                      self.offset_size, self.out_channels,
                                      self.no_trans, self.group_size,
                                      self.offset_size, self.sample_per_part,
                                      self.trans_std)
        else:
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.offset_size, self.out_channels, True,
                                   self.group_size, self.offset_size[0],
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size[0], self.out_size[1])
            offset = F.interpolate(offset, scale_factor=2, mode='nearest')

            roi_feats = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                           (self.out_size[0] * 2, self.out_size[1] * 2), self.out_channels,
                                           self.no_trans, self.group_size,
                                           self.part_size * 2, self.sample_per_part,
                                           self.trans_std)
            roi_feats = self.sa(roi_feats)
            if self.ca is None:
                return roi_feats
            roi_feats = self.ca(roi_feats)
            return roi_feats


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 num_offset_fcs=3,
                 num_mask_fcs=2,
                 deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack,
              self).__init__(spatial_scale, out_size, out_channels, no_trans,
                             group_size, part_size, sample_per_part, trans_std)

        self.num_offset_fcs = num_offset_fcs
        self.num_mask_fcs = num_mask_fcs
        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            offset_fc_seq = []
            ic = self.out_size[0] * self.out_size[1] * self.out_channels
            for i in range(self.num_offset_fcs):
                if i < self.num_offset_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1] * 2
                offset_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_offset_fcs - 1:
                    offset_fc_seq.append(nn.ReLU(inplace=True))
            self.offset_fc = nn.Sequential(*offset_fc_seq)
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

            mask_fc_seq = []
            ic = self.out_size[0] * self.out_size[1] * self.out_channels
            for i in range(self.num_mask_fcs):
                if i < self.num_mask_fcs - 1:
                    oc = self.deform_fc_channels
                else:
                    oc = self.out_size[0] * self.out_size[1]
                mask_fc_seq.append(nn.Linear(ic, oc))
                ic = oc
                if i < self.num_mask_fcs - 1:
                    mask_fc_seq.append(nn.ReLU(inplace=True))
                else:
                    mask_fc_seq.append(nn.Sigmoid())
            self.mask_fc = nn.Sequential(*mask_fc_seq)
            self.mask_fc[-2].weight.data.zero_()
            self.mask_fc[-2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        n = rois.shape[0]
        if n == 0:
            return data.new_empty(n, self.out_channels, self.out_size[0],
                                  self.out_size[1])
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                      self.out_size, self.out_channels,
                                      self.no_trans, self.group_size,
                                      self.part_size, self.sample_per_part,
                                      self.trans_std)
        else:
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.out_size, self.out_channels, True,
                                   self.group_size, self.part_size,
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size[0], self.out_size[1])
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size[0], self.out_size[1])
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std) * mask
