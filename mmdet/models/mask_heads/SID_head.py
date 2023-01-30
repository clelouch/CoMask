import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from mmdet.core import mask_target
from mmdet.ops.dcn.deform_pool import normPRED


@HEADS.register_module
class SIDHead(nn.Module):

    def __init__(self, num_convs=8, roi_feat_size=14, in_channels=256, map_size=14,
                 num_classes=81, class_agnostic=False, conv_cfg=None, norm_cfg=None):
        super(SIDHead, self).__init__()
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_out_channels = 256
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.map_size = map_size
        self.num_classes = num_classes
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0

        self.convs = []
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            stride = 2 if i == 0 else 1
            self.convs.append(
                ConvModule(in_channels, self.conv_out_channels, 3, stride=stride,
                           padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=False))
        self.convs = nn.Sequential(*self.convs)

        self.SID_reg = nn.Conv2d(self.conv_out_channels, 4, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(self.conv_out_channels, 256, 2, stride=2)
        self.norm1 = nn.GroupNorm(32, 256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.norm2 = nn.GroupNorm(32, 256)
        self.SID_instance = nn.Conv2d(256, num_classes, 3, padding=1)

        self.maskscoringconv = nn.Sequential(nn.Conv2d(257, 256, 3, 1, 1), nn.GroupNorm(32, 256), nn.ReLU(),
                                             nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(32, 256), nn.ReLU())
        self.fcs = nn.ModuleList()
        for i in range(2):
            in_channels = 256 * 7 * 7 if i == 0 else 1024
            self.fcs.append(nn.Linear(in_channels, 1024))
        self.fc_instance_iou = nn.Linear(1024, 81)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_init(m)
        normal_init(self.SID_reg, std=0.001)
        nn.init.kaiming_normal_(
            self.deconv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv1.bias, 0)
        nn.init.kaiming_normal_(
            self.deconv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv2.bias, 0)

        for fc in self.fcs:
            kaiming_init(fc, a=1, mode='fan_in', nonlinearity='leaky_relu', distribution='uniform')
        normal_init(self.fc_instance_iou, std=0.01)

    def forward(self, x, idx=None):
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        x0 = self.convs(x)
        x_r = self.relu(self.SID_reg(x0))

        x2 = self.deconv1(x0)
        x2 = F.relu(self.norm1(x2), inplace=True)
        x2 = self.deconv2(x2)
        x_s = self.SID_instance(F.relu(self.norm2(x2), inplace=True))

        xs = x_s[idx > 0, idx].detach()
        xs = F.max_pool2d(xs.unsqueeze(1), 4, 4)
        xi = torch.cat([x0, xs.sigmoid()], dim=1)
        xi = self.maskscoringconv(xi)
        # xi = F.max_pool2d(xi, 2, 2)
        xi = xi.view(xi.size(0), -1)
        for fc in self.fcs:
            xi = self.relu(fc(xi))
        x_i = self.fc_instance_iou(xi)
        return x_r, x_s, x_i

    def get_target_mask(self, sampling_results, gt_masks, rcnn_train_cfg):
        # mix all samples (across images) together.
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results], dim=0)
        pos_gt_bboxes = torch.cat([res.pos_gt_bboxes for res in sampling_results], dim=0)

        assert pos_bboxes.shape == pos_gt_bboxes.shape

        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        # instances: a tensor of shape (N, M, M), where N is the number of instances, M (default: 28) is the mask_size
        instances = mask_target([res.pos_bboxes for res in sampling_results], pos_assigned_gt_inds, gt_masks,
                                rcnn_train_cfg)

        num_rois = pos_bboxes.shape[0]

        targets = pos_bboxes.new_zeros((num_rois, 4, self.map_size, self.map_size), dtype=torch.float)
        points = pos_bboxes.new_zeros((num_rois, 4, self.map_size, self.map_size), dtype=torch.float)

        for j in range(self.map_size):
            y = pos_bboxes[:, 1] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / self.map_size * (j + 0.5)

            for i in range(self.map_size):
                x = pos_bboxes[:, 0] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / self.map_size * (i + 0.5)

                targets[:, 0, j, i] = x - pos_gt_bboxes[:, 0]
                targets[:, 1, j, i] = pos_gt_bboxes[:, 2] - x
                targets[:, 2, j, i] = y - pos_gt_bboxes[:, 1]
                targets[:, 3, j, i] = pos_gt_bboxes[:, 3] - y

                points[:, 0, j, i] = x
                points[:, 1, j, i] = y
                points[:, 2, j, i] = pos_bboxes[:, 2] - pos_bboxes[:, 0]
                points[:, 3, j, i] = pos_bboxes[:, 3] - pos_bboxes[:, 1]

        return points, targets, instances

    def get_bboxes_avg(self, det_bboxes, SID_pred, SID_pred_instance, img_meta):
        assert det_bboxes.shape[0] == SID_pred.shape[0]

        det_bboxes = det_bboxes
        SID_pred = SID_pred
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]

        targets = torch.zeros((det_bboxes.shape[0], 4, self.map_size, self.map_size),
                              dtype=torch.float, device=SID_pred.device)

        idx = (torch.arange(0, self.map_size).float() + 0.5).cuda() / self.map_size

        h = (det_bboxes[:, 3] - det_bboxes[:, 1]).view(-1, 1, 1)
        w = (det_bboxes[:, 2] - det_bboxes[:, 0]).view(-1, 1, 1)
        y = det_bboxes[:, 1].view(-1, 1, 1) + h * idx.view(1, self.map_size, 1)
        x = det_bboxes[:, 0].view(-1, 1, 1) + w * idx.view(1, 1, self.map_size)

        targets[:, 0, :, :] = x - SID_pred[:, 0, :, :] * w
        targets[:, 2, :, :] = x + SID_pred[:, 1, :, :] * w
        targets[:, 1, :, :] = y - SID_pred[:, 2, :, :] * h
        targets[:, 3, :, :] = y + SID_pred[:, 3, :, :] * h

        reg_weights = F.interpolate(SID_pred_instance, targets.shape[2:], mode='bilinear', align_corners=True)
        reg_weights = torch.sigmoid(reg_weights)

        targets = targets.permute(0, 2, 3, 1).view(targets.shape[0], -1, 4)
        reg_weights = reg_weights.view(-1, self.map_size * self.map_size, 1)
        targets = torch.sum(targets * reg_weights, dim=1) / (torch.sum(reg_weights, dim=1) + 0.00001)

        aa = torch.isnan(targets)
        if aa.sum() != 0:
            print('nan error...')

        bbox_res = torch.cat([targets, cls_scores], dim=1)
        bbox_res[:, [0, 2]].clamp_(min=0, max=img_meta[0]['img_shape'][1] - 1)
        bbox_res[:, [1, 3]].clamp_(min=0, max=img_meta[0]['img_shape'][0] - 1)

        return bbox_res

    def get_target_maskiou(self, sampling_results, gt_masks, mask_pred, mask_targets, sample_idx):
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            gt_masks (list[ndarray]): Gt masks (the whole instance) of each
                image, binary maps with the same shape of the input image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (dict): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        area_ratios = map(self._get_area_ratio, pos_proposals,
                          pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        area_ratios = area_ratios[sample_idx]
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_pred = (mask_pred > 0.5).float()
        mask_pred_areas = mask_pred.sum((-1, -2))

        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)
        mask_iou_targets = overlap_areas / (torch.abs(
            mask_pred_areas + gt_full_areas - overlap_areas) + 1e-7)
        mask_iou_targets = mask_iou_targets.clamp(min=0)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals, pos_assigned_gt_inds, gt_masks):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance"""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # compute mask areas of gt instances (batch processing for speedup)
            gt_instance_mask_area = gt_masks.sum((-1, -2))
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                # crop the gt mask inside the proposal
                x1, y1, x2, y2 = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask[y1:y2 + 1, x1:x2 + 1]

                ratio = gt_mask_in_proposal.sum() / (
                        gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
                pos_proposals.device)
        else:
            area_ratios = pos_proposals.new_zeros((0,))
        return area_ratios
