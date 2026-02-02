

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def soft_argmax(heatmap, temp):
    B, C, H, W = heatmap.shape
    softmax_probs = F.softmax(heatmap.view(B, C, -1) / temp, dim=2).view(B, C, H, W)
    x_coords = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype).float()
    y_coords = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype).float()
    expected_x = torch.sum(softmax_probs * x_coords.view(1, 1, 1, W), dim=[2, 3])
    expected_y = torch.sum(softmax_probs * y_coords.view(1, 1, H, 1), dim=[2, 3])
    return torch.stack([expected_x, expected_y], dim=2)

def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


def topk(scores, K=1):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs


class FocalLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(FocalLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        pred = torch.clamp(pred, self.epsilon, 1. - self.epsilon)

        pos_inds = gt.eq(1).float();
        neg_inds = gt.lt(1).float();
        neg_weights = torch.pow(1 - gt, 4)
        loss = 0;
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds;
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum();
        pos_loss = pos_loss.sum();
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class WingLoss(nn.Module):
    def __init__(self, w=10, epsilon=2): super(WingLoss,
                                               self).__init__();self.w = w;self.epsilon = epsilon;self.C = self.w - self.w * np.log(
        1 + self.w / self.epsilon)

    def forward(self, pred, target): diff = torch.abs(pred - target);loss = torch.where(diff < self.w,
                                                                                        self.w * torch.log(
                                                                                            1 + diff / self.epsilon),
                                                                                        diff - self.C);return loss.mean()


def gwd_loss(pred, target):
    xy_p, R_p, S_p = xywhr2xyrs(pred);
    xy_t, R_t, S_t = xywhr2xyrs(target);
    xy_distance = (xy_p - xy_t).square().sum(dim=-1);
    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1));
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1));
    _t = Sigma_p.matmul(Sigma_t);
    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1);
    _t_det_sqrt = (S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1) * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)).clamp(
        0).sqrt();
    whr_distance = (S_p.square().diagonal(dim1=-2, dim2=-1).sum(dim=-1) + S_t.square().diagonal(dim1=-2, dim2=-1).sum(
        dim=-1) - 2 * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt());
    distance = (xy_distance + whr_distance).clamp(0);
    loss = 1 - 1 / (1 + distance)
    return loss.mean()


def xywhr2xyrs(xywhr):
    xy = xywhr[..., :2];
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7);
    r = torch.deg2rad(xywhr[..., 4]);
    cos_r = torch.cos(r);
    sin_r = torch.sin(r);
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).view(-1, 2, 2);
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S


class GWDLoss(nn.Module):
    def __init__(self, temperature=0.1): super(GWDLoss, self).__init__()

    def forward(self, pred_hm, pred_ab, pred_trig, pred_center, target_ellipse_xywhr):
        pred_xy = pred_center
        _, inds, _, _ = topk(torch.sigmoid(pred_hm), K=1)
        inds = inds.view(inds.size(0), -1)
        pred_ab_gathered = _transpose_and_gather_feat(pred_ab, inds).squeeze(1)
        pred_trig_gathered = _transpose_and_gather_feat(pred_trig, inds).squeeze(1)
        pred_wh = pred_ab_gathered * 2.0;
        sin2A = pred_trig_gathered[:, 0];
        cos2A = pred_trig_gathered[:, 1]
        pred_r = torch.rad2deg(torch.atan2(sin2A, cos2A) / 2.0)
        pred_xywhr = torch.cat([pred_xy, pred_wh, pred_r.unsqueeze(1)], dim=1)
        return gwd_loss(pred_xywhr, target_ellipse_xywhr)



class CtdetLoss(nn.Module):
    def __init__(self, loss_weight, wing_w=5.0, wing_epsilon=1.0):
        super(CtdetLoss, self).__init__()
        self.loss_weight = loss_weight
        self.crit_hm = FocalLoss()
        self.crit_coord = WingLoss(w=wing_w, epsilon=wing_epsilon)
        self.crit_ab = nn.L1Loss()
        self.crit_trig = nn.L1Loss()
        self.crit_iou = GWDLoss()
        self.temp = 1.0

    def set_temperature(self, temp): self.temp = temp

    def _calculate_eye_loss(self, pred, batch, suffix):
        pred_hm = pred[f'hm_{suffix}'];
        pred_ab = pred[f'ab_{suffix}'];
        pred_trig = pred[f'trig_{suffix}']
        gt_hm = batch[f'hm_{suffix}'];
        gt_center = batch[f'center_{suffix}'];
        gt_ab = batch[f'ab_{suffix}']
        gt_trig = batch[f'trig_{suffix}'];
        gt_ellipse = batch[f'ellipse_{suffix}'];
        gt_reg_mask = batch[f'reg_mask_{suffix}'].bool()

        if gt_reg_mask.sum() == 0: return None, {}
        valid_mask = gt_reg_mask.squeeze(1)

        hm_loss = self.crit_hm(torch.sigmoid(pred_hm), gt_hm)

        pred_center = soft_argmax(pred_hm, self.temp).squeeze(1)
        coord_loss = self.crit_coord(pred_center[valid_mask], gt_center[valid_mask])

        gt_ind = batch[f'ind_{suffix}']
        pred_ab_gathered = _transpose_and_gather_feat(pred_ab, gt_ind)

        pred_ab_gathered = F.relu(pred_ab_gathered)
        pred_trig_gathered = _transpose_and_gather_feat(pred_trig, gt_ind)
        ab_loss = self.crit_ab(pred_ab_gathered[valid_mask], gt_ab[valid_mask]);
        trig_loss = self.crit_trig(pred_trig_gathered[valid_mask], gt_trig[valid_mask])

        gt_ellipse_xywhr = gt_ellipse.clone()
        gt_ellipse_xywhr[:, 2:4] *= 2.0

        iou_loss = self.crit_iou(pred_hm[valid_mask], pred_ab[valid_mask], pred_trig[valid_mask],
                                 pred_center[valid_mask], gt_ellipse_xywhr[valid_mask])

        total_loss = (self.loss_weight['hm_weight'] * hm_loss +
                      self.loss_weight['coord_weight'] * coord_loss +
                      self.loss_weight['ab_weight'] * ab_loss +
                      self.loss_weight['trig_weight'] * trig_loss +
                      self.loss_weight['iou_weight'] * iou_loss)

        loss_show = {
            f"loss_hm_{suffix}": hm_loss, f"loss_coord_{suffix}": coord_loss, f"loss_ab_{suffix}": ab_loss,
            f"loss_trig_{suffix}": trig_loss, f"loss_iou_{suffix}": iou_loss,
        }
        return total_loss, loss_show

    def forward(self, pred, batch):
        total_loss_L, loss_show_L = self._calculate_eye_loss(pred, batch, "L")
        total_loss_R, loss_show_R = self._calculate_eye_loss(pred, batch, "R")
        loss_show = {**loss_show_L, **loss_show_R} if loss_show_L and loss_show_R else (
                    loss_show_L or loss_show_R or {})
        return total_loss_L, total_loss_R, loss_show