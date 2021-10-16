#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class IOUsloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # pred = [num, [x, y, w, h]] torch.Size([261, 4])
        # target = xyxy
        assert pred.shape[0] == target.shape[0]
        # print(pred.shape)
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # print(pred.shape)

        # 左下? torch.Size([261, 2])
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        # 右上?
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        # torch.prod = 張量中所有元素的乘積
        # area_p = 預測框的面積
        area_p = torch.prod(pred[:, 2:], 1)
        # area_g = 目標框的面積
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        
        # overlap
        area_i = torch.prod(br - tl, 1) * en
        
        # union
        area_u = area_p + area_g - area_i
        
        iou = (area_i) / (area_u + 1e-16)  # IoU

        if self.loss_type == "giou" or self.loss_type == "diou" or self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            if self.loss_type == "diou" or self.loss_type == "ciou":
                enclose_wh = (c_br - c_tl).clamp(min=0)
                cw = enclose_wh[:, 0]
                ch = enclose_wh[:, 1]
                c2 = c2 = cw**2 + ch**2 + 1e-16

                b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
                b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
                b2_x1, b2_y1 = target[:, 0], target[:, 1]
                b2_x2, b2_y2 = target[:, 2], target[:, 3]

                left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
                right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
                rho2 = left + right
                if self.loss_type == "diou":
                    diou = iou - rho2 / c2  # DIoU
                    loss = 1 - diou.clamp(min=-1.0, max=1.0)
                else:
                    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-16
                    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-16
                    factor = 4 / math.pi**2
                    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + 1e-16))
                        ciou = iou - (rho2 / c2 + v * alpha)  # CIoU
                    loss = 1 - ciou.clamp(min=-1.0, max=1.0)
            else:
                area_c = torch.prod(c_br - c_tl, 1)
                giou = iou - (area_c - area_u) / area_c.clamp(1e-16)  # GIoU
                
                loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            loss = 1 - iou ** 2
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# rotation
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss