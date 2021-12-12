#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math
from yolox.utils import cal_iou, cal_giou, cal_diou, enclosing_box

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou", alpha=None):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.alpha = alpha

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

        if self.alpha:
            iou = torch.pow((area_i) / (area_u + 1e-16), self.alpha)  # IoU = (A∩B)/(A∪B)
        else:
            iou = (area_i) / (area_u + 1e-16)  # IoU = (A∩B)/(A∪B)

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
                if self.alpha:
                    c2 = (cw**2 + ch**2) ** self.alpha + 1e-16
                else:
                    c2 = cw**2 + ch**2 + 1e-16

                # b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
                # b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
                # b2_x1, b2_y1 = target[:, 0], target[:, 1]
                # b2_x2, b2_y2 = target[:, 2], target[:, 3]
                
                # left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
                # right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
                # rho2 = left + right
                
                if self.alpha:
                    rho2 = ((target[:, 0] - pred[:, 0])**2 + (target[:, 1] - pred[:, 1])**2) ** self.alpha
                else:
                    rho2 = (target[:, 0] - pred[:, 0])**2 + (target[:, 1] - pred[:, 1])**2



                if self.loss_type == "diou":
                    diou = iou - rho2 / c2  # DIoU
                    loss = 1 - diou.clamp(min=-1.0, max=1.0)
                else:
                    # w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-16
                    # w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-16
                    w1, h1 = pred[:, 2], pred[:, 3]
                    w2, h2 = target[:, 2], target[:, 3]
                    factor = 4 / math.pi**2
                    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha_ciou = v / (v - iou + (1 + 1e-16))
                    if self.alpha:
                        iou - (rho2 / c2 + torch.pow(v * alpha_ciou + 1e-16, self.alpha))  # CIoU
                    else:
                        ciou = iou - (rho2 / c2 + v * alpha_ciou)  # CIoU = DIou - αv
                    loss = 1 - ciou.clamp(min=-1.0, max=1.0)
            else:
                area_c = torch.prod(c_br - c_tl, 1)  # convex area
                if self.alpha:
                    giou = iou - torch.pow((area_c - area_u) / area_c + 1e-16, self.alpha)  # GIoU
                else:
                    giou = iou - (area_c - area_u) / area_c.clamp(1e-16)  # GIoU = IoU - (C-A∪B)/C
                
                loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            loss = 1 - iou ** 2
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class RIOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(RIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # pred = [num, [x, y, w, h]] torch.Size([261, 4])
        # target = xyxy
        assert pred.shape[0] == target.shape[0]
        print(pred.shape)
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # print(pred.shape)

        # 左下? torch.Size([261, 2])
        # tl = torch.max(
        #     (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        # )
        # # 右上?
        # br = torch.min(
        #     (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        # )

        # # torch.prod = 張量中所有元素的乘積
        # # area_p = 預測框的面積
        # area_p = torch.prod(pred[:, 2:], 1)
        # # area_g = 目標框的面積
        # area_g = torch.prod(target[:, 2:], 1)

        # en = (tl < br).type(tl.type()).prod(dim=1)
        
        # # overlap
        # area_i = torch.prod(br - tl, 1) * en
        
        # # union
        # area_u = area_p + area_g - area_i

        # iou = (area_i) / (area_u + 1e-16)  # IoU = (A∩B)/(A∪B)
        # pred_ctrx = torch.unsqueeze((pred[:, 0] + pred[:, 2]) * 0.5, 1)
        # pred_ctry = torch.unsqueeze((pred[:, 1] + pred[:, 3]) * 0.5, 1)
        # pred_w = torch.unsqueeze(pred[:, 2] - pred[:, 0], 1)
        # pred_h = torch.unsqueeze(pred[:, 3] - pred[:, 1], 1)
        # pred_angle = torch.unsqueeze(torch.deg2rad(pred_angles.float()), 1)
        # pred_ = torch.unsqueeze(torch.cat((pred_ctrx, pred_ctry, pred_w, pred_h, pred_angle), 1), 0)


        # target_ctrx = torch.unsqueeze((target[:, 0] + target[:, 2]) * 0.5, 1)
        # target_ctry = torch.unsqueeze((target[:, 1] + target[:, 3]) * 0.5, 1)
        # target_w = torch.unsqueeze(target[:, 2] - target[:, 0], 1)
        # target_h = torch.unsqueeze(target[:, 3] - target[:, 1], 1)
        # target_angle = torch.unsqueeze(torch.deg2rad(target_angles.float()), 1)
        # target_ = torch.unsqueeze(torch.cat((target_ctrx, target_ctry, target_w, target_h, target_angle), 1), 0)



        # # gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
        # gious, iou = cal_giou(pred_, target_, "smallest")
        # loss = 1 - gious
        # return loss
        print(pred.shape)
        print(target.shape)
        iou = cal_iou(pred, target)

        if self.loss_type == "giou" or self.loss_type == "diou" or self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            area_c = torch.prod(c_br - c_tl, 1)  # convex area
            if self.alpha:
                giou = iou - torch.pow((area_c - area_u) / area_c + 1e-16, self.alpha)  # GIoU
            else:
                giou = iou - (area_c - area_u) / area_c.clamp(1e-16)  # GIoU = IoU - (C-A∪B)/C
            
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
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss() 這裡的loss_fcn基礎定義為多分類交叉熵損失函數
        self.gamma = gamma  # Focal loss中的gamma參數 用於削弱簡單樣本對loss的貢獻程度
        self.alpha = alpha  # Focal loss中的alpha參數 用於平衡正負樣本個數不均衡的問題
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # 需要將Focal loss應用於每一個樣本之中

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true) # 這裡的loss代表正常的BCE loss結果
        
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        # 通過sigmoid函數返回得到的概率 即Focal loss 中的y'
        pred_prob = torch.sigmoid(pred) 
        # 這裡對p_t屬於正樣本還是負樣本進行了判別，正樣本對應true=1,即Focal loss中的大括號
        # 正樣本時 返回pred_prob為是正樣本的概率y'，負樣本時為1-y'
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # 這裡同樣對alpha_factor進行了屬於正樣本還是負樣本的判別，即Focal loss中的
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        # 這裡代表Focal loss中的指數項
        # 正樣本對應(1-y')的gamma次方 負樣本度對應y'的gamma次方
        modulating_factor = (1.0 - p_t) ** self.gamma
        # 返回最終的loss大tensor
        loss *= alpha_factor * modulating_factor
        # 以下幾個判斷代表返回loss的均值/和/本體了
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss