#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision
import math
import pdb
import time
import cv2
from yolox.utils.Rotated_IoU.oriented_iou_loss import cal_iou, box2corners_th, enclosing_box

from yolox.utils.Rotated_IoU.box_intersection_2d import oriented_box_intersection_2d

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "postprocess_rotation",
    "postprocess_rotation_head",
    "xyxy2xyxyxyxy",
    "xywh2cxcywh",
    "adjust_box_anns_without_clip",
    # "cvminAreaRect2longsideformat",
]

def box_rotated_iou(boxes1, boxes2):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    # (N, N, 5)
    box1_ = boxes1.expand(boxes1.size(1), -1, -1).permute(1,0,2)
    # (N, N, 5)
    box2_ = boxes2.expand(boxes2.size(1), -1, -1)

    corners1 = box2corners_th(box1_)
    corners2 = box2corners_th(box2_)

    # (N, N)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)

    area1 = boxes1[:, :, 2] * boxes1[:, :, 3]
    area2 = boxes2[:, :, 2] * boxes2[:, :, 3]

    u = area1 + area2 - inter_area
    iou = inter_area / u

    return iou, corners1, corners2, u

def fast_nms(boxes, scores, giou=False, diou=False, ciou=False, cluster=False, spm=False, NMS_threshold=0.5, enclosing_type="smallest"):
    '''Fast NMS results
    Arguments:
        boxes (Tensor[N, 5])
        scores (Tensor[N])
    Returns:
        Fast NMS results
    '''
    # 對框按得分降序排列
    # scores, idx = scores.sort(0, descending=True)
    # boxes = boxes[idx]
    # boxes = boxes[:3000]

    iou, corners1, corners2, u = box_rotated_iou(boxes.float().unsqueeze(0), boxes.float().unsqueeze(0))
    if giou:
        w, h = enclosing_box(corners1, corners2, enclosing_type)
        area_c =  w*h
        iou = iou - ( area_c - u )/area_c
    if diou:
        w, h = enclosing_box(corners1, corners2, enclosing_type)
        c2 = w*w + h*h      # (B, N)
        x_offset = boxes.float().unsqueeze(0)[...,0] - boxes.float().unsqueeze(0)[..., 0]
        y_offset = boxes.float().unsqueeze(0)[...,1] - boxes.float().unsqueeze(0)[..., 1]
        d2 = x_offset*x_offset + y_offset*y_offset
        iou = iou - d2/c2
    if ciou:
        pass
    iou.triu_(diagonal=1)  # 上三角化
    if cluster:
        # cluster_nms
        C = iou
        for i in range(200):    
            A=C
            maxA = A.max(dim=0)[0]   # 列最大值向量
            E = (maxA < NMS_threshold).float().unsqueeze(1).expand_as(A)   # 對角矩陣E的替代
            C = iou.mul(E)     # 按元素相乘
            if A.equal(C)==True:     # 終止條件
                break
        if spm:
            scores = torch.prod(torch.exp(-C**2/0.2),0)*scores  # 懲罰得分
            keep = scores > NMS_threshold  # 得分阈值筛选 這邊要調
        else:
            keep = maxA < NMS_threshold  # 列最大值向量，二值化
        del C
    else:
        # fast_nms
        keep = iou.max(dim=0)[0] < NMS_threshold  # 列最大值向量，二值化
    
    del scores, iou
    return keep


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def postprocess_rotation(prediction, num_classes, num_angles, conf_thre=0.7, nms_thre=0.45, isnms=True):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        angles_conf, angles_pred = torch.max(image_pred[:, 5 + num_classes:], 1, keepdim=True)


        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), angles_pred), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        # batched_nms 會增加召回率和 mAP  適用於訓練及評估   純nms適用於demo
        # batched_nms：根據每個類別進行過濾，只對同一種類別進行計算期票和閾值過濾。
        # NMS：不區分類別對所有BBOX進行過濾如果有不同類別的BBOX重疊的話會導致被過濾掉並不會分開計算。
        if isnms:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def postprocess_rotation_head(prediction, num_classes, num_angles, conf_thre=0.7, nms_thre=0.45, isnms=True, isfastnms=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        angles_conf, angles_pred = torch.max(image_pred[:, 5+num_classes:5+num_classes+num_angles], 1, keepdim=True)

        heads_conf, heads_pred = torch.max(image_pred[:, 5+num_classes+num_angles:], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), angles_pred, heads_pred), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        if isnms:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
            detections = detections[nms_out_index]
        if isfastnms:
            scores = detections[:, 4] * detections[:, 5]
            scores, idx = scores.sort(0, descending=True)
            detections = detections[idx]
            detections = detections[:1000]

            dets = detections.clone()
            dets[:, 2] = detections[:, 2] - detections[:, 0]
            dets[:, 3] = detections[:, 3] - detections[:, 1]
            dets[:, 0] = dets[:, 0] + dets[:, 2]*0.5
            dets[:, 1] = dets[:, 1] + dets[:, 3]*0.5

            

            nms_out_index = fast_nms(boxes=dets[:, :5], scores=scores, NMS_threshold=0.1, cluster=True, giou=True)
            detections = detections[nms_out_index]
            del dets
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
        del detections
    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, inplace=False, iou_mode='iou'):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if inplace:
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br_hw = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            br_hw.sub_(tl)  # hw
            br_hw.clamp_min_(0)  # [rows, 2]
            del tl
            area_ious = torch.prod(br_hw, 2)  # area
            del br_hw
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br_hw = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )
            br_hw.sub_(tl)  # hw
            br_hw.clamp_min_(0)  # [rows, 2]
            del tl
            area_ious = torch.prod(br_hw, 2)  # area
            del br_hw
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)

        union = (area_a[:, None] + area_b - area_ious)
        area_ious.div_(union)  # ious

        return area_ious
    else:
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_g = torch.prod(bboxes_a[:, 2:], 1)
            area_p = torch.prod(bboxes_b[:, 2:], 1)

            br_hw = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )
            br_hw.sub_(tl)  # hw
            # print('test')
            # print(br_hw.shape)
            br_hw.clamp_min_(0)  # [rows, 2]
            # print(br_hw.shape)
            del tl
            area_i = torch.prod(br_hw, 2)  # area
            del br_hw

        # union
        # print('union')
        # print(area_g.shape)
        # print(area_g[:, None].shape)
        # print(area_p.shape)
        # print((area_g[:, None] + area_p).shape)
        # print(area_i.shape)
        area_u = (area_g[:, None] + area_p - area_i)

        ious = area_i / (area_u + 1e-16)

        if iou_mode == "giou" or iou_mode == "diou" or iou_mode == "ciou":
            # print(bboxes_a.shape)
            # print(bboxes_b.shape)
            c_tl = torch.min(
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2), (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2)
            )
            c_br = torch.max(
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2), (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2)
            )
            if iou_mode == "diou" or iou_mode == "ciou":
                enclose_wh = (c_br - c_tl).clamp(min=0)
                cw = enclose_wh[:, 0]
                ch = enclose_wh[:, 1]
                # c2 = cw**2 + ch**2 + 1e-16
                # print(cw.shape)
                c2 = torch.prod(cw, 1) + torch.prod(ch, 1) + 1e-16


                rho2 = (bboxes_a[:, None, 0] - bboxes_b[:, 0])**2 + (bboxes_a[:, None, 1] - bboxes_b[:, 1])**2


                if iou_mode == "diou":
                    # print('diou')
                    # print(rho2.shape)
                    # print(c2.shape)
                    # print(c2[:, None].shape)
                    # print(ious.shape)
                    # print(rho2 / c2[:, None])
                    diou = ious - rho2 / c2[:, None]  # DIoU
                    return diou
                else:
                    w1, h1 = bboxes_b[:, 2], bboxes_b[:, 3]
                    w2, h2 = bboxes_a[:, None, 2], bboxes_a[:, None, 3]
                    factor = 4 / math.pi**2
                    # print('ciou')
                    # print(w2.shape)
                    # print(h2.shape)
                    # print(torch.atan(w2 / h2) - torch.atan(w1 / h1))
                    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - ious + (1 + 1e-16))
                    ciou = ious - (rho2 / c2[:, None] + v * alpha)  # CIoU = DIou - αv
                    return ciou
            else:
                c_br.sub_(c_tl)  # hw
                c_br.clamp_min_(0)  # [rows, 2]
                del c_tl
                area_c = torch.prod(c_br, 2)  # convex area
                del c_br

                giou = ious - (area_c - area_u) / (area_c + 1e-16)  # GIoU = IoU - (C-A∪B)/C
                return giou
        else:
            # print('===============================================hnsrthstrhrsth')
            return ious

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox

def adjust_box_anns_without_clip(bbox, scale_ratio, padw, padh):
    bbox[:, 0::2] = bbox[:, 0::2] * scale_ratio + padw
    bbox[:, 1::2] = bbox[:, 1::2] * scale_ratio + padh
    return bbox

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


# fix bug
def xyxy2xyxyxyxy(bboxes):
    return bboxes

# fix bug
def xywh2cxcywh(bboxes):
    return bboxes

# fix bug
def cxcywh2xyxy(bboxes):
    new_bboxes = bboxes.copy()
    new_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    new_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    new_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    new_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return new_bboxes

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside, isdebug=False):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    兩者區別為:
            當opencv表示法中width為最長邊時（包括正方形的情況），則兩種表示方法一致
            當opencv表示法中width不為最長邊 ，則最長邊表示法的角度要在opencv的Θ基礎上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最長邊
    @param shortside: 最短邊
    @param theta_longside: 最長邊和x軸逆時針旋轉的夾角，逆時針方向角度為負 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x軸逆時針旋轉碰到的第一條邊最長邊
            height: 與width不同的邊
            theta: x軸逆時針旋轉與width的夾角，由於原點位於圖像的左上角，逆時針旋轉角度為負 [-90, 0)
    '''
    if ((theta_longside >= -180) and (theta_longside < -90)):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if (theta < -90) or (theta >= 0):
        if isdebug:
            print('當前θ=%.1f，超出opencv的θ定義範圍[-90, 0)' % theta)

    return ((int(x_c), int(y_c)), (int(width), int(height)), theta)