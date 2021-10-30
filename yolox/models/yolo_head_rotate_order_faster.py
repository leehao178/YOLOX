#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, angle_smooth_label  # rotation

from .losses import IOUloss, FocalLoss  # rotation
from .network_blocks import BaseConv, DWConv

import numpy as np  # rotation

from torch.cuda.amp import autocast


class YOLOXRotateHeadOrderHead(nn.Module):
    def __init__(
        self,
        num_classes,
        num_angles,
        iou_loss="iou",
        cls_loss="bce",
        obj_loss="bce",
        ang_loss="focalloss",
        head_loss="bce",
        label_type=0,
        label_raduius=6,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.num_angles = num_angles  # rotation
        self.label_type = label_type  # rotation
        self.label_raduius = label_raduius  # rotation
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()  # rotation
        self.head_preds = nn.ModuleList()  # headorder
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.ang_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_angles,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )  # rotation
            self.head_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )  # headorder
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type=iou_loss)
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        # faster
        self.xy_shifts = [torch.zeros(1)] * len(in_channels)
        self.org_grids = [torch.zeros(1)] * len(in_channels)
        self.grid_sizes = [[0, 0, 0] for _ in range(len(in_channels))]
        self.expanded_strides = [torch.zeros(1)] * len(in_channels)
        self.center_ltrbes = [torch.zeros(1)] * len(in_channels)
        # gt框中心點的2.5個網格半徑的矩形框內的anchor
        self.center_radius = 2.5
        
        if obj_loss == "bce":
            self.obj_loss = nn.BCEWithLogitsLoss(reduction="none")
        elif obj_loss == "focalloss":
            self.obj_loss = FocalLoss(nn.BCEWithLogitsLoss(reduction="none"))
        
        if ang_loss == "focalloss":
            self.ang_loss = FocalLoss(nn.BCEWithLogitsLoss(reduction="none"))
        elif ang_loss == "bce":
            self.ang_loss = nn.BCEWithLogitsLoss(reduction="none")
        
        if head_loss == "bce":
            self.head_loss = nn.BCEWithLogitsLoss(reduction="none")
        elif head_loss == "focalloss":
            self.head_loss = FocalLoss(nn.BCEWithLogitsLoss(reduction="none"))
        
        if cls_loss == "bce":
            self.cls_loss = nn.BCEWithLogitsLoss(reduction="none")
        elif cls_loss == "focalloss":
            self.cls_loss = FocalLoss(nn.BCEWithLogitsLoss(reduction="none"))


    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        for conv in self.ang_preds:  # rotation
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.head_preds:  # headorder
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        cls_preds = []
        bbox_preds = []
        obj_preds = []
        org_xy_shifts = []
        xy_shifts = []
        center_ltrbes = []

        ang_preds = []
        head_preds = []

        cls_xs = xin[0::2]
        reg_xs = xin[1::2]
        in_type = xin[0].type()
        h, w = reg_xs[0].shape[2:4]
        h *= self.stride[0]
        w *= self.stride[0]

        # for k, (stride_this_level, cls_x, reg_x) in enumerate(
        #     zip(self.stride, cls_xs, reg_xs)
        # ):
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            ang_output = self.ang_preds[k](cls_feat)  # rotation
            head_output = self.head_preds[k](cls_feat)  # headorder

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # cls_output = self.cls_preds[k](cls_x)  # [batch_size, num_classes, hsize, wsize]

            # reg_output = self.reg_preds[k](reg_x)  # [batch_size, 4, hsize, wsize]
            # obj_output = self.obj_preds[k](reg_x)  # [batch_size, 1, hsize, wsize]

            if self.training:
                # output = torch.cat([reg_output, obj_output, cls_output, ang_output, head_output], 1)   # rotation headorder
                # output, grid = self.get_output_and_grid(
                #     output, k, stride_this_level, xin[0].type() # 6400*85, 1600*85, 400*85
                # )
                # x_shifts.append(grid[:, :, 0])
                # y_shifts.append(grid[:, :, 1])
                # expanded_strides.append(
                #     torch.zeros(1, grid.shape[1])
                #     .fill_(stride_this_level)
                #     .type_as(xin[0])
                # )
                batch_size = cls_output.shape[0]
                hsize, wsize = cls_output.shape[-2:]
                size = hsize * wsize
                cls_output = cls_output.view(batch_size, -1, size).permute(0, 2, 1).contiguous()  # [batch_size, num_classes, hsize*wsize] -> [batch_size, hsize*wsize, num_classes]
                reg_output = reg_output.view(batch_size, 4, size).permute(0, 2, 1).contiguous()  # [batch_size, 4, hsize*wsize] -> [batch_size, hsize*wsize, 4]
                obj_output = obj_output.view(batch_size, 1, size).permute(0, 2, 1).contiguous()  # [batch_size, 1, hsize*wsize] -> [batch_size, hsize*wsize, 1]
                
                ang_output = ang_output.view(batch_size, -1, size).permute(0, 2, 1).contiguous()  # [batch_size, num_classes, hsize*wsize] -> [batch_size, hsize*wsize, num_classes]
                head_output = head_output.view(batch_size, -1, size).permute(0, 2, 1).contiguous()  # [batch_size, num_classes, hsize*wsize] -> [batch_size, hsize*wsize, num_classes]
                if self.use_l1:
                    # batch_size = reg_output.shape[0]
                    # hsize, wsize = reg_output.shape[-2:]
                    # reg_output = reg_output.view(
                    #     batch_size, self.n_anchors, 4, hsize, wsize
                    # )
                    # reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    #     batch_size, -1, 4
                    # )
                    origin_preds.append(reg_output.clone())
                
                reg_output1, grid, xy_shift, expanded_stride, center_ltrb = self.get_output_and_grid(reg_output, hsize, wsize, k, stride_this_level, in_type)

                org_xy_shifts.append(grid)  # 網格x, y坐標, [1, 1*hsize*wsize, 2]
                xy_shifts.append(xy_shift)  # 網格x, y坐標, [1, 1*hsize*wsize, 2]
                expanded_strides.append(expanded_stride)   # dims: [1, hsize*wsize]
                center_ltrbes.append(center_ltrb)  # [1, 1*hsize*wsize, 4]
                cls_preds.append(cls_output)
                bbox_preds.append(reg_output1)
                obj_preds.append(obj_output)

                ang_preds.append(ang_output)
                head_preds.append(head_output)

                
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)
            #     output = torch.cat(
            #         [reg_output, obj_output.sigmoid(), cls_output.sigmoid(), ang_output.sigmoid(), head_output.sigmoid()], 1   # rotation headorder
            #     )

            # outputs.append(output)

        if self.training:
            bbox_preds = torch.cat(bbox_preds, 1)  # [batch, n_anchors_all, 4]
            obj_preds = torch.cat(obj_preds, 1)  # [batch, n_anchors_all, 1]
            cls_preds = torch.cat(cls_preds, 1)  # [batch, n_anchors_all, n_cls]

            ang_preds = torch.cat(ang_preds, 1)  # [batch, n_anchors_all, n_cls]
            head_preds = torch.cat(head_preds, 1)  # [batch, n_anchors_all, n_cls]



            org_xy_shifts = torch.cat(org_xy_shifts, 1)  # [1, n_anchors_all, 2]
            xy_shifts = torch.cat(xy_shifts, 1)  # [1, n_anchors_all, 2]
            expanded_strides = torch.cat(expanded_strides, 1)
            center_ltrbes = torch.cat(center_ltrbes, 1)  # [1, n_anchors_all, 4]
            
            if self.use_l1:
                origin_preds = torch.cat(origin_preds, 1)  # dims: [n, n_anchors_all, 4]
            else:
                origin_preds = bbox_preds.new_zeros(1)
            whwh = torch.Tensor([[w, h, w, h]]).type_as(bbox_preds)
            
            return self.get_losses(
                bbox_preds,
                cls_preds,
                obj_preds,
                ang_preds,
                head_preds,
                origin_preds,
                org_xy_shifts,
                xy_shifts,
                expanded_strides,
                center_ltrbes,
                whwh,
                labels,
                dtype=xin[0].dtype,
            )
        else:
            # self.hw = [out.shape[-2:] for out in outputs]
            # # [batch, n_anchors_all, 85]
            # outputs = torch.cat(
            #     [out.flatten(start_dim=2) for out in outputs], dim=2
            # ).permute(0, 2, 1)
            # outputs = self.decode_outputs(outputs, dtype=x[0].type())
            # return (outputs, )

            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    # def get_output_and_grid(self, output, k, stride, dtype):
    #     grid = self.grids[k]

    #     batch_size = output.shape[0]
    #     n_ch = 5 + self.num_classes + self.num_angles + 4   # rotation headorder
    #     hsize, wsize = output.shape[-2:]
    #     if grid.shape[2:4] != output.shape[2:4]:
    #         yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
    #         grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
    #         self.grids[k] = grid

    #     output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
    #     output = output.permute(0, 1, 3, 4, 2).reshape(
    #         batch_size, self.n_anchors * hsize * wsize, -1
    #     )
    #     grid = grid.view(1, -1, 2)
    #     output[..., :2] = (output[..., :2] + grid) * stride
    #     output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
    #     return output, grid

    def get_output_and_grid(self, reg_box, hsize, wsize, k, stride, dtype):
        grid_size = self.grid_sizes[k]
        if (grid_size[0] != hsize) or (grid_size[1] != wsize) or (grid_size[2] != stride):
            grid_size[0] = hsize
            grid_size[1] = wsize
            grid_size[2] = stride

            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2).type(dtype).contiguous()  # [1, 1*hsize*wsize, 2]
            self.grids[k] = grid
            
            xy_shift = (grid + 0.5)*stride
            self.xy_shifts[k] = xy_shift
            expanded_stride = torch.full((1, grid.shape[1], 1), stride, dtype=grid.dtype, device=grid.device)
            self.expanded_strides[k] = expanded_stride
            center_radius = self.center_radius*expanded_stride
            center_radius = center_radius.expand_as(xy_shift)
            center_lt = center_radius + xy_shift
            center_rb = center_radius - xy_shift
            center_ltrb = torch.cat([center_lt, center_rb], dim=-1)
            self.center_ltrbes[k] = center_ltrb

        xy_shift = self.xy_shifts[k]
        grid = self.grids[k]
        expanded_stride = self.expanded_strides[k]
        center_ltrb = self.center_ltrbes[k]

        # l, t, r, b
        half_wh = torch.exp(reg_box[..., 2:4]) * (stride/2)  # (第k層)預測物體的半寬高
        reg_box[..., :2] = (reg_box[..., :2]+grid)*stride  # (第k層)預測物體的中心坐標
        reg_box[..., 2:4] = reg_box[..., :2] + half_wh  # (第k層)預測物體的右下坐標
        reg_box[..., :2] = reg_box[..., :2] - half_wh  # (第k層)預測物體的左上坐標

        return reg_box, grid, xy_shift, expanded_stride, center_ltrb

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        # imgs,
        # x_shifts,
        # y_shifts,
        # expanded_strides,
        # labels,
        # outputs,
        # origin_preds,
        # dtype,
        bbox_preds,
        cls_preds,
        obj_preds,
        ang_preds,
        head_preds,
        origin_preds,
        org_xy_shifts,
        xy_shifts,
        expanded_strides,
        center_ltrbes,
        whwh,
        labels,
        dtype,
    ):
        device = labels.device  # rotation
        # bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        # cls_preds = outputs[:, :, 5: 5+self.num_classes]  # [batch, n_anchors_all, n_cls]  # rotation
        # ang_preds = outputs[:, :, 5+self.num_classes: 5+self.num_classes+self.num_angles]  # [batch, n_anchors_all, n_ang]  # rotation
        # head_preds = outputs[:, :, 5+self.num_classes+self.num_angles:]  # [batch, n_anchors_all, n_head]  # headorder

        # # calculate targets 
        # nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        nlabel = labels[:, 0].long().bincount(minlength=cls_preds.shape[0]).tolist()

        # total_num_anchors = outputs.shape[1]
        # x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        # y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        # expanded_strides = torch.cat(expanded_strides, 1)
        # if self.use_l1:
        #     origin_preds = torch.cat(origin_preds, 1)

        # cls_targets = []
        # ang_targets = []  # rotation
        # head_targets = []  # headorder
        # reg_targets = []
        # l1_targets = []
        # obj_targets = []
        # fg_masks = []



        
        batch_gt_classes = labels[:, 1].type_as(cls_preds).contiguous()  # [num_gt, 1]
        
        batch_org_gt_bboxes = labels[:, 2:6].contiguous()  # [num_gt, 4]  bbox: cx, cy, w, h
        batch_org_gt_bboxes.mul_(whwh)
        batch_gt_bboxes = torch.empty_like(batch_org_gt_bboxes)  # [num_gt, 4]  bbox: l, t, r, b
        batch_gt_half_wh = batch_org_gt_bboxes[:, 2:]/2
        batch_gt_bboxes[:, :2] = batch_org_gt_bboxes[:, :2] - batch_gt_half_wh
        batch_gt_bboxes[:, 2:] = batch_org_gt_bboxes[:, :2] + batch_gt_half_wh
        batch_org_gt_bboxes = batch_org_gt_bboxes.type_as(bbox_preds)
        batch_gt_bboxes = batch_gt_bboxes.type_as(bbox_preds)
        del batch_gt_half_wh

        total_num_anchors = bbox_preds.shape[1]

        cls_targets = []
        ang_targets = []  # rotation
        head_targets = []  # headorder
        reg_targets = []
        l1_targets = []
        fg_mask_inds = []

        # num_fg = 0.0
        # num_gts = 0.0
        num_fg = 0.0
        num_gts = 0
        index_offset = 0
        batch_size = bbox_preds.shape[0]

        # for batch_idx in range(outputs.shape[0]):
        for batch_idx in range(batch_size):
            num_gt = int(nlabel[batch_idx])
            # num_gts += num_gt
            if num_gt == 0:
                # cls_target = outputs.new_zeros((0, self.num_classes))
                # ang_target = outputs.new_zeros((0, self.num_angles))  # rotation
                # head_target = outputs.new_zeros((0, 4))  # headorder
                # reg_target = outputs.new_zeros((0, 4))
                # l1_target = outputs.new_zeros((0, 4))
                # obj_target = outputs.new_zeros((total_num_anchors, 1))
                # fg_mask = outputs.new_zeros(total_num_anchors).bool()
                cls_target = bbox_preds.new_zeros((0, self.num_classes))
                
                ang_target = bbox_preds.new_zeros((0, self.num_angles))  # rotation
                head_target = bbox_preds.new_zeros((0, 4))  # headorder
                
                reg_target = bbox_preds.new_zeros((0, 4))
                l1_target = bbox_preds.new_zeros((0, 4))
            else:
                # gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                # gt_classes = labels[batch_idx, :num_gt, 0]
                # gt_angles = labels[batch_idx, :num_gt, 5]  # rotation
                # gt_heads = labels[batch_idx, :num_gt, 6]  # headorder
                # bboxes_preds_per_image = bbox_preds[batch_idx]

                _num_gts = num_gts + num_gt
                org_gt_bboxes_per_image = batch_org_gt_bboxes[num_gts:_num_gts]  # [num_gt, 4]  bbox: cx, cy, w, h
                gt_bboxes_per_image = batch_gt_bboxes[num_gts:_num_gts]  # [num_gt, 4]  bbox: l, t, r, b
                gt_classes = batch_gt_classes[num_gts:_num_gts]  # [num_gt]
                gt_angles = labels[batch_idx, :num_gt, 5]  # rotation
                gt_heads = labels[batch_idx, :num_gt, 6]  # headorder

                num_gts = _num_gts
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [n_anchors_all, 4]
                cls_preds_per_image = cls_preds[batch_idx]  # [n_anchors_all, n_cls]
                obj_preds_per_image = obj_preds[batch_idx]  # [n_anchors_all, 1]
                
                ang_preds_per_image = ang_preds[batch_idx]  # [n_anchors_all, n_cls]
                head_preds_per_image = head_preds[batch_idx]  # [n_anchors_all, n_cls]                
                try:
                    (
                        gt_matched_classes,
                        gt_matched_angles,  # rotation
                        gt_matched_heads,  # headorder
                        # fg_mask,
                        # pred_ious_this_matching,
                        # matched_gt_inds,
                        # num_fg_img,
                        fg_mask_ind,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        # batch_idx,
                        # num_gt,
                        # total_num_anchors,
                        # gt_bboxes_per_image,
                        # gt_classes,
                        # gt_angles,  # rotation
                        # gt_heads,  # headorder
                        # bboxes_preds_per_image,
                        # expanded_strides,
                        # x_shifts,
                        # y_shifts,
                        # cls_preds,
                        # ang_preds,  # rotation
                        # head_preds,  # headorder
                        # bbox_preds,
                        # obj_preds,
                        # labels,
                        # imgs,
                        num_gt,
                        total_num_anchors,
                        org_gt_bboxes_per_image,
                        gt_bboxes_per_image,
                        gt_classes,
                        self.num_classes,
                        bboxes_preds_per_image,
                        cls_preds_per_image,
                        obj_preds_per_image,
                        center_ltrbes,
                        xy_shifts,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    print("------------CPU Mode for This Batch-------------")
                    _org_gt_bboxes_per_image = org_gt_bboxes_per_image.cpu().float()
                    _gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
                    _bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
                    _cls_preds_per_image = cls_preds_per_image.cpu().float()
                    _obj_preds_per_image = obj_preds_per_image.cpu().float()
                    _gt_classes = gt_classes.cpu().float()
                    _center_ltrbes = center_ltrbes.cpu().float()
                    _xy_shifts = xy_shifts.cpu()
                    (
                        gt_matched_classes,
                        gt_matched_angles,  # rotation
                        gt_matched_heads,  # headorder
                        # fg_mask,
                        # pred_ious_this_matching,
                        # matched_gt_inds,
                        # num_fg_img,
                        fg_mask_ind,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        # batch_idx,
                        # num_gt,
                        # total_num_anchors,
                        # gt_bboxes_per_image,
                        # gt_classes,
                        # gt_angles,  # rotation
                        # gt_heads,  # headorder
                        # bboxes_preds_per_image,
                        # expanded_strides,
                        # x_shifts,
                        # y_shifts,
                        # cls_preds,
                        # ang_preds,  # rotation
                        # head_preds,  # headorder
                        # bbox_preds,
                        # obj_preds,
                        # labels,
                        # imgs,
                        # "cpu",
                        num_gt,
                        total_num_anchors,
                        _org_gt_bboxes_per_image,
                        _gt_bboxes_per_image,
                        _gt_classes,
                        self.num_classes,
                        _bboxes_preds_per_image,
                        _cls_preds_per_image,
                        _obj_preds_per_image,
                        _center_ltrbes,
                        _xy_shifts
                    )

                    gt_matched_classes = gt_matched_classes.cuda()
                    fg_mask_ind = fg_mask_ind.cuda()
                    pred_ious_this_matching = pred_ious_this_matching.cuda()
                    matched_gt_inds = matched_gt_inds.cuda()

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.view(-1, 1)  # [num_gt, num_classes]
                # ) * pred_ious_this_matching.unsqueeze(-1)
                ang_target = torch.from_numpy(angle_smooth_label(angle_label=gt_matched_angles.to(torch.int64).cpu().numpy(),
                                                                    num_angle_cls=self.num_angles,
                                                                    label_type=self.label_type,
                                                                    raduius=self.label_raduius)).to(device=device)  # rotation
                # gt_ang_per_image = angular_targets.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
                # import matplotlib.pyplot as plt
                # for index, value in enumerate(angular_targets):
                #     x = np.array(range(0, 180, 1))
                #     plt.plot(x, value, "r-", linewidth=1)
                #     plt.grid(True)
                #     plt.savefig('loss_img/{}.png'.format(gt_angles.cpu().int().numpy()[index]))
                #     plt.clf()
                head_target = F.one_hot(
                    gt_matched_heads.to(torch.int64), 4
                ) * pred_ious_this_matching.view(-1, 1)  # headorder
                # obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # [num_gt, 4]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        bbox_preds.new_empty((num_fg_img, 4)),
                        org_gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask_ind],
                        xy_shifts=org_xy_shifts[0][fg_mask_ind],
                    )
                    # l1_target = self.get_l1_target(
                    #     outputs.new_zeros((num_fg_img, 4)),
                    #     gt_bboxes_per_image[matched_gt_inds],
                    #     expanded_strides[0][fg_mask],
                    #     x_shifts=x_shifts[0][fg_mask],
                    #     y_shifts=y_shifts[0][fg_mask],
                    # )
            
            # cls_targets.append(cls_target)
            # ang_targets.append(ang_target)  # rotation
            # head_targets.append(head_target)  # headorder
            # reg_targets.append(reg_target)
            # obj_targets.append(obj_target.to(dtype))
            # fg_masks.append(fg_mask)
                if index_offset > 0:
                    fg_mask_ind.add_(index_offset)
                fg_mask_inds.append(fg_mask_ind)
            index_offset += total_num_anchors

            cls_targets.append(cls_target)  # [num_gt, num_classes]
            reg_targets.append(reg_target)  # [num_gt, 4]
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        # ang_targets = torch.cat(ang_targets, 0)  # rotation
        # head_targets = torch.cat(head_targets, 0)  # headorder
        reg_targets = torch.cat(reg_targets, 0)
        # obj_targets = torch.cat(obj_targets, 0)
        fg_mask_inds = torch.cat(fg_mask_inds, 0)
        # fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        # loss_iou = (
        #     self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        # ).sum() / num_fg
        # loss_obj = (
        #     self.obj_loss(obj_preds.view(-1, 1), obj_targets)
        # ).sum() / num_fg
        # loss_cls = (
        #     self.cls_loss(
        #         cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
        #     )
        # ).sum() / num_fg
        # loss_ang = (
        #     self.ang_loss(
        #         ang_preds.view(-1, self.num_angles)[fg_masks], ang_targets
        #     )
        # ).sum() / num_fg  # rotation
        # loss_head = (
        #     self.head_loss(
        #         head_preds.view(-1, 4)[fg_masks], head_targets
        #     )
        # ).sum() / num_fg  # headorder
        # if self.use_l1:
        #     loss_l1 = (
        #         self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
        #     ).sum() / num_fg
        # else:
        #     loss_l1 = 0.0

        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_mask_inds], reg_targets, True)
        ).sum() / num_fg
        obj_preds = obj_preds.view(-1, 1)
        obj_targets = torch.zeros_like(obj_preds).index_fill_(0, fg_mask_inds, 1)
        loss_obj = (
            self.bcewithlog_loss(obj_preds, obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_mask_inds], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_mask_inds], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = torch.zeros_like(loss_iou)

        reg_weight = 5.0
        loss_iou = reg_weight * loss_iou
        # loss = loss_iou + loss_obj + loss_cls + loss_ang + loss_head + loss_l1  # rotation headorder

        # return (
        #     loss,
        #     loss_iou,
        #     loss_obj,
        #     loss_cls,
        #     loss_ang,  # rotation
        #     loss_head,  # headorder
        #     loss_l1,
        #     num_fg / max(num_gts, 1),
        # )
        loss = loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    # def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
    #     l1_target[:, 0] = gt[:, 0] / stride - x_shifts
    #     l1_target[:, 1] = gt[:, 1] / stride - y_shifts
    #     l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
    #     l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
    #     return l1_target

    def get_l1_target(l1_target, gt, stride, xy_shifts, eps=1e-8):
        l1_target[:, 0:2] = gt[:, 0:2] / stride - xy_shifts
        l1_target[:, 2:4] = torch.log(gt[:, 2:4] / stride + eps)
        return l1_target


    @torch.no_grad()
    def get_assignments(
        self,
        # batch_idx,
        num_gt,
        total_num_anchors,
        org_gt_bboxes_per_image,  # [num_gt, 4]
        gt_bboxes_per_image,  # [num_gt, 4]
        gt_classes,  # [num_gt]
        num_classes,
        gt_angles,  # rotation
        gt_heads,  # headorder
        bboxes_preds_per_image,  # [n_anchors_all, 4]
        cls_preds_per_image,  # [n_anchors_all, n_cls]
        obj_preds_per_image,  # [n_anchors_all, 1]
        center_ltrbes,  # [1, n_anchors_all, 4]
        xy_shifts,  # [1, n_anchors_all, 2]
    ):

        # fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
        #     gt_bboxes_per_image,
        #     expanded_strides,
        #     x_shifts,
        #     y_shifts,
        #     total_num_anchors,
        #     num_gt,
        # )
        fg_mask_inds, is_in_boxes_and_center = self.get_in_boxes_info(
            org_gt_bboxes_per_image,  # [num_gt, 4]
            gt_bboxes_per_image,  # [num_gt, 4]
            center_ltrbes,  # [1, n_anchors_all, 4]
            xy_shifts,  # [1, n_anchors_all, 2]
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask_inds]  # [fg_count, 4]
        cls_preds_ = cls_preds_per_image[fg_mask_inds]  # [fg_count, num_classes]
        obj_preds_ = obj_preds_per_image[fg_mask_inds]  # [fg_count, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]  # num_in_boxes_anchor == fg_count

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, True, inplace=True)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # [num_gt, fg_count]

        cls_preds_ = cls_preds_.float().sigmoid_().unsqueeze(0).expand(num_gt, num_in_boxes_anchor, num_classes)
        obj_preds_ = obj_preds_.float().sigmoid_().unsqueeze(0).expand(num_gt, num_in_boxes_anchor, 1)
        cls_preds_ = (cls_preds_ * obj_preds_).sqrt_()  # [num_gt, fg_count, num_classes]

        del obj_preds_

        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), num_classes).float()  # [num_gt, num_classes]
        gt_cls_per_image = gt_cls_per_image[:, None, :].expand(num_gt, num_in_boxes_anchor, num_classes)

        with autocast(enabled=False):
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_, gt_cls_per_image, reduction="none").sum(-1)  # [num_gt, fg_count]
        del cls_preds_, gt_cls_per_image

        # 负例给非常大的cost（100000.0及以上）
        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )  # [num_gt, fg_count]
        del pair_wise_cls_loss, pair_wise_ious_loss, is_in_boxes_and_center

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds)
        del cost, pair_wise_ious

        return (
            gt_matched_classes,
            gt_angles[matched_gt_inds],  # rotation
            gt_heads[matched_gt_inds],  # headorder
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    # def get_in_boxes_info(
    #     self,
    #     gt_bboxes_per_image,
    #     expanded_strides,
    #     x_shifts,
    #     y_shifts,
    #     total_num_anchors,
    #     num_gt,
    # ):
    #     expanded_strides_per_image = expanded_strides[0]
    #     x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    #     y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    #     x_centers_per_image = (
    #         (x_shifts_per_image + 0.5 * expanded_strides_per_image)
    #         .unsqueeze(0)
    #         .repeat(num_gt, 1)
    #     )  # [n_anchor] -> [n_gt, n_anchor]
    #     y_centers_per_image = (
    #         (y_shifts_per_image + 0.5 * expanded_strides_per_image)
    #         .unsqueeze(0)
    #         .repeat(num_gt, 1)
    #     )

    #     gt_bboxes_per_image_l = (
    #         (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
    #         .unsqueeze(1)
    #         .repeat(1, total_num_anchors)
    #     )
    #     gt_bboxes_per_image_r = (
    #         (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
    #         .unsqueeze(1)
    #         .repeat(1, total_num_anchors)
    #     )
    #     gt_bboxes_per_image_t = (
    #         (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
    #         .unsqueeze(1)
    #         .repeat(1, total_num_anchors)
    #     )
    #     gt_bboxes_per_image_b = (
    #         (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
    #         .unsqueeze(1)
    #         .repeat(1, total_num_anchors)
    #     )

    #     b_l = x_centers_per_image - gt_bboxes_per_image_l
    #     b_r = gt_bboxes_per_image_r - x_centers_per_image
    #     b_t = y_centers_per_image - gt_bboxes_per_image_t
    #     b_b = gt_bboxes_per_image_b - y_centers_per_image
    #     bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    #     is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    #     is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    #     # in fixed center

    #     center_radius = 2.5

    #     gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
    #         1, total_num_anchors
    #     ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    #     gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
    #         1, total_num_anchors
    #     ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    #     gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
    #         1, total_num_anchors
    #     ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    #     gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
    #         1, total_num_anchors
    #     ) + center_radius * expanded_strides_per_image.unsqueeze(0)

    #     c_l = x_centers_per_image - gt_bboxes_per_image_l
    #     c_r = gt_bboxes_per_image_r - x_centers_per_image
    #     c_t = y_centers_per_image - gt_bboxes_per_image_t
    #     c_b = gt_bboxes_per_image_b - y_centers_per_image
    #     center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    #     is_in_centers = center_deltas.min(dim=-1).values > 0.0
    #     is_in_centers_all = is_in_centers.sum(dim=0) > 0

    #     # in boxes and in centers
    #     is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

    #     is_in_boxes_and_center = (
    #         is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    #     )
    #     return is_in_boxes_anchor, is_in_boxes_and_center

    def get_in_boxes_info(
            self,
            org_gt_bboxes_per_image,  # [num_gt, 4]
            gt_bboxes_per_image,  # [num_gt, 4]
            center_ltrbes,  # [1, n_anchors_all, 4]
            xy_shifts,  # [1, n_anchors_all, 2]
            total_num_anchors,
            num_gt,
    ):
        xy_centers_per_image = xy_shifts.expand(num_gt, total_num_anchors, 2)
        gt_bboxes_per_image = gt_bboxes_per_image[:, None, :].expand(num_gt, total_num_anchors, 4)

        b_lt = xy_centers_per_image - gt_bboxes_per_image[..., :2]
        b_rb = gt_bboxes_per_image[..., 2:] - xy_centers_per_image
        bbox_deltas = torch.cat([b_lt, b_rb], 2)  # [n_gt, n_anchor, 4]
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # [_n_gt, _n_anchor]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_ltrbes = center_ltrbes.expand(num_gt, total_num_anchors, 4)
        org_gt_xy_center = org_gt_bboxes_per_image[:, 0:2]
        org_gt_xy_center = torch.cat([-org_gt_xy_center, org_gt_xy_center], dim=-1)
        org_gt_xy_center = org_gt_xy_center[:, None, :].expand(num_gt, total_num_anchors, 4)
        center_deltas = org_gt_xy_center + center_ltrbes
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # [_n_gt, _n_anchor]
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # fg_mask [n_anchors_all]

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return torch.nonzero(is_in_boxes_anchor)[..., 0], is_in_boxes_and_center

    def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds):
        # Dynamic K
        # ---------------------------------------------------------------
        device = cost.device
        matching_matrix = torch.zeros(cost.shape, dtype=torch.uint8, device=device)  # [num_gt, fg_count]

        ious_in_boxes_matrix = pair_wise_ious  # [num_gt, fg_count]
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = topk_ious.sum(1).int().clamp_min_(1)
        if num_gt > 3:
            min_k, max_k = torch._aminmax(dynamic_ks)
            min_k, max_k = min_k.item(), max_k.item()
            if min_k != max_k:
                offsets = torch.arange(0, matching_matrix.shape[0] * matching_matrix.shape[1],
                                       step=matching_matrix.shape[1], dtype=torch.int, device=device)[:, None]
                masks = (torch.arange(0, max_k, dtype=dynamic_ks.dtype, device=device)[None, :].expand(num_gt, max_k) < dynamic_ks[:, None])
                _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
                pos_idxes.add_(offsets)
                pos_idxes = torch.masked_select(pos_idxes, masks)
                matching_matrix.view(-1).index_fill_(0, pos_idxes, 1)
                del topk_ious, dynamic_ks, pos_idxes, offsets, masks
            else:
                _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
                matching_matrix.scatter_(1, pos_idxes, 1)
                del topk_ious, dynamic_ks
        else:
            ks = dynamic_ks.tolist()
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
                matching_matrix[gt_idx][pos_idx] = 1
            del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        anchor_matching_one_more_gt_mask = anchor_matching_gt > 1

        anchor_matching_one_more_gt_inds = torch.nonzero(anchor_matching_one_more_gt_mask)
        if anchor_matching_one_more_gt_inds.shape[0] > 0:
            anchor_matching_one_more_gt_inds = anchor_matching_one_more_gt_inds[..., 0]
            # _, cost_argmin = torch.min(cost[:, anchor_matching_one_more_gt_inds], dim=0)
            _, cost_argmin = torch.min(cost.index_select(1, anchor_matching_one_more_gt_inds), dim=0)
            # matching_matrix[:, anchor_matching_one_more_gt_inds] = 0
            matching_matrix.index_fill_(1, anchor_matching_one_more_gt_inds, 0)
            matching_matrix[cost_argmin, anchor_matching_one_more_gt_inds] = 1
            # fg_mask_inboxes = matching_matrix.sum(0) > 0
            fg_mask_inboxes = matching_matrix.any(dim=0)
            fg_mask_inboxes_inds = torch.nonzero(fg_mask_inboxes)[..., 0]
        else:
            fg_mask_inboxes_inds = torch.nonzero(anchor_matching_gt)[..., 0]
        num_fg = fg_mask_inboxes_inds.shape[0]

        matched_gt_inds = matching_matrix.index_select(1, fg_mask_inboxes_inds).argmax(0)
        fg_mask_inds = fg_mask_inds[fg_mask_inboxes_inds]
        gt_matched_classes = gt_classes[matched_gt_inds]

        # pred_ious_this_matching = pair_wise_ious[:, fg_mask_inboxes_inds][matched_gt_inds, torch.arange(0, matched_gt_inds.shape[0])]  # [matched_gt_inds_count]
        pred_ious_this_matching = pair_wise_ious.index_select(1, fg_mask_inboxes_inds).gather(dim=0, index=matched_gt_inds[None, :])  # [1, matched_gt_inds_count]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask_inds
    # def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    #     # Dynamic K
    #     # ---------------------------------------------------------------
    #     matching_matrix = torch.zeros_like(cost)

    #     ious_in_boxes_matrix = pair_wise_ious
    #     n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
    #     topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
    #     dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    #     for gt_idx in range(num_gt):
    #         _, pos_idx = torch.topk(
    #             cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
    #         )
    #         matching_matrix[gt_idx][pos_idx] = 1.0

    #     del topk_ious, dynamic_ks, pos_idx

    #     anchor_matching_gt = matching_matrix.sum(0)
    #     if (anchor_matching_gt > 1).sum() > 0:
    #         _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
    #         matching_matrix[:, anchor_matching_gt > 1] *= 0.0
    #         matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
    #     fg_mask_inboxes = matching_matrix.sum(0) > 0.0
    #     num_fg = fg_mask_inboxes.sum().item()

    #     fg_mask[fg_mask.clone()] = fg_mask_inboxes

    #     matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    #     gt_matched_classes = gt_classes[matched_gt_inds]

    #     pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
    #         fg_mask_inboxes
    #     ]
    #     return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds