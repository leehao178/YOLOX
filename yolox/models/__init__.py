#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .darknet import CSPDarknet, Darknet
from .swin_transformer import SwinTransformer
from .losses import IOUloss, FocalLoss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .swin_pafpn import SWIMPAFPN
from .yolox import YOLOX

# YOLOX Head
from .yolo_head_rotate_less import YOLOXRotateHeadLessHead
from .yolo_head_rotate_order import YOLOXRotateHeadOrderHead
from .yolo_head_rotate_order_hydra import YOLOXRotateHeadOrderHydraHead
from .yolo_head_faster import YOLOXFasterHead

# YOLOX
from .yolox_head_less import YOLOXHeadLess
from .yolox_head_order import YOLOXHeadOrder
