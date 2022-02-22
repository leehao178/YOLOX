#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection

# rotation
from .coco_head_less import COCOHeadLessDataset
from .coco_head_order import COCOHeadOrderDataset
from .mosaicdetection_headless import MosaicHeadLessDetection
from .mosaicdetection_headorder import MosaicHeadOrderDetection

from .classes import DOTA_10_CLASSES, DOTA_15_CLASSES, DOTA_20_CLASSES, HRSC2016_CLASSES
