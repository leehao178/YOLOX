#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco_evaluator import COCOEvaluator
from .voc_evaluator import VOCEvaluator

from .evaluation import evaluation, voc_eval, evaluation2
from .evaluation_utils import mergebypoly, evaluation_trans, draw_DOTA_image, multi_classes_nms