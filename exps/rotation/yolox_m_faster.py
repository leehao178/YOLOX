#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
import torch.nn as nn

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        print(self.exp_name)

        self.data_num_workers = 2
        self.num_classes = 8
        self.data_dir = "/home/danny/DataSet/car8_head_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.print_interval = 1
        self.save_ckpt_interval = 999


    def get_model(self):
        from yolox.models import YOLOXHeadLess, YOLOPAFPN, YOLOXFasterHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXFasterHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOXHeadLess(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model