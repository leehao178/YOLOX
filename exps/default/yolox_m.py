#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_num_workers = 2
        self.num_classes = 8
        self.data_dir = "/home/danny/DataSet/car8_head_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
