#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2021 CYCU Lab602 Hao Lee All rights reserved.

import os


import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 15
        self.depth = 0.67
        self.width = 0.75
        self.act = 'silu'
        self.iou_loss = "iou"
        self.obj_loss = "bce"
        self.cls_loss = "bce"
        self.ang_loss = "focalloss"
        self.head_loss = "bce"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (800, 800)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 0
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = "/home/danny/DataSet/dota_head_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = False

        # --------------- data augment config --------------- #
        self.enable_flip=False
        self.flip_prob=0.5
        self.enable_rotate=False
        self.rotate_prob=0.5
        self.degrees = 45

        epoch_scale = 2
        # --------------  training config --------------------- #
        self.warmup_epochs = 5 * epoch_scale
        self.max_epoch = 300 * epoch_scale
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15 * epoch_scale
        self.min_lr_ratio = 0.05
        self.ema = True
        self.optimize = 'sgd'
        # self.optimize = 'adam'
        self.weight_decay = 5e-4
        self.momentum = 0.9
        # self.momentum = 0.937
        self.print_interval = 1
        self.eval_interval = 300
        self.save_ckpt_interval = 30
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # --------------  angles config --------------------- #
        self.label_type = 0
        self.label_raduius = 6
        self.num_angles = 180

        # -----------------  testing config ------------------ #
        self.test_size = (800, 800)
        self.test_conf = 0.05
        self.nmsthre = 0.1
    
    def get_model(self):
        from yolox.models import YOLOXHeadOrder, YOLOPAFPN, YOLOXRotateHeadOrderHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXRotateHeadOrderHead(num_classes=self.num_classes, 
                                            num_angles=self.num_angles, 
                                            iou_loss=self.iou_loss,
                                            obj_loss=self.obj_loss,
                                            cls_loss=self.cls_loss,
                                            ang_loss=self.ang_loss,
                                            head_loss=self.head_loss,
                                            label_type=self.label_type, 
                                            label_raduius=self.label_raduius, 
                                            width=self.width, 
                                            in_channels=in_channels)
            self.model = YOLOXHeadOrder(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            COCOHeadOrderDataset,
            TrainTransformHeadOrder,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicHeadOrderDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCOHeadOrderDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="train2017",
                img_size=self.input_size,
                preproc=TrainTransformHeadOrder(
                    max_labels=9999,
                    flip_prob=0.0,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicHeadOrderDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformHeadOrder(
                max_labels=9999,
                flip_prob=0.0,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            enable_flip=self.enable_flip,
            flip_prob=self.flip_prob,
            enable_rotate=self.enable_rotate,
            rotate_prob=self.rotate_prob
        )
        
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            if self.optimize == 'sgd':
                optimizer = torch.optim.SGD(
                    pg0, lr=lr, momentum=self.momentum, nesterov=True
                )
            elif self.optimize == 'adam':
                optimizer = torch.optim.Adam(
                    pg0, lr=lr, betas=(self.momentum, 0.999)
                )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCOHeadOrderDataset, ValTransformHeadOrder

        valdataset = COCOHeadOrderDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "instances_dev2017.json",
            name="val2017" if not testdev else "dev2017",
            img_size=self.test_size,
            preproc=ValTransformHeadOrder(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
    