#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis, postprocess_rotation_head, vis_rotation

from yolox.utils import xyxy2cxcywh, longsideformat2cvminAreaRect

from yolox.evaluators import evaluation ,mergebypoly, evaluation_trans, draw_DOTA_image, multi_classes_nms


import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--output", default="/home/danny/Lab/yolox_test/test_dota_md", help="save path to images or video"
    )
    parser.add_argument(
        "--output_path", help="save path to images or video"
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--classes", default="dota10", type=str, help="dataset name")
    parser.add_argument("--conf", default=0.05, type=float, help="test conf")
    parser.add_argument("--nms", default=0.1, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=800, type=int, help="test img size")
    parser.add_argument("--mode", type=str, help="mode")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=None,
        device="cpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.num_angles = exp.num_angles
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)

            outputs = postprocess_rotation_head(
                outputs, self.num_classes, self.num_angles, self.confthre, self.nmsthre, isnms=False
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def savetxt(self, output, img_info, img_name, out_path, cls_conf=0.35):
        ratio = img_info["ratio"]
        if output is not None:
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            bboxes /= ratio

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            angles = output[:, 7]

            bboxes = xyxy2cxcywh(bboxes)
            
            for i in range(len(bboxes)):
                box = bboxes[i]
                cls_id = int(cls[i])
                angle = int(angles[i])
                score = scores[i]
                if score < cls_conf:
                    continue

                # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
                rect = longsideformat2cvminAreaRect(box[0], box[1], box[2], box[3], (angle - 179.9))
                # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                poly = np.float32(cv2.boxPoints(rect))
                poly = np.int0(poly)

                splitname = img_name.split('__')  # 分割待merge的圖像的名稱 eg:['P0706','1','0','_0']
                oriname = splitname[0]  # 獲得待merge圖像的原圖像名稱 eg:P706

                # 目標所屬圖片名稱_分割id 置信度 poly classname
                lines = '{} {} {} {} {} {} {} {} {} {} {}'.format(  img_name, 
                                                                    score.item(), 
                                                                    poly[0][0],
                                                                    poly[0][1],
                                                                    poly[1][0],
                                                                    poly[1][1],
                                                                    poly[2][0],
                                                                    poly[2][1],
                                                                    poly[3][0],
                                                                    poly[3][1],
                                                                    self.cls_names[cls_id])
                # 移除之前的輸出文件夾,並新建輸出文件夾
                if not os.path.exists(out_path):
                    os.makedirs(out_path)  # make new output folder

                with open(str(out_path + '/' + oriname) + '.txt', 'a') as f:
                    f.writelines(lines + '\n')


def image_demo(predictor, path, output_path):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        predictor.savetxt(outputs[0], img_info, os.path.basename(image_name), output_path, predictor.confthre)


def mkfolder(root):
    # result/epoch_30_ckpt/detection
    detection = os.path.join(root, "detection")
    os.makedirs(detection, exist_ok=True)

    # result/epoch_30_ckpt/detection/merged_drawed
    os.makedirs(os.path.join(detection, "merged_drawed"), exist_ok=True)

    # result/epoch_30_ckpt/detection/result_txt
    os.makedirs(os.path.join(detection, "result_txt"), exist_ok=True)

    # result/epoch_30_ckpt/detection/result_txt/result_before_merge
    before_merge = os.path.join(detection, "result_txt", "result_before_merge")
    os.makedirs(before_merge, exist_ok=True)

    # result/epoch_30_ckpt/detection/result_txt/result_classname
    result_classname = os.path.join(detection, "result_txt", "result_classname")
    os.makedirs(result_classname, exist_ok=True)

    # result/epoch_30_ckpt/detection/result_txt/result_merged
    after_merge = os.path.join(detection, "result_txt", "result_merged")
    os.makedirs(after_merge, exist_ok=True)

    return detection, before_merge, result_classname, after_merge


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    res_folder = os.path.join(file_name, "result_riou")
    os.makedirs(res_folder, exist_ok=True)

    if args.classes == 'dota10':
        from yolox.data.datasets import DOTA_10_CLASSES as CLASSES
    elif args.classes == 'dota15':
        from yolox.data.datasets import DOTA_15_CLASSES as CLASSES
    elif args.classes == 'dota20':
        from yolox.data.datasets import DOTA_20_CLASSES as CLASSES
    elif args.classes == 'hrsc2016':
        from yolox.data.datasets import HRSC2016_CLASSES as CLASSES
    elif args.classes == 'car':
        from yolox.data.datasets import CAR_CLASSES as CLASSES
    elif args.classes == 'car8':
        from yolox.data.datasets import CAR8_CLASSES as CLASSES
    else:
        logger.info('未選擇類別')

    logger.info("Args: {}".format(args))
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    if args.mode == 'val':
        print(os.listdir(file_name))
        for ckpts in os.listdir(file_name):
            # ckpts = epoch_30_ckpt.pth
            if ckpts.startswith('epoch_'):
                logger.info(ckpts)
                ckpt_folder = os.path.join(res_folder, ckpts.split('.')[0])
                os.makedirs(ckpt_folder, exist_ok=True)
                
                detection, before_merge, result_classname, after_merge = mkfolder(ckpt_folder)

                model = exp.get_model()
                logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

                if args.device == "gpu":
                    model.cuda()
                model.eval()

                logger.info("loading checkpoint")
                ckpt = torch.load(os.path.join(file_name, ckpts), map_location="cpu")
                # load the model state dict
                model.load_state_dict(ckpt["model"])
                logger.info("loaded checkpoint done.")

                if args.fuse:
                    logger.info("\tFusing model...")
                    model = fuse_model(model)
                
                predictor = Predictor(model, exp, CLASSES, args.device)
                image_demo(predictor, args.path, before_merge)
                classap, classap75 = evaluation(
                    detoutput=detection,
                    imageset=r'/home/aimlusr/dataset/dota10/val/images', # 原始未裁切圖片
                    annopath=r'/home/aimlusr/dataset/dota10/val/labelTxt/{:s}.txt',
                    classnames=CLASSES,
                    isnotmerge=True,
                    iscountmAP=True,
                    ismulti_cls_nms=True
                )

                with open(os.path.join(res_folder, 'classap.txt'), 'a') as f:
                    f.write('\n\n{}'.format(ckpts.split('.')[0]))
                    for i in classap:
                        f.write('\n{}'.format(i))
                with open(os.path.join(res_folder, 'classap75.txt'), 'a') as f:
                    f.write('\n\n{}'.format(ckpts.split('.')[0]))
                    for i in classap75:
                        f.write('\n{}'.format(i))
    elif args.mode == 'test':

        test_folder = os.path.join(res_folder, 'test')
        os.makedirs(test_folder, exist_ok=True)
        
        detection, before_merge, result_classname, after_merge = mkfolder(test_folder)

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if args.device == "gpu":
            model.cuda()
        model.eval()

        logger.info("loading checkpoint")
        ckpt = torch.load(os.path.join(file_name, 'epoch_300_ckpt.pth'), map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)
        
        predictor = Predictor(model, exp, CLASSES, args.device)
        image_demo(predictor, args.path, before_merge)

        # see demo for example
        mergebypoly(
            before_merge,
            after_merge,
            ismulti_cls_nms=True
        )
        logger.info("result merge done.")

        evaluation_trans(
            after_merge,
            result_classname
        )
        logger.info("result classify done.")

        multi_classes_nms(result_classname)
        logger.info("檢測分類結果已NMS完成")


if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.mode == 'test':
        args.path = '/home/aimlusr/dataset/dota10/test_split'
    elif args.mode == 'val':
        args.path = '/home/aimlusr/dataset/dota10/val_split/images'

    exp = get_exp(args.exp_file, args.name)

    main(exp, args)