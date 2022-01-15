#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np
from numpy.core.numeric import empty_like

from yolox.utils import adjust_box_anns, get_local_rank, adjust_box_anns_without_clip

from ..data_augment import box_candidates
from ..data_augment_headorder import random_perspective_with_angles
from .datasets_wrapper import Dataset

from yolox.utils.utils import drawRotationbox, longsideformat2cvminAreaRect, debugDrawBox, cvminAreaRect2longsideformat
from yolox.utils.utils import checkAngleRange, findNewOrder, countAngle, distPoints, findHeadPoint
from yolox.utils.boxes import xyxy2cxcywh, cxcywh2xyxy
import math

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicHeadOrderDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0,
        enable_flip=True, flip_prob=0.5, enable_rotate=True,
        rotate_prob=0.5, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()

        self.enable_flip = enable_flip
        self.flip_prob = flip_prob
        self.enable_rotate = enable_rotate
        self.rotate_prob = rotate_prob
    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
                # mosaic_img.size = [resized_height,resized_ width, 3]
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]  # mosaic_img[ymin:ymax, xmin:xmax]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1  # 原圖片未剪裁進mosaic_img中的寬高度

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format 歸一化的xywh轉為非歸一化的xyxy(左上右下)坐標形式
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw  # Left_top_x
                    labels[:, 1] = scale * _labels[:, 1] + padh  # Left_top_y
                    labels[:, 2] = scale * _labels[:, 2] + padw  # right_bottom_x
                    labels[:, 3] = scale * _labels[:, 3] + padh  # right_bottom_y
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                # np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])  #xmin
                # np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])  #ymin
                # np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])  #xmax
                # np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])  #ymax
            # randomNum = random.randint(0, 1000)
            
            # debugDrawBox(mosaic_img.copy(), xyxy2cxcywh(mosaic_labels.copy()), isDraw=True, num='{}_1'.format(randomNum))

            # 隨機對圖片進行平移，縮放，裁剪
            mosaic_img, mosaic_labels = random_perspective_with_angles(
                mosaic_img,
                mosaic_labels,
                degrees=0,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove 要移除的邊框

            if len(mosaic_labels) != 0:
                mosaic_img, mosaic_labels = self.checkLabels(origin_img=mosaic_img, origin_labels=mosaic_labels)

            # debugDrawBox(mosaic_img.copy(), mosaic_labels, isDraw=True, num='{}_2'.format(randomNum))

            # 上下左右翻轉
            if self.enable_flip and not len(mosaic_labels) == 0 and random.random() < self.flip_prob:
                mosaic_img, mosaic_labels = self.flipup(mosaic_img, mosaic_labels)
            
            # debugDrawBox(mosaic_img.copy(), mosaic_labels, isDraw=True, num='{}_3'.format(randomNum))


            # 旋轉
            if self.enable_rotate and not len(mosaic_labels) == 0 and random.random() < self.rotate_prob:
                mosaic_img, mosaic_labels = self.rotate(mosaic_img, mosaic_labels, self.degrees, isdebug=False)

            # debugDrawBox(mosaic_img.copy(), mosaic_labels, isDraw=True, num='{}_4'.format(randomNum))

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)

            # debugDrawBox(mosaic_img.copy(), mosaic_labels, isDraw=True)

            mosaic_labels = cxcywh2xyxy(mosaic_labels)
            
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def checkLabels(self, origin_img, origin_labels, isdebug=False):
        height, width, _ = origin_img.shape
        labels = xyxy2cxcywh(origin_labels.copy())
        labels = labels[(labels[:, 0] > 0) & (labels[:, 0] < width)]
        labels = labels[(labels[:, 1] > 0) & (labels[:, 1] < height)]
        # labels = cxcywh2xyxy(labels)

        return origin_img, labels

    def rotate(self, origin_img, origin_labels, degrees=45, scale=1, isdebug=False):
        height, width, _ = origin_img.shape
        a, b = height / 2, width / 2
        target_angle = random.randint(-degrees, degrees)
        
        if isdebug:
            target_angle = 45
            unrotate = origin_img.copy()
            # source_labels = xyxy2cxcywh(origin_labels.copy())
            source_labels = origin_labels.copy()
            unrotate = drawRotationbox(img=unrotate, bboxes=source_labels[:, :4], angles=source_labels[:, 5])
        
        M = cv2.getRotationMatrix2D(center=(a, b), angle=target_angle, scale=scale)
        rotated_img = cv2.warpAffine(origin_img.copy(), M, (height, width))  # 旋轉後的圖像保持大小不變

        Pi_angle = -target_angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
        
        if isdebug:
            step1img = origin_img.copy()
            step2img = rotated_img.copy()

        rotated_labels = []
        # labels = xyxy2cxcywh(origin_labels.copy())
        labels = origin_labels.copy()
        for label in labels:
            order = int(label[6])
            rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
            poly = cv2.boxPoints(rect)
            
            if isdebug:
                cv2.circle(step1img, ((int(poly[0][0])), (int(poly[0][1]))), 4, (0, 0, 255), -1)
                cv2.circle(step1img, ((int(poly[1][0])), (int(poly[1][1]))), 4, (0, 0, 255), -1)
                cv2.circle(step1img, ((int(poly[2][0])), (int(poly[2][1]))), 4, (0, 0, 255), -1)
                cv2.circle(step1img, ((int(poly[3][0])), (int(poly[3][1]))), 4, (0, 0, 255), -1)
            
            # 下面是计算旋转后目标相对旋转过后的图像的位置
            X0 = (poly[order][0] - a) * math.cos(Pi_angle) - (poly[order][1] - b) * math.sin(Pi_angle) + a
            Y0 = (poly[order][0] - a) * math.sin(Pi_angle) + (poly[order][1] - b) * math.cos(Pi_angle) + b

            X1 = (poly[order-1][0] - a) * math.cos(Pi_angle) - (poly[order-1][1] - b) * math.sin(Pi_angle) + a
            Y1 = (poly[order-1][0] - a) * math.sin(Pi_angle) + (poly[order-1][1] - b) * math.cos(Pi_angle) + b

            X2 = (poly[order-2][0] - a) * math.cos(Pi_angle) - (poly[order-2][1] - b) * math.sin(Pi_angle) + a
            Y2 = (poly[order-2][0] - a) * math.sin(Pi_angle) + (poly[order-2][1] - b) * math.cos(Pi_angle) + b

            X3 = (poly[order-3][0] - a) * math.cos(Pi_angle) - (poly[order-3][1] - b) * math.sin(Pi_angle) + a
            Y3 = (poly[order-3][0] - a) * math.sin(Pi_angle) + (poly[order-3][1] - b) * math.cos(Pi_angle) + b

            # get rotated x, y, w, h
            poly_rotated = np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])

            if isdebug:
                cv2.circle(step2img, ((int(poly_rotated[0][0])), (int(poly_rotated[0][1]))), 4, (0, 0, 255), -1)
                cv2.circle(step2img, ((int(poly_rotated[1][0])), (int(poly_rotated[1][1]))), 4, (0, 0, 255), -1)
                cv2.circle(step2img, ((int(poly_rotated[2][0])), (int(poly_rotated[2][1]))), 4, (0, 0, 255), -1)
                cv2.circle(step2img, ((int(poly_rotated[3][0])), (int(poly_rotated[3][1]))), 4, (0, 0, 255), -1)
            
            rect_rotated = cv2.minAreaRect(np.float32(poly_rotated))
            if rect_rotated[0][0] < 0 or rect_rotated[0][0] > width or rect_rotated[0][1] < 0 or rect_rotated[0][1] > height:
                continue
            label[:4] = rect_rotated[0][0], rect_rotated[0][1], max(rect_rotated[1]), min(rect_rotated[1])
            
            # if class is storage-tank or roundabout
            if int(label[4]) == 9 or int(label[4]) == 11 or int(label[4]) == 16:
                pass
            else:
                # get right angle
                point1, point2 = distPoints([X0, Y0], [X1, Y1], [X2, Y2])
                label[5] = checkAngleRange(round(countAngle([point1[0]+5, point1[1]], point1, point2))) # angle range [0~179]
                
                # get new order
                rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
                poly = cv2.boxPoints(rect)
                label[6] = findNewOrder(poly, poly_rotated[0])

            if isdebug:
                poly = np.int0(poly)
                # cv2.circle(rotated_img, (int(poly[int(label[6])][0]), int(poly[int(label[6])][1])), 3, (0, 0, 255), -1)
                rotated_img = cv2.drawContours(image=rotated_img.copy(), contours=[poly], contourIdx=-1, color=(0, 255, 0))
            
            rotated_labels.append(label)
        
        if len(rotated_labels) != 0:
            labels = np.array(rotated_labels)
        else:
            labels = np.empty((0, 7), dtype=origin_labels.dtype)

        if isdebug:
            randomNum = random.randint(0, 1000)
            cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_unrotate_img.jpg'.format(randomNum), unrotate)
            cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_step1img.jpg'.format(randomNum), step1img)
            cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_step2img.jpg'.format(randomNum), step2img)
            cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_rotated_img.jpg'.format(randomNum), rotated_img)
            # cv2.imshow('origin_img', unrotate)
            # cv2.imshow('rotated_img', rotated_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return rotated_img, labels

    def flipup(self, origin_img, origin_labels, isdebug=False):
        weight, height, _ = origin_img.shape

        # src_labels = xyxy2cxcywh(origin_labels.copy())

        flip_labels = []
        if isdebug:
            randomNum = random.randint(0, 1000)
            unflip = origin_img.copy()
        #     source_labels = xyxy2cxcywh(origin_labels.copy())
        #     source_labels = origin_labels.copy()
        #     unflip = drawRotationbox(img=unflip, bboxes=source_labels[:, :4], angles=source_labels[:, 5], heads=source_labels[:, 6])
        isFlipLR = random.uniform(0, 1) > 0.5
        isDraw = False

        if isFlipLR:
            flip_img = np.fliplr(origin_img.copy())
        else:
            flip_img = np.flipud(origin_img.copy())
        flip_img_ = flip_img.copy()
        for label in origin_labels:
            # 未翻轉
            rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
            # rect = [(x, y), (w, h), theta]
            poly = cv2.boxPoints(rect)
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            polyShift = np.array([[poly[int(label[6])][0], poly[int(label[6])][1]], 
                                [poly[int(label[6])-1][0], poly[int(label[6])-1][1]], 
                                [poly[int(label[6])-2][0], poly[int(label[6])-2][1]], 
                                [poly[int(label[6])-3][0], poly[int(label[6])-3][1]]])
            
            

            if isDraw:
                polyShiftInt = np.int0(polyShift)
                if float(label[1]) > polyShift[0][1]:
                    # red color
                    cv2.circle(unflip, (polyShiftInt[0][0], polyShiftInt[0][1]), 4, (0, 0, 255), -1)
                else:
                    # blue color
                    cv2.circle(unflip, (polyShiftInt[0][0], polyShiftInt[0][1]), 4, (255, 0, 0), -1)
                cv2.circle(unflip, (polyShiftInt[1][0], polyShiftInt[1][1]), 1, (0, 255, 0), -1)
                cv2.circle(unflip, (polyShiftInt[2][0], polyShiftInt[2][1]), 1, (0, 255, 0), -1)
                cv2.circle(unflip, (polyShiftInt[3][0], polyShiftInt[3][1]), 1, (0, 255, 0), -1)


            # fliplr 左右翻轉
            if isFlipLR:
                polyShift[0][0] = weight - polyShift[0][0]
                polyShift[1][0] = weight - polyShift[1][0]
                polyShift[2][0] = weight - polyShift[2][0]
                polyShift[3][0] = weight - polyShift[3][0]
            else:
                polyShift[0][1] = height - polyShift[0][1]
                polyShift[1][1] = height - polyShift[1][1]
                polyShift[2][1] = height - polyShift[2][1]
                polyShift[3][1] = height - polyShift[3][1]

            # 校正起點
            polyShift = np.array([[polyShift[0-1][0], polyShift[0-1][1]], 
                                [polyShift[1-1][0], polyShift[1-1][1]], 
                                [polyShift[2-1][0], polyShift[2-1][1]], 
                                [polyShift[3-1][0], polyShift[3-1][1]]])
            head = polyShift[0]
            
            if isDraw:
                polyShiftInt = np.int0(polyShift)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[0][0], polyShiftInt[0][1]), 3, (0, 0, 255), -1)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[1][0], polyShiftInt[1][1]), 1, (0, 255, 0), -1)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[2][0], polyShiftInt[2][1]), 2, (0, 255, 0), -1)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[3][0], polyShiftInt[3][1]), 2, (0, 255, 0), -1)

            rect = cv2.minAreaRect(polyShift)
            _, _, _, _, theta = cvminAreaRect2longsideformat(x_c=rect[0][0], y_c=rect[0][1], width=rect[1][0], height=rect[1][1], theta=rect[-1])
            if theta == 180:
                theta = 0
            if isFlipLR:
                rect = longsideformat2cvminAreaRect(weight - label[0], label[1], label[2], label[3], (theta - 179.9))
            else:
                rect = longsideformat2cvminAreaRect(label[0], height - label[1], label[2], label[3], (theta - 179.9))
            # rect = [(x, y), (w, h), theta]
            poly = cv2.boxPoints(rect)
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            ans, indexHead = findHeadPoint(headpoint=head, points=poly.tolist())




            if isDraw:
                polyShiftInt = np.int0(poly)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[0][0], polyShiftInt[0][1]), 3, (0, 0, 255), -1)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[1][0], polyShiftInt[1][1]), 1, (0, 255, 0), -1)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[2][0], polyShiftInt[2][1]), 2, (0, 255, 0), -1)
                flip_img = cv2.circle(flip_img.copy(), (polyShiftInt[3][0], polyShiftInt[3][1]), 2, (0, 255, 0), -1)
                flip_img_ = cv2.drawContours(image=flip_img_, contours=[polyShiftInt], contourIdx=-1, color=(0, 255, 0), thickness=1)
                flip_img_ = cv2.circle(flip_img_, (polyShiftInt[indexHead][0], polyShiftInt[indexHead][1]), 3, (0, 0, 255), -1)
            
            label[0] = rect[0][0]
            label[1] = rect[0][1]
            label[2] = rect[1][0]
            label[3] = rect[1][1]

            if int(label[4]) == 9 or int(label[4]) == 11 or int(label[4]) == 16:
                pass
            else:
                label[5] = theta
                label[6] = indexHead

            flip_labels.append(label)
        
        if isdebug:
            # cv2.imwrite('/home/danny/Lab/yolox_test/img_test/{}_mosaic_img.jpg'.format(randomNum), unflip)
            cv2.imwrite('/home/danny/Lab/yolox_test/img_test/{}_mosaic_img.jpg'.format(randomNum), flip_img_)
            # cv2.imshow('origin_img', unflip)
            # cv2.imshow('leftright', flip_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        if len(flip_labels) != 0:
            origin_labels = np.array(flip_labels)
        else:
            origin_labels = np.empty((0, 7), dtype=origin_labels.dtype)
        
        return flip_img, origin_labels

    def mixup(self, img1, origin_labels, input_dim):
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img2, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        # print(img1.shape)
        # print(img2.shape)
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        # print(r)
        mixup_img = (img1 * r + img2 * (1 - r))
        cp_labels = xyxy2cxcywh(cp_labels.copy())
        mixup_labels = np.concatenate((origin_labels, cp_labels), 0)
        # print(origin_labels.shape)
        # print(cp_labels.shape)
        # print(mixup_labels.shape)
        # print(mixup_img.shape)


        return mixup_img.astype(np.uint8), mixup_labels

    def mixup2(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        # FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, img_info, _ = self._dataset.pull_item(cp_index)
        
        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        # cp_scale_ratio = 0.5
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        # if FLIP:
        #     cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]
        print(cp_scale_ratio)
        cp_bboxes_origin_np = adjust_box_anns(
            bbox=cp_labels[:, :4].copy(), scale_ratio=cp_scale_ratio, padw=0, padh=0, w_max=origin_w, h_max=origin_h
        )
        # if FLIP:
        #     cp_bboxes_origin_np[:, 0::2] = (
        #         origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
        #     )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)
        cv2.imwrite('/home/danny/Lab/yolox_test/img_test/{}_flip_img.jpg'.format(img_info), padded_cropped_img)
        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            ang_labels = cp_labels[keep_list, 5:6].copy()  # rotation
            head_labels = cp_labels[keep_list, 6:7].copy()  # headorder
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels, ang_labels, head_labels))  # rotation
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
