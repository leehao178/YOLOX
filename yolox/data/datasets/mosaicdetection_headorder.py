#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank, adjust_box_anns_without_clip

from ..data_augment import box_candidates
from ..data_augment_headorder import random_perspective_with_angles
from .datasets_wrapper import Dataset

from yolox.utils.utils import drawRotationbox, longsideformat2cvminAreaRect
from yolox.utils.utils import checkAngleRange, findNewOrder, countAngle, distPoints
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

            # src_img = mosaic_img.copy()
            # for label in mosaic_labels.copy():
            #     cv2.rectangle(src_img, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), (0, 255, 0), 2)

            # src_labels = xyxy2cxcywh(mosaic_labels.copy())
            # src_img = drawRotationbox(img=src_img.copy(), bboxes=src_labels[:, :4], angles=src_labels[:, 5], heads=src_labels[:, 6])
            # cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_mosaic_img.jpg'.format(img_id), src_img)
            

            
            # 上下左右翻轉
            if self.enable_flip and random.random() < self.flip_prob:
                flip_img, flip_labels = self.flipup(mosaic_img, mosaic_labels)
                if len(flip_labels) != 0:
                    mosaic_img = flip_img
                    mosaic_labels = flip_labels
                # else:
                #     print('flip_labels = 0')
                    # target_labels = xyxy2cxcywh(mosaic_labels.copy())
                    # flip_img = drawRotationbox(img=flip_img.copy(), bboxes=target_labels[:, :4], angles=target_labels[:, 5], heads=target_labels[:, 6])
            
                    # cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_flip_img.jpg'.format(img_id), flip_img)
            # 旋轉
            if self.enable_rotate and random.random() < self.rotate_prob:
                rotated_img, rotated_labels = self.rotate(mosaic_img, mosaic_labels, self.degrees)
                if len(rotated_labels) != 0:
                    mosaic_img = rotated_img
                    mosaic_labels = rotated_labels
                # else:
                #     print('rotated_labels = 0')
                    # print('enable_rotate')
                    # target_labels = xyxy2cxcywh(mosaic_labels)
                    # rotate_img = drawRotationbox(img=mosaic_img.copy(), bboxes=target_labels[:, :4], angles=target_labels[:, 5], heads=target_labels[:, 6])
                    # cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_flip_img.jpg'.format(img_id), rotate_img)
            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            # print('enable_mixup')
            # target_labels = xyxy2cxcywh(mosaic_labels)
            # mixup_img = drawRotationbox(img=mosaic_img.copy(), bboxes=target_labels[:, :4], angles=target_labels[:, 5], heads=target_labels[:, 6])
            # cv2.imwrite('/home/danny/Lab/yolox_test/img_test/flip/{}_flip_img.jpg'.format(img_id), mixup_img)
            mosaic_img, mosaic_labels = self.checkLabels(origin_img=mosaic_img, origin_labels=mosaic_labels)
            
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])
            # cv2.imshow('perspective_img', perspective_img)
            # cv2.imshow('enable_flip', flip_img)
            # cv2.imshow('enable_rotate', rotate_img)
            # cv2.imshow('enable_mixup', mixup_img)


            
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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
        labels = xyxy2cxcywh(origin_labels)
        labels = labels[labels[:,0] > 0]
        labels = labels[labels[:,0] < width]
        labels = labels[labels[:,1] > 0]
        labels = labels[labels[:,1] < height]
        labels = cxcywh2xyxy(labels)

        return origin_img, labels

    def rotate(self, origin_img, origin_labels, degrees=45, scale=1, isdebug=False):
        height, width, _ = origin_img.shape
        a, b = height / 2, width / 2
        target_angle = random.randint(-degrees, degrees)
        
        if isdebug:
            target_angle = 45
            unrotate = origin_img.copy()
            source_labels = xyxy2cxcywh(origin_labels.copy())
            unrotate = drawRotationbox(img=unrotate, bboxes=source_labels[:, :4], angles=source_labels[:, 5], heads=source_labels[:, 6])
        
        M = cv2.getRotationMatrix2D(center=(a, b), angle=target_angle, scale=scale)
        rotated_img = cv2.warpAffine(origin_img.copy(), M, (height, width))  # 旋轉後的圖像保持大小不變

        Pi_angle = -target_angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
        
        rotated_labels = []
        labels = xyxy2cxcywh(origin_labels)
        for label in labels:
            order = int(label[6])
            rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
            poly = cv2.boxPoints(rect)

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
            rect_rotated = cv2.minAreaRect(np.float32(poly_rotated))
            if rect_rotated[0][0] < 0 or rect_rotated[0][0] >= width or rect_rotated[0][1] < 0 or rect_rotated[0][1] >= height:
                continue
            label[:4] = rect_rotated[0][0], rect_rotated[0][1], max(rect_rotated[1]), min(rect_rotated[1])
            
            # get right angle
            point1, point2 = distPoints([X0, Y0], [X1, Y1], [X2, Y2])
            label[5] = checkAngleRange(round(countAngle([point1[0]+5, point1[1]], point1, point2))) # angle range [0~179]
            
            # get new order
            rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
            poly = cv2.boxPoints(rect)
            label[6] = findNewOrder(poly, poly_rotated[0])

            if isdebug:
                poly = np.int0(poly)
                cv2.circle(rotated_img, (int(poly[int(label[6])][0]), int(poly[int(label[6])][1])), 3, (0, 0, 255), -1)
                rotated_img = cv2.drawContours(image=rotated_img.copy(), contours=[poly], contourIdx=-1, color=(0, 255, 0))
            
            rotated_labels.append(label)
        
        if len(rotated_labels) != 0:
            labels = cxcywh2xyxy(np.array(rotated_labels))
        else:
            labels = np.array([])

        if isdebug:
            cv2.imshow('origin_img', unrotate)
            cv2.imshow('rotated_img', rotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return rotated_img, labels

    def flipup(self, origin_img, origin_labels, isdebug=False):
        weight, height, _ = origin_img.shape
        
        # flip_img = origin_img.copy()
        src_labels = xyxy2cxcywh(origin_labels.copy())

        flip_labels = []
        if isdebug:
            unflip = origin_img.copy()
            source_labels = xyxy2cxcywh(origin_labels.copy())
            unflip = drawRotationbox(img=unflip, bboxes=source_labels[:, :4], angles=source_labels[:, 5], heads=source_labels[:, 6])
            
        
        # fliplr 左右翻轉
        if random.uniform(0, 1) > 1:
            # print('左右翻轉')
            flip_img = np.fliplr(origin_img.copy())
            for label in src_labels:
                # print('label')
                # print(label)
                # 未左右翻轉
                rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
                # rect = [(x, y), (w, h), theta]
                # print('rect')
                # print(rect)
                poly = cv2.boxPoints(rect)
                # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                # print('poly')
                # print(poly)
                # head point
                poly[0][0] = weight - poly[0][0]
                poly[1][0] = weight - poly[1][0]
                poly[2][0] = weight - poly[2][0]
                poly[3][0] = weight - poly[3][0]
                poly = np.int0(poly)
                # print(poly)
                
                # real angle
                point1, point2 = distPoints([poly[0][0], poly[0][1]], [poly[1][0], poly[1][1]], [poly[2][0], poly[2][1]])
                test = countAngle([point1[0]+5, point1[1]], point1, point2)
                # print(test)
                realangle = round(test)
                if realangle == 180:
                    realangle = 0
                elif realangle > 180:
                    print('realangle > 180:')          

                flipAngle = 180 - label[5]
                if flipAngle == 180:
                    flipAngle = 0

                rectFlip = longsideformat2cvminAreaRect(weight - label[0], label[1], label[2], label[3], (flipAngle - 179.9))
                polyFlip = cv2.boxPoints(rectFlip)
                new_order = findNewOrder(polyFlip, (poly[int(label[6])-3][0], poly[int(label[6])-3][1]))

                polyFlip = np.int0(polyFlip)
                
                label[0] = weight - label[0]
                label[5] = realangle
                label[6] = new_order
                if isdebug:
                    rectFinal = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
                    polyFinal = cv2.boxPoints(rectFinal)
                    polyFinal = np.int0(polyFinal)
                    flip_img = cv2.drawContours(image=flip_img.copy(), contours=[polyFinal], contourIdx=-1, color=(0, 255, 0))
                    flip_img = cv2.circle(flip_img, (polyFinal[int(label[6])][0], polyFinal[int(label[6])][1]), 3, (0, 255, 0), -1)
                    # flip_img = cv2.putText(flip_img, '{}'.format(realangle), (int(polyFlip[new_order][0]), int(polyFlip[new_order][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

                flip_labels.append(label)
        # flipud 上下翻轉
        else:
            # print('上下翻轉')
            flip_img = np.flipud(origin_img.copy())
            for label in src_labels:
                # 未上下翻轉
                # print('label')
                # print(label)
                rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
                # rect = [(x, y), (w, h), theta]
                # print('rect')
                # print(rect)
                poly = cv2.boxPoints(rect)
                # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                # print('poly')
                # print(poly)
                # head point
                poly[0][1] = height - poly[0][1]
                poly[1][1] = height - poly[1][1]
                poly[2][1] = height - poly[2][1]
                poly[3][1] = height - poly[3][1]
                poly = np.int0(poly)

                
                # print(poly)
                # real angle
                point1, point2 = distPoints([poly[0][0], poly[0][1]], [poly[1][0], poly[1][1]], [poly[2][0], poly[2][1]])
                test = countAngle([point1[0]+15, point1[1]], point1, point2)
                # print(test)
                realangle = round(test)
                if realangle == 180:
                    realangle = 0
                elif realangle > 180:
                    print('realangle > 180:')          

                flipAngle = 180 - label[5]
                if flipAngle == 180:
                    flipAngle = 0

                rectFlip = longsideformat2cvminAreaRect(label[0], height - label[1], label[2], label[3], (flipAngle - 179.9))
                polyFlip = cv2.boxPoints(rectFlip)
                new_order = findNewOrder(polyFlip, (poly[int(label[6])-3][0], poly[int(label[6])-3][1]))

                polyFlip = np.int0(polyFlip)

                label[1] = height - label[1]
                label[5] = realangle
                label[6] = new_order
                if isdebug:
                    rectFinal = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[5] - 179.9))
                    polyFinal = cv2.boxPoints(rectFinal)
                    polyFinal = np.int0(polyFinal)
                    flip_img = cv2.drawContours(image=flip_img.copy(), contours=[polyFinal], contourIdx=-1, color=(0, 255, 0))
                    flip_img = cv2.circle(flip_img, (polyFinal[int(label[6])][0], polyFinal[int(label[6])][1]), 3, (0, 255, 0), -1)
                    # flip_img = cv2.putText(flip_img, '{}'.format(realangle), (int(polyFlip[new_order][0]), int(polyFlip[new_order][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

                flip_labels.append(label)
        
        if isdebug:
            cv2.imshow('origin_img', unflip)
            cv2.imshow('leftright', flip_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if len(flip_labels) != 0:
            origin_labels = cxcywh2xyxy(np.array(flip_labels))
        else:
            origin_labels = np.array([])
        
        return flip_img, origin_labels

    def mixup(self, origin_img, origin_labels, input_dim):
        # print(origin_img.shape)
        cp_index = random.randint(0, self.__len__() - 1)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        # print(img.shape)
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        img = (origin_img * r + img * (1 - r)).astype(np.uint8)
        labels = np.concatenate((origin_labels, cp_labels), 0)
        return img, labels


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