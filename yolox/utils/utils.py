#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2021 lab602 Hao Lee. All rights reserved.

import cv2
import numpy as np
from yolox.utils import xyxy2cxcywh
import math
from math import sqrt, acos

from yolox.utils.boxes import cxcywh2xyxy
import random
# import math
__all__ = ["longsideformat2cvminAreaRect", 
           "drawRotationbox", 
           "cvminAreaRect2longsideformat", 
           "angle_smooth_label",
           "checkAngleRange",
           "findNewOrder", 
           "countAngle",
           "distPoints",
           "debugDrawBox",
           "findHeadPoint"]


def findHeadPoint(headpoint, points):
    realheadpoint = []
    hx, hy = headpoint
    for i in points:
        dist = math.hypot(i[0] - hx, i[1] - hy)
        realheadpoint.append(dist)
                                                            # 最接近原始起點的點位置
    return points[realheadpoint.index(min(realheadpoint))] , realheadpoint.index(min(realheadpoint))

def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    兩者區別為:
            當opencv表示法中width為最長邊時（包括正方形的情況），則兩種表示方法一致
            當opencv表示法中width不為最長邊 ，則最長邊表示法的角度要在opencv的Θ基礎上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x軸逆時針旋轉碰到的第一條邊
    @param height: 與width不同的邊
    @param theta: x軸逆時針旋轉與width的夾角，由於原點位於圖像的左上角，逆時針旋轉角度為負 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最長邊
            shortside: 最短邊
            theta_longside: 最長邊和x軸逆時針旋轉的夾角，逆時針方向角度為負 [-180, 0)
    '''
    '''
    意外情況:(此時要將它們恢復符合規則的opencv形式：wh交換，Θ置為-90)
    豎直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90說明wh中有為0的元素，即gt信息不完整，無需提示異常，直接刪除
            print('Θ=90說明wh中有為0的元素，即gt信息不完整，無需提示異常，直接刪除，當前數據為：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的範圍：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ計算出現異常，當前數據為：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的範圍：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最長邊
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最長邊(包括正方形的情況)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋轉框轉換錶示形式後出現問題：最長邊小於短邊;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋轉框轉換錶示形式時出現問題:θ超出長邊表示法的範圍：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside, isdebug=False):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    兩者區別為:
            當opencv表示法中width為最長邊時（包括正方形的情況），則兩種表示方法一致
            當opencv表示法中width不為最長邊 ，則最長邊表示法的角度要在opencv的Θ基礎上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最長邊
    @param shortside: 最短邊
    @param theta_longside: 最長邊和x軸逆時針旋轉的夾角，逆時針方向角度為負 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x軸逆時針旋轉碰到的第一條邊最長邊
            height: 與width不同的邊
            theta: x軸逆時針旋轉與width的夾角，由於原點位於圖像的左上角，逆時針旋轉角度為負 [-90, 0)
    '''
    if ((theta_longside >= -180) and (theta_longside < -90)):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if (theta < -90) or (theta >= 0):
        if isdebug:
            print('當前θ=%.1f，超出opencv的θ定義範圍[-90, 0)' % theta)

    return ((int(x_c), int(y_c)), (int(width), int(height)), theta)

def drawRotationbox(img, bboxes, angles, heads=None):
    if heads is not None:
        for box, angle, head in zip(bboxes, angles, heads):
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            rect = longsideformat2cvminAreaRect(box[0], box[1], box[2], box[3], (angle - 179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))
            poly = np.int0(poly)
            cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=(0, 255, 0))
            if heads is not None:
                cv2.circle(img, (int(poly[int(head)][0]), int(poly[int(head)][1])), 2, (0, 255, 0), -1)
    else:
        for box, angle in zip(bboxes, angles):
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            rect = longsideformat2cvminAreaRect(box[0], box[1], box[2], box[3], (angle - 179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))
            poly = np.int0(poly)
            cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=(0, 255, 0))
    
    return img

def debugDrawBox(img, bboxes, isDraw=False, num='0'):
    if bboxes.shape[1] == 7:
        xyxy_bboxes = cxcywh2xyxy(bboxes)
        for cxcywhcls0h, xyxycls0h in zip(bboxes, xyxy_bboxes):
            rect = longsideformat2cvminAreaRect(cxcywhcls0h[0], cxcywhcls0h[1], cxcywhcls0h[2], cxcywhcls0h[3], (cxcywhcls0h[5] - 179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))
            poly = np.int0(poly)
            cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=(0, 255, 0), thickness=1)
            cv2.circle(img, (int(poly[int(cxcywhcls0h[6])][0]), int(poly[int(cxcywhcls0h[6])][1])), 2, (0, 255, 0), -1)

            cv2.rectangle(img, (int(xyxycls0h[0]), int(xyxycls0h[1])), (int(xyxycls0h[2]), int(xyxycls0h[3])), (255, 0, 0), 1)

    elif bboxes.shape[1] == 6:
        xyxy_bboxes = cxcywh2xyxy(bboxes)
        for cxcywhcls0h, xyxycls0h in zip(bboxes, xyxy_bboxes):
            rect = longsideformat2cvminAreaRect(cxcywhcls0h[0], cxcywhcls0h[1], cxcywhcls0h[2], cxcywhcls0h[3], (cxcywhcls0h[5] - 179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))
            poly = np.int0(poly)
            cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=(0, 255, 0), thickness=1)

            cv2.rectangle(img, (int(xyxycls0h[0]), int(xyxycls0h[1])), (int(xyxycls0h[2]), int(xyxycls0h[3])), (255, 0, 0), 1)
    else:
        pass

    if isDraw:
        
        cv2.imwrite('/home/danny/Lab/yolox_test/img_test/{}_mosaic_img.jpg'.format(num), img)
    else:
        cv2.imshow('test', img)
        cv2.waitKey(0)

    return img

def gaussian_label(label, num_class, u=0, sig=4.0):
    x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    return np.concatenate([y_sig[math.ceil(num_class/2)-label:],
                        y_sig[:math.ceil(num_class/2)-label]], axis=0)

def get_all_smooth_label(num_label, label_type=0, raduius=4):
    all_smooth_label = []

    if label_type == 0:
        for i in range(num_label):
            all_smooth_label.append(gaussian_label(i, num_label, sig=raduius))
    # elif label_type == 1:
    #     for i in range(num_label):
    #         all_smooth_label.append(rectangular_label(i, num_label, raduius=raduius))
    # elif label_type == 2:
    #     for i in range(num_label):
    #         all_smooth_label.append(pulse_label(i, num_label))
    # elif label_type == 3:
    #     for i in range(num_label):
    #         all_smooth_label.append(triangle_label(i, num_label, raduius=raduius))
    else:
        raise Exception('Only support gaussian, rectangular, triangle and pulse label')
    return np.array(all_smooth_label)

def angle_smooth_label(angle_label, num_angle_cls=36, label_type=0, raduius=6):
    """
    :param angle_label: [-90,0) or [-90, 0)
    :param angle_range: 90 or 180
    :return:
    """
    # assert angle_range % omega == 0, 'wrong omega'

    # angle_range /= omega
    # angle_label /= omega

    # angle_label = np.array(-np.round(angle_label), np.int32)
    # angle_label = np.array(np.round(angle_label), np.int32)
    all_smooth_label = get_all_smooth_label(num_label=num_angle_cls, label_type=label_type, raduius=raduius)
    inx = angle_label == num_angle_cls
    angle_label[inx] = num_angle_cls - 1
    smooth_label = all_smooth_label[angle_label]

    return np.array(smooth_label, np.float32)

def checkAngleRange(angle):
    if angle == 180:
        angle = 0
        # print('== 180')
    elif angle > 180:
        pass
        # print('angle > 180:')
    return angle

def findNewOrder(points, target):
    distList = []
    for point in points:
        distList.append(math.hypot(point[0] - target[0], point[1] - target[1]))
    order = distList.index(min(distList))
    return order  

# import numpy as np

# a = np.array([32.49, -39.96,-3.86])
# b = np.array([31.39, -39.28, -4.66])
# c = np.array([31.14, -38.09,-4.49])

# ba = a - b
# bc = c - b

# cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
# angle = np.arccos(cosine_angle)

# print np.degrees(angle)

def countAngle(a, b, c):
    # print('countAngle1')
    # print(a)
    # print(b)
    # print(c)
    # Create vectors from points
    ba = [ aa-bb for aa,bb in zip(a,b) ]
    bc = [ cc-bb for cc,bb in zip(c,b) ]
    
    # Normalize vector
    nba = sqrt ( sum ( (x**2.0 for x in ba) ) )
    ba = [ x/nba for x in ba ]
    
    nbc = sqrt ( sum ( (x**2.0 for x in bc) ) )
    
    # print(nbc)
    bc = [ x/nbc for x in bc ]
    
    # Calculate scalar from normalized vectors
    scalar = sum ( (aa*bb for aa,bb in zip(ba,bc)) )
    
    # calculate the angle in radian
    angle = acos(scalar)
    
    # print(angle)
    angle_degree = math.degrees(angle)
    
    # print(angle_degree)
    # print('countAngle2')
    return angle_degree

def distPoints(a, b, c):
    dista = math.hypot(a[0] - b[0], a[1] - b[1])
    distb = math.hypot(c[0] - b[0], c[1] - b[1])
    if dista > distb:
        if a[1] > b[1]:
            return b, a
        else:
            return a, b
    else:
        if b[1] > c[1]:
            return c, b
        else:
            return b, c

def checkOpencvIsClockwise():
    # [[x, y], [w, h], angle]
    rect = [[100, 100], [100, 50], 30]

    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = cv2.boxPoints(rect)
    poly = np.int0(poly)

    img = np.zeros([250,250,3])
    cv2.drawContours(img, [poly], 0, (255, 0, 0), 1)

    cv2.circle(img, (int(poly[0][0]), int(poly[0][1])), 7, (0, 0, 255), -1)
    cv2.circle(img, (int(poly[1][0]), int(poly[1][1])), 5, (0, 0, 255), -1)
    cv2.circle(img, (int(poly[2][0]), int(poly[2][1])), 3, (0, 0, 255), -1)
    cv2.circle(img, (int(poly[3][0]), int(poly[3][1])), 1, (0, 0, 255), -1)

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
