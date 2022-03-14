# -*- coding: utf-8 -*-
import dota_utils as util
import os
import numpy as np
from PIL import Image
import cv2
import random
import  shutil
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint  # 多邊形
import time
import argparse
import math
# import shutil
## trans dota format to format YOLO(darknet) required
def dota2Darknet(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
           txt format: id x y w h
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_dota_poly(fullname)
        '''
        objects =
        [{'name': 'ship', 
          'difficult': '1', 
          'poly': [(1054.0, 1028.0), (1063.0, 1011.0), (1111.0, 1040.0), (1112.0, 1062.0)], 
          'area': 1159.5
          },
          ...
        ]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']  # poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))  # bbox=[x y w h]
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:  # 若bbox中有<=0或>= 1的元素則將該box排除
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=類名的索引 比如'plane'對應id=0
                else:
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))  # outline='id x y w h'
                f_out.write(outline + '\n')  # 寫入txt文件中並加上換行符號 \n

def findHeadPoint(headpoint, points):
    realheadpoint = []
    hx, hy = headpoint
    for i in points:
        dist = math.hypot(i[0] - hx, i[1] - hy)
        realheadpoint.append(dist)
                                                            # 最接近原始起點的點位置
    return points[realheadpoint.index(min(realheadpoint))] , realheadpoint.index(min(realheadpoint))

## trans dota format to  (cls, c_x, c_y, Longest side, short side, angle:[0,179))
def dota2LongSideFormat(imgpath, txtpath, dstpath, extractclassname):
    """
    trans dota farmat to longside format
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_dota_poly(fullname)
        '''
        objects =
        [{'name': 'ship', 
          'difficult': '1', 
          'poly': [(1054.0, 1028.0), (1063.0, 1011.0), (1111.0, 1040.0), (1112.0, 1062.0)], 
          'area': 1159.5
          },
          ...
        ]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        img = Image.open(img_fullname)
        cvimg = cv2.imread(img_fullname)
        img_w, img_h = img.size
        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            num_gt = 0
            for i, obj in enumerate(objects):
                num_gt = num_gt + 1  # 為當前有效gt計數
                poly = obj['poly']  # poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                # print(poly)

                head = poly[0]

                poly = np.float32(np.array(poly))
                # 四點坐標歸一化 轉化成yolo格式
                poly[:, 0] = poly[:, 0]/img_w
                poly[:, 1] = poly[:, 1]/img_h


                rect = cv2.minAreaRect(poly)  # 得到最小外接矩形的（中心(x,y), (寬,高), 旋轉角度）
               # box = np.float32(cv2.boxPoints(rect))  # 返回rect四個點的值

                c_x = rect[0][0]
                c_y = rect[0][1]
                w = rect[1][0]
                h = rect[1][1]
                theta = rect[-1]  # Range for angle is [-90，0)

                trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                if not trans_data:
                    if theta != 90:  # Θ=90說明wh中有為0的元素，即gt信息不完整，無需提示異常，直接刪除
                        print('opencv表示法轉長邊表示法出現異常,已將第%d個box排除,問題出現在該圖片中:%s' % (i, img_fullname))
                    num_gt = num_gt - 1
                    continue
                else:
                    # range:[-180，0)
                    c_x, c_y, longside, shortside, theta_longside = trans_data

                bbox = np.array((c_x, c_y, longside, shortside))

                if (sum(bbox <= 0) + sum(bbox[:2] >= 1) ) >= 1:  # 0<xy<1, 0<side<=1
                    print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已將第%d個box排除,問題出現在該圖片中:%s' % (i, img_fullname))
                    print('出問題的longside形式數據:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (c_x, c_y, longside, shortside, theta_longside))
                    num_gt = num_gt - 1
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=類名的索引 比如'plane'對應id=0
                else:
                    print('預定類別中沒有類別:%s;已將該box排除,問題出現在該圖片中:%s' % (obj['name'], fullname))
                    num_gt = num_gt - 1
                    continue
                theta_label = int(theta_longside + 180.5)  # range int[0,180] 四捨五入
                if theta_label == 180:  # range int[0,179]
                    theta_label = 179
                # outline='id x y longside shortside Θ'

                # final check
                if id > 15 or id < 0:
                    print('id problems,問題出現在該圖片中:%s' % (i, img_fullname))
                    print('出問題的longside形式數據:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                    c_x, c_y, longside, shortside, theta_longside))
                if theta_label < 0 or theta_label > 179:
                    print('id problems,問題出現在該圖片中:%s' % (i, img_fullname))
                    print('出問題的longside形式數據:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                
                rect = longsideformat2cvminAreaRect(bbox[0], bbox[1], bbox[2], bbox[3], (theta_label-179.9))
                # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                poly = np.float32(cv2.boxPoints(rect))  # 返回rect對應的四個點的值 normalized
                # 四點坐標反歸一化 取整
                poly[:, 0] = poly[:, 0] * img_w
                poly[:, 1] = poly[:, 1] * img_h
                poly = np.int0(poly)

                ans, indexHead = findHeadPoint(headpoint=head, points=poly.tolist())
                # print(index)
                cv2.circle(cvimg, (int(ans[0]), int(ans[1]) ), 5, (255, 0, 0), -1)
                
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox))) + ' ' + str(theta_label) + ' ' + str(indexHead)
                
                # print(bbox)
                f_out.write(outline + '\n')  # 寫入txt文件中並加上換行符號 \n
            # cv2.imshow('test', cvimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        if num_gt == 0:
            os.remove(os.path.join(dstpath, name + '.txt'))  #
            # os.remove(img_fullname)
            # os.remove(fullname)
            print('%s 圖片對應的txt不存在有效目標,已刪除對應圖片與txt' % img_fullname)
        else:
            shutil.copy2(img_fullname, os.path.join(dstpath, name + '.png'))
    print('已完成文件夾內DOTA數據形式到長邊表示法的轉換')


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
            print('θ計算出現異常，當前數據為：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的範圍：[-90,0)' % (x_c, y_c, width, height, theta))
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

def drawLongsideFormatimg(imgpath, txtpath, dstpath, extractclassname, thickness=2):
    """
    根據labels繪製邊框(label_format:classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, Θ)
    :param imgpath: the path of images
    :param txtpath: the path of txt in longside format
    :param dstpath: the path of image_drawed
    :param extractclassname: the category you selected
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    # 設置畫框的顏色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]隨機設置RGB顏色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        if fullname.find('png') != -1:
            continue
        objects = util.parse_longsideformat(fullname)
        '''
        objects[i] = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, theta]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        img_savename = os.path.join(dstpath, name + '_.png')  # img_fullname='/.../_P000?.png'
        img = Image.open(img_fullname)  # 圖像被打開但未被讀取
        img_w, img_h = img.size
        print(img_w)
        print(img_h)
        img = cv2.imread(img_fullname)  # 讀取圖像像素
        for i, obj in enumerate(objects):
            print(obj)
            
            # obj = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, float:0-179]
            class_index = obj[0]
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            rect = longsideformat2cvminAreaRect(obj[1], obj[2], obj[3], obj[4], (obj[5]-179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))  # 返回rect對應的四個點的值 normalized

            # 四點坐標反歸一化 取整
            poly[:, 0] = poly[:, 0] * img_w
            poly[:, 1] = poly[:, 1] * img_h
            poly = np.int0(poly)
            print(poly)

            cv2.circle(img, (int(poly[obj[6]][0]), int(poly[obj[6]][1])), 3, (0,255,0), -1)
            # 畫出來
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(class_index)],
                             thickness=thickness)
        cv2.imwrite(img_savename, img)

    # time.sleep()

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
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
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('當前θ=%.1f，超出opencv的θ定義範圍[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)

def delete(imgpath, txtpath):
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.png')  # img_fullname='/.../P000?.png'
        if not os.path.exists(img_fullname):  # 如果文件bu存在
            os.remove(fullname)

if __name__ == '__main__':
    ## an example
    # dota1.0
    dota2LongSideFormat(imgpath='example/train_split/images',
                        txtpath='example/train_split/labelTxt',
                        dstpath='example/yolo/train_split',
                        extractclassname=util.wordname_15)

    drawLongsideFormatimg(imgpath='example/train_split/images',
                          txtpath='example/yolo/train_split',
                          dstpath='example/yolo/train_result',
                          extractclassname=util.wordname_15)

