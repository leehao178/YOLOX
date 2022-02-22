import torch
from yolox.utils import xyxy2cxcywh, longsideformat2cvminAreaRect
import cv2
import os
# -*- coding: utf-8 -*-
"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import re
import time
from yolox.utils.iou import polyiou
import copy
import cv2
import random
from PIL import Image
from yolox.utils.utils import checkAngleRange, findNewOrder, countAngle, distPoints
from yolox.utils.boxes import fast_nms
## the IoU thresh for nms when merge image
nms_thresh = 0.3

def py_cpu_nms_poly(dets, thresh):
    """
    任意四點poly nms.取出nms後的邊框的索引
    @param dets: shape(detection_num, [poly, confidence1]) 原始圖像中的檢測出的目標數量
    @param thresh:
    @return:
            keep: 經nms後的目標邊框的索引
    """
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)

    # argsort將元素小到大排列 返回索引值 [::-1]即從後向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 順序為從大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出當前剩餘置信度最大的目標邊框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly與其他所有poly的IoU
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小於閾值的索引
        order = order[inds + 1]
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nameboxdict_classname, nms, thresh):
    """
    對namedict中的目標信息進行nms.不改變輸入的數據形式
    @param nameboxdict: eg:{
                           'P706':[[poly1, confidence1], ..., [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., [poly9, confidence9]]
                            }
    @param nameboxdict_classname: eg:{
                           'P706':[[poly1, confidence1,'classname'], ..., [poly9, confidence9, 'classname']],
                           ...
                           'P700':[[poly1, confidence1, 'classname'], ..., [poly9, confidence9, 'classname']]
                            }
    @param nms:
    @param thresh: nms阈值, IoU阈值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']],
                                 ...
                                'P700':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']]
                               }
    """
    # 初始化字典
    nameboxnmsdict = {x: [] for x in nameboxdict}  # eg: nameboxnmsdict={'P0770': [], 'P1888': []}
    for imgname in nameboxdict:  # 提取nameboxdict中的key eg:P0770   P1888
        keep = nms(np.array(nameboxdict[imgname]), thresh)  # rotated_nms索引值列表
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict_classname[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict

def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def mergebase(srcpath, dstpath, nms, ismulti_cls_nms=False):
    """
    將源路徑中所有的txt目標信息,經nms後存入目標路徑中的同名txt
    @param srcpath: 合併前信息保存的txt源路徑
    @param dstpath: 合併後信息保存的txt目標路徑
    @param nms: NMS函數
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夾下的所有文件相對路徑 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, name + '.txt')  # eg: example_merge/..P0001.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            nameboxdict_classname = {}
            lines = f_in.readlines()  # 讀取txt中所有行,每行作為一個元素存於list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m個元素
            for splitline in splitlines:  # splitline:每行中的m個元素
                # splitline = [待merge圖片名(該目標所處圖片名稱), confidence, x1, y1, x2, y2, x3, y3, x4, y4, classname]
                subname = splitline[0]  # 每行的第一個元素 是被分割的圖片的圖片名 eg:P0706__1__0___0
                splitname = subname.split('__')  # 分割待merge的圖像的名稱 eg:['P0706','1','0','_0']
                oriname = splitname[0]  # 獲得待merge圖像的原圖像名稱 eg:P706
                pattern1 = re.compile(r'__\d+___\d+')  # 預先編譯好r'__\d+___\d+' 提高重複使用效率 \d表示數字

                x_y = re.findall(pattern1, subname)  # 匹配subname中的字符串 eg: x_y=['__0___0']
                x_y_2 = re.findall(r'\d+', x_y[0])  # 匹配subname中的字符串 eg: x_y_2= ['0','0']
                x, y = int(x_y_2[0]), int(x_y_2[1])  # 找到當前subname圖片在原圖中的分割位置

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')  # \.表示一切字符

                rate = re.findall(pattern2, subname)[0]  # 找到該subname分割圖片時的分割rate (resize rate before cut)

                confidence = splitline[1]
                classname = splitline[-1]
                poly = list(map(float, splitline[2:10]))  # 每個元素映射為浮點數 再放入列表中
                origpoly = poly2origpoly(poly, x, y, rate)  # 將目標位置信息resize 恢復成原圖的poly坐標
                det = origpoly  # shape(8)
                det.append(confidence)  # [poly, 'confidence']
                det = list(map(float, det))  # [poly, confidence]

                det_classname = copy.deepcopy(det)
                det_classname.append(classname)  # [poly, 'confidence','classname']
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []   # 弄個元組,匯集原圖目標信息 eg: 'P706':[[poly1, confidence1], ..., ]
                    nameboxdict_classname[oriname] = []   # 弄個元組,匯集原圖目標信息 eg: 'P706':[[poly1, confidence1,'classname'], ..., ]
                nameboxdict[oriname].append(det)
                nameboxdict_classname[oriname].append(det_classname)

            if ismulti_cls_nms:
                pass
            else:
                nameboxnmsdict = nmsbynamedict(nameboxdict, nameboxdict_classname, nms, nms_thresh)  # 對nameboxdict元組進行nms
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:  # 'P706'
                    for det in nameboxnmsdict[imgname]:  # 取出對應圖片的nms後的目標信息
                        # det:[poly1, confidence1, 'classname']
                        #print('det:', det)
                        confidence = det[-2]
                        bbox = det[0:-2]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox)) + ' ' + det[-1]
                        #print('outline:', outline)
                        f_out.write(outline + '\n')

def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath, ismulti_cls_nms=False):
    """
    @param srcpath: result files before merge and nms.txt的信息格式為:[P0770__1__0___0 confidence poly 'classname']
    @param dstpath: result files after merge and nms.保存的txt信息格式為:[P0770 confidence poly 'classname']
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    if ismulti_cls_nms:
        mergebase(srcpath,
                dstpath,
                py_cpu_nms_poly)
    else:
        mergebase(srcpath,
                dstpath,
                py_cpu_nms_poly)

def rbox2txt(rbox, classname, conf, img_name, out_path, pi_format=False):
    """
    將分割圖片的目標信息填入原始圖片.txt中
    @param robx: rbox:[tensor(x),tensor(y),tensor(l),tensor(s),tensor(θ)]
    @param classname: string
    @param conf: string
    @param img_name: string
    @param path: 文件夾路徑 str
    @param pi_format: θ是否為pi且 θ ∈ [-pi/2,pi/2)  False說明 θ∈[0,179]
    """
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()

    #rbox = np.array(x)
    if pi_format:  # θ∈[-pi/2,pi/2)
        rbox[-1] = (rbox[-1] * 180 / np.pi) + 90  # θ∈[0,179]

    # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
    rect = longsideformat2cvminAreaRect(rbox[0], rbox[1], rbox[2], rbox[3], (rbox[4] - 179.9))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.float32(cv2.boxPoints(rect))
    poly = np.int0(poly).reshape(8)

    splitname = img_name.split('__')  # 分割待merge的圖像的名稱 eg:['P0706','1','0','_0']
    oriname = splitname[0]  # 獲得待merge圖像的原圖像名稱 eg:P706

    # 目標所屬圖片名稱_分割id 置信度 poly classname
    lines = img_name + ' ' + conf + ' ' + ' '.join(list(map(str, poly))) + ' ' + classname
    # 移除之前的輸出文件夾,並新建輸出文件夾
    if not os.path.exists(out_path):
        os.makedirs(out_path)  # make new output folder

    with open(str(out_path + '/' + oriname) + '.txt', 'a') as f:
        f.writelines(lines + '\n')

def evaluation_trans(srcpath, dstpath):
    """
    將srcpath文件夾中的所有txt中的目標提取出來,按照目標類別分別存入 Task1_類別名.txt中:
            txt中的內容格式:  目標所屬原始圖片名稱 置信度 poly
    @param srcpath: 存放圖片的目標檢測結果(文件夾,內含多個txt)
                    txt中的內容格式: 目標所屬圖片名稱 置信度 poly 'classname'
    @param dstpath: 存放圖片的目標檢測結果(文件夾, 內含多個Task1_類別名.txt )
                    txt中的內容格式:  目標所屬原始圖片名稱 置信度 poly
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夾下的所有文件相對路徑 eg:['result_merged/P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'result_merged/P0001.txt'
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            lines = f_in.readlines()  # 讀取txt中所有行,每行作為一個元素存於list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m個元素
            for splitline in splitlines:  # splitline:每行中的m个元素
                # splitline = [目標所屬圖片名稱, confidence, x1, y1, x2, y2, x3, y3, x4, y4, 'classname']
                classname = splitline[-1]  # 每行的最後一個元素 是被分割的圖片的種類名
                dstname = os.path.join(dstpath, 'Task1_' + classname + '.txt')  # eg: result/Task1_plane.txt
                lines_ = ' '.join(list(splitline[:-1]))
                with open(dstname, 'a') as f:
                    f.writelines(lines_ + '\n')

def image2txt(srcpath, dstpath):
    """
    將srcpath文件夾下的所有子文件名稱打印到namefile.txt中
    @param srcpath: imageset
    @param dstpath: imgnamefile.txt的存放路徑
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夾下的所有文件相對路徑 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, 'imgnamefile.txt')  # eg: result/imgnamefile.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(dstname, 'a') as f:
            f.writelines(name + '\n')

def draw_DOTA_image(imgsrcpath, imglabelspath, dstpath, extractclassname, thickness):
    """
    繪製工具merge後的目標/DOTA GT圖像
    @param imgsrcpath: merged後的圖像路徑(原始圖像路徑)
    @param imglabelspath: merged後的labels路徑
    @param dstpath: 目標繪製之後的保存路徑
    @param extractclassname: the category you selected
    """
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    # 設置畫框的顏色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]隨機設置RGB顏色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = GetFileFromThisRootDir(imglabelspath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = []
        with open(fullname, 'r') as f_in:  # 打開merge後/原始的DOTA圖像的gt.txt
            lines = f_in.readlines()  # 讀取txt中所有行,每行作為一個元素存於list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m個元素
            if len(splitlines[0]) == 1:  # 首行為"imagesource:GoogleEarth",說明為DOTA原始labels
                # DOTA labels:[polys classname 1/0]
                del splitlines[0]
                del splitlines[0]   # 刪除前兩個無用信息
                objects = [x[0:-2] for x in splitlines]
                classnames = [x[-2] for x in splitlines]
            else:
                # P0003 0.911 660.0 309.0 639.0 209.0 661.0 204.0 682.0 304.0 large-vehicle
                objects = [x[2:-1] for x in splitlines]
                classnames = [x[-1] for x in splitlines]

        '''
        objects[i] = str[poly, classname]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgsrcpath, name + '.png')
        img_savename = os.path.join(dstpath, name + '_.png')
        img = cv2.imread(img_fullname)  # 讀取圖像像素

        for i, obj in enumerate(objects):
            # obj = [poly ,'classname']
            classname = classnames[i]
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.array(list(map(float, obj)))
            poly = poly.reshape(4, 2)  # 返回rect對應的四個點的值 normalized
            poly = np.int0(poly)

            # 畫出來
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(extractclassname.index(classname))],
                             thickness=thickness)
        cv2.imwrite(img_savename, img)


def multi_classes_nms(srcpath):
    for classnametxt in os.listdir(srcpath):
        
        imgs_dict = {}
        with open(os.path.join(srcpath, classnametxt), mode='r') as f:
            for imgs_det in f.readlines():
                img, conf, x1, y1, x2, y2, x3, y3, x4, y4 = imgs_det.replace('\n', '').split(' ')
                if img in imgs_dict:
                    imgs_dict[img].append([float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4), float(conf)])
                else:
                    imgs_dict[img] = [[float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4), float(conf)]]
        class_lines = []
        for img, dets in imgs_dict.items():
            keep = py_cpu_nms_poly(np.array(dets), 0.1)
            for i in keep:
                # print(keep)
                # print(dets)
                class_lines.append('{} {} {} {} {} {} {} {} {} {}\n'.format( img, dets[i][8], dets[i][0], dets[i][1], dets[i][2], dets[i][3], dets[i][4], dets[i][5], dets[i][6], dets[i][7]))
        os.makedirs(os.path.join(srcpath, 'nms'), exist_ok=True)
        with open(os.path.join(srcpath, 'nms', classnametxt), mode='w') as nf:
            nf.writelines(class_lines)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def fast_nms2(srcpath):
    for classnametxt in os.listdir(srcpath):
        if classnametxt.find('.txt') != -1:
            imgs_dict = {}
            with open(os.path.join(srcpath, classnametxt), mode='r') as f:
                for imgs_det in f.readlines():
                    
                    img, conf, x1, y1, x2, y2, x3, y3, x4, y4 = imgs_det.replace('\n', '').split(' ')
                    poly = [(float(x1), float(y1)), (float(x2), float(y2)), (float(x3), float(y3)), (float(x4), float(y4))]
                    rect = cv2.minAreaRect(np.float32(np.array(poly)))
                    c_x = rect[0][0]
                    c_y = rect[0][1]
                    longside = max(rect[1])
                    shortside = min(rect[1])
                    theta = rect[-1]  # Range for angle is [-90，0)
                    # label[:4] = rect_rotated[0][0], rect_rotated[0][1], max(rect_rotated[1]), min(rect_rotated[1])
                    # get right angle
                    point1, point2 = distPoints([float(x1), float(y1)], [float(x2), float(y2)], [float(x3), float(y3)])
                    theta = checkAngleRange(round(countAngle([point1[0]+5, point1[1]], point1, point2))) # angle range [0~179]
                    # print(len(rect))
                    # trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                    # print(len(trans_data))
                    bbox = np.array((c_x, c_y, longside, shortside, theta, float(conf), float(conf), float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)))
                    # print(bbox)
                    if img in imgs_dict:
                        imgs_dict[img].append(bbox)
                    else:
                        imgs_dict[img] = [bbox]
            class_lines = []
            for img, dets in imgs_dict.items():
                dets = torch.tensor(dets).to(device)
                keep = fast_nms(boxes=dets[:, :5], scores=dets[:, 5], NMS_threshold=0.1, cluster=True, giou=True)
                dets = dets[keep]
                # print(dets.shape)
                dets = dets.cpu().numpy()
                # print(dets.shape)
                for i in dets:
                    class_lines.append('{} {} {} {} {} {} {} {} {} {}\n'.format( img, i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14]))
            os.makedirs(os.path.join(srcpath, 'fast_nms'), exist_ok=True)
            with open(os.path.join(srcpath, 'fast_nms', classnametxt), mode='w') as nf:
                nf.writelines(class_lines)


if __name__ == '__main__':
    '''
        計算AP的流程:
        1.detect.py檢測一個文件夾的所有圖片並把檢測結果按照圖片原始來源存入 原始圖片名稱.txt中:   (rbox2txt函數)
            txt中的內容格式:  目標所屬圖片名稱_分割id 置信度 poly classname
        2.ResultMerge.py將所有 原始圖片名稱.txt 進行merge和nms,並將結果存入到另一個文件夾的 原始圖片名稱.txt中: (mergebypoly函數)
            txt中的內容格式:  目標所屬圖片名稱 置信度 poly classname
        3.寫一個evaluation_trans函數將上個文件夾中的所有txt中的目標提取出來,按照目標類別分別存入 Task1_類別名.txt中:
            txt中的內容格式:  目標所屬原始圖片名稱 置信度 poly
        4.寫一個imgname2txt.py 將測試集的所有圖片名稱打印到namefile.txt中
    '''
    # see demo for example
    mergebypoly(r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_before_merge',
                r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged')
    print('檢測結果已merge')
    evaluation_trans(r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged',
                     r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_classname')
    print('檢測結果已按照類別分類')
    image2txt(r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_images',  # val原圖數據集路徑
              r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt')
    print('校驗數據集名稱文件已生成')

    # classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
    #                    'ship', 'tennis-court',
    #                    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
    #                    'helicopter', 'container-crane']
    #
    # draw_DOTA_image(imgsrcpath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_images',
    #                 imglabelspath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged',
    #                 dstpath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/merged_drawed',
    #                 extractclassname=classnames_v1_5,
    #                 thickness=2
    #                 )