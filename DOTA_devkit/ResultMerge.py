"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import dota_utils as util
import re
import time
import polyiou

## the thresh for nms when merge image
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

def nmsbynamedict(nameboxdict, nms, thresh):
    """
    對namedict中的目標信息進行nms.不改變輸入的數據形式
    @param nameboxdict: eg:{
                           'P706':[[poly1, confidence1], ..., [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., [poly9, confidence9]]
                            }
    @param nms:
    @param thresh: nms閾值, IoU閾值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1], ..., [poly_nms, confidence9]],
                                 ...
                                'P700':[[poly1, confidence1], ..., [poly_nms, confidence9]]
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
            outdets.append(nameboxdict[imgname][index])
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

def mergebase(srcpath, dstpath, nms):
    """
    將源路徑中所有的txt目標信息,經nms後存入目標路徑中的同名txt
    @param srcpath: 合併前信息保存的txt源路徑
    @param dstpath: 合併後信息保存的txt目標路徑
    @param nms: NMS函數
    """
    filelist = util.GetFileFromThisRootDir(srcpath)  # srcpath文件夾下的所有文件相對路徑 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = util.custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, name + '.txt')  # eg: example_merge/..P0001.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            lines = f_in.readlines()  # 讀取txt中所有行,每行作為一個元素存於list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m個元素
            for splitline in splitlines:  # splitline:每行中的m個元素
                # splitline = [待merge圖片名(該目標所處圖片名稱), confidence, x1, y1, x2, y2, x3, y3, x4, y4]
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
                poly = list(map(float, splitline[2:]))  # 每個元素映射為浮點數 再放入列表中
                origpoly = poly2origpoly(poly, x, y, rate)  # 將目標位置信息resize 恢復成原圖的poly坐標
                det = origpoly  # shape(8)
                det.append(confidence)  # [poly, confidence]
                det = list(map(float, det))
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []   # 弄個元組,匯集原圖目標信息 eg: 'P706':[[poly1, confidence1], ..., ]
                nameboxdict[oriname].append(det)
            nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)  # 對nameboxdict元組進行nms
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:
                    for det in nameboxnmsdict[imgname]:  # 取出對應圖片的nms後的目標信息
                        #print('det:', det)
                        confidence = det[-1]
                        bbox = det[0:-1]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
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
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms.txt的信息格式為:[P0770__1__0___0 confidence poly]
    dstpath: result files after merge and nms.保存的txt信息格式為:[P0770 confidence poly]
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)
if __name__ == '__main__':
    # see demo for example
    mergebypoly(r'ResultMerge_example', r'ResultMerge_example_result')
    # mergebyrec()