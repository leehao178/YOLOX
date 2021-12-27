import os
import numpy as np
import cv2
from torch.serialization import validate_cuda_device
from yolox.utils import cvminAreaRect2longsideformat
from yolox.utils.boxes import fast_nms
from yolox.utils.utils import checkAngleRange, findNewOrder, countAngle, distPoints
import torch
from yolox.evaluators import voc_eval
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

def multi_classes_nms(srcpath):
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
                keep = fast_nms(boxes=dets[:, :5], scores=dets[:, 5], NMS_threshold=0.1, cluster=True, giou=True, enclosing_type='pca')
                dets = dets[keep]
                # print(dets.shape)
                dets = dets.cpu().numpy()
                # print(dets.shape)
                for i in dets:
                    class_lines.append('{} {} {} {} {} {} {} {} {} {}\n'.format( img, i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14]))
            os.makedirs(os.path.join(srcpath, 'fast_nms'), exist_ok=True)
            with open(os.path.join(srcpath, 'fast_nms', classnametxt), mode='w') as nf:
                nf.writelines(class_lines)
val = False

root = '/home/aimlusr/YOLOX/YOLOX_outputs/18_m_ms_mc_fp_r_diou/result_riou/test/detection'
result_before_merge_path = str(root + '/result_txt/result_before_merge')
result_merged_path = str(root + '/result_txt/result_merged')
result_classname_path = str(root + '/result_txt/result_classname')
imageset_name_file_path = str(root + '/result_txt')

annopath = r'/home/aimlusr/dataset/dota10/val/labelTxt/{:s}.txt'
imagesetfile = str(imageset_name_file_path +'/imgnamefile.txt')  # 'r/.../imgnamefile.txt'  測試集圖片名稱txt
detpath = str(result_classname_path + '/fast_nms/Task1_{:s}.txt')  # 'r/.../Task1_{:s}.txt'  存放各類別結果文件txt的路徑

multi_classes_nms(result_classname_path)

if val == True:
    classaps = []
    map = 0
    skippedClassCount = 0
    for classname in classnames:
        # if iscountmAP == False:
        # print(classname)
        detfile = detpath.format(classname)
        if not (os.path.exists(detfile)):
            skippedClassCount += 1
            print('This class is not be detected in your dataset: {:s}'.format(classname))
            continue
        rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                ovthresh=0.5,
                                use_07_metric=True)
        map = map + ap
        print(ap)
        classaps.append(str(ap.item()))
    map = map/(len(classnames)-skippedClassCount)
    print(map)
