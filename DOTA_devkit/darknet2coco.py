import cv2 as cv
from pathlib import Path
import argparse
import json
import configparser as cfg
import os
import shutil

'''
gendata_path : the path of gen_config.data


classes=2
train=gen_train.txt
valid=gen_valid.txt
names=qmobj.names

'''

class DARKNET2COCO:
    def __init__(self,genconfig_data):
        self.src_data=genconfig_data
        self.src=Path(self.src_data).parent
        self.dst=Path(self.src) / "coco_dataset"
        self.coco_train="train2017"
        self.coco_valid="val2017"
        self.coco_images="images"
        self.coco_annotation="annotations"
        self.coco_train_json=Path(self.dst)/Path(self.coco_annotation) / f'instances_{self.coco_train}.json'
        self.coco_valid_json=Path(self.dst)/Path(self.coco_annotation) / f'instances_{self.coco_valid}.json'
        self.type = 'instances'
        self.categories = []
        self.info = {
            'year': 2021,
            'version': '1.0',
            'description': 'For object detection',
            'date_created': '2021',
        }
        self.licenses = [{
            'id': 1,
            'name': 'GNU General Public License v3.0',
            'url': 'https://github.com/RapidAI/YOLO2COCO/LICENSE',
        }]

        if not Path(self.dst).is_dir():
            Path(self.dst).mkdir()

        if not Path(self.dst/ self.coco_images).is_dir():
            Path(self.dst/self.coco_images).mkdir()

        if not (Path(self.dst )/self.coco_images/ self.coco_train).is_dir():
            ( Path(self.dst)/self.coco_images/self.coco_train).mkdir()


        if not Path(self.dst /self.coco_images/ self.coco_valid).is_dir():
            ( Path(self.dst)/self.coco_images/self.coco_valid).mkdir()


        if not (Path(self.dst )/ self.coco_annotation).is_dir():
            ( Path(self.dst)/self.coco_annotation).mkdir()

        if Path(self.src_data).is_file():
            self.ready=True
            self.initcfg()
        else:
            self.ready=False

    def initcfg(self):
        if  not self.ready:
            return
        self.cnf = cfg.RawConfigParser()
        with open(self.src_data) as f:
            file_content = '[dummy_section]\n' + f.read()
        self.cnf.read_string(file_content)

    def getint(self,key):
        if not self.ready:
            return 0
        return int(self.cnf.get("dummy_section",key))

    def getstring(self,key):
        if not self.ready:
            return ""
        return self.cnf.get("dummy_section",key)




    def get_path(self,name):
        content=[]
        with open(name) as f:
            allfiles=f.readlines()
        for file in allfiles:
            if not os.path.isabs(file):
                this_path=Path(self.src)/file.strip()
                content.append(str(this_path))
            else:
                content.append(file.strip())
        return content

    def get_list(self,name):
        content=[]
        with open(name) as f:
            allfiles=f.readlines()
        for file in allfiles:
            content.append(file.strip())

        return content
# derived from https://github.com/zhiqwang/yolov5-rt-stack/blob/master/yolort/utils/yolo2coco.py
    def _get_annotation(self,vertex_info, height, width):

        cx, cy, w, h, angle, order = [float(i) for i in vertex_info]
        cx = cx * width
        cy = cy * height
        w = w * width
        h = h * height
        x = cx - w / 2
        y = cy - h / 2

        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        area = w * h

        bbox = [x, y, w, h]
        return segmentation, bbox, area, angle, order, cx, cy

    def read_annotation(self,txtfile,img_id,height,width,annotation_id):
        annotation=[]

        if not Path(txtfile).exists():
            return {},0
        with open(txtfile) as f:
                allinfo=f.readlines()

        for line in allinfo:
                label_info=line.replace('\n', '').replace('\r', '')
                label_info=label_info.strip().split(" ")
                if len(label_info) < 5:
                    continue

                category_id, vertex_info = label_info[0], label_info[1:]

                segmentation, bbox, area, angle, order, cx, cy = self._get_annotation(vertex_info, height, width)
                if cx < 0 or cy < 0:
                    print('if cx < 0 or cy < 0:')
                    continue
                if cx > width or cy > height:
                    print('if cx > width or cy > height:')
                    continue                
                annotation.append( {
                            'segmentation': segmentation,
                            'area': area,
                            'iscrowd': 0,
                            'image_id': img_id,
                            'bbox': bbox,
                            'category_id': int(int(category_id)+1),
                            'id': annotation_id,
                            'order': int(order),
                            'angle': int(angle),
                        })
                annotation_id+=1

        return annotation,annotation_id

    def get_category(self):

        for id,category in enumerate(self.name_lists,1):
          self.categories.append({
            'id': id,
            'name': category,
            'supercategory': category,
        } )

    def generate(self):
        self.classnum= self.getint("classes")

        self.train= Path( self.src_data).parent / Path(self.getstring("train")).name
        self.valid= Path( self.src_data).parent / Path(self.getstring("valid")).name
        self.names=Path( self.src_data).parent / Path(self.getstring("names")).name

        self.train_files=self.get_path(self.train)

        if os.path.exists(self.valid):
            self.valid_files=self.get_path(self.valid)
        self.name_lists=self.get_list(self.names)
        self.get_category()

        dest_path_train=Path(self.dst)/ self.coco_images/self.coco_train
        self.gen_dataset(self.train_files,dest_path_train,self.coco_train_json)

        dest_path_valid=Path(self.dst)/self.coco_images/self.coco_valid
        if os.path.exists(self.valid):
            self.gen_dataset(self.valid_files,dest_path_valid,self.coco_valid_json)

        print("The output directory is :",str(self.dst))

#  https://cocodataset.org/#format-data
    def gen_dataset(self,file_lists,target_img_path,target_json):

        images=[]
        annotations=[]
        annotation_id=1

        for img_id,file in   enumerate(file_lists,1):
            if not Path(file).exists():
                continue
            txt= str(Path(file).parent / Path(file).stem) + ".txt"  # from 0,  0 readhead, 1 stamp

            tmpname=str(img_id)
            prefix="0"*(12- len(tmpname))
            destfilename=prefix+tmpname+".jpg"
            
            imgsrc = cv.imread(file)


            image = imgsrc.shape
            height = image[0]
            width = image[1]


            if Path(txt).exists():
                new_anno,annotation_id=self.read_annotation(txt,img_id,height,width,annotation_id)
                if len(new_anno)>0:
                    annotations.extend(new_anno)
                else:
                    print('刪圖')
                    continue


            if Path(file).suffix.lower() == ".jpg":
                shutil.copyfile(file,target_img_path/destfilename)
            else:
                cv.imwrite(str(target_img_path/destfilename),imgsrc)
            # shutil.copyfile(file,target_img_path/ )
            
            images.append({
                'date_captured': '2021',
                'file_name': destfilename,
                'id': img_id,

                'height': height,
                'width': width,
            })



        json_data = {
            'info': self.info,
            'images': images,
            'licenses': self.licenses,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        with open(target_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)


class YOLO2COCO(DARKNET2COCO):
    def __init__(self,srcdir):
        self.srcdir=srcdir
        self.srcimgdir=Path(srcdir)/"images/train2017"
        self.srclabeldir=Path(srcdir)/"labels/train2017"
        


        if not self.srcimgdir.exists() or not self.srclabeldir.exists():
            raise "wrong path, not found labels or images dir."
        self.dstdir=Path(srcdir)/"darknet"
        self.gen_config=self.dstdir/"gen_config.data"
        self.train=self.dstdir/"gen_train.txt"

        self.dstimgdir=self.dstdir/"images"/"train"
        if not self.dstimgdir.exists():
            self.dstimgdir.mkdir(parents=True, exist_ok=True)

        self.classlist=set()
        self.classname=self.dstdir/"classes.names"
        self.convert2darknet()

    def convert2darknet(self):
        imgfiles= self.srcimgdir.rglob("*.jpg")
        with open(self.train,"w") as f:
            for file in imgfiles:
                labelfile=self.srclabeldir/file.stem
                labelfile=str(labelfile)+".txt"
                dstimg= self.dstimgdir/file.name
                dstlabel=self.dstimgdir/Path(labelfile).name
                shutil.copy(file,dstimg)
                shutil.copy(labelfile,dstlabel)
                f.write(str(dstimg)+"\n")
                # get class number
                with open(labelfile,"r") as lf:
                    lines=lf.readlines()
                    for line in lines:
                        info=line.split()
                        self.classlist.add(int(info[0]))

        if self.train.exists():
            with open(self.gen_config,"w") as f:
                line="classes="+str(max(self.classlist)+1)+"\n"
                f.write(line)
                f.write("train=gen_train.txt"+"\n")
                f.write("names=classes.names"+"\n")
                f.write("valid=none\n")


            with open(self.classname,"w") as f:
                maxclass=max(self.classlist)
                for clsid in range(maxclass+1):
                    f.write("class_"+str(clsid)+"\n")

    def generate(self):
        super(YOLO2COCO,self).__init__(str(self.gen_config))
        super(YOLO2COCO,self).generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Datasets converter from yolo to coco', add_help=False)

    parser.add_argument('--data_path', default='data/getn_config.data',
                        help='Dataset root path')
    parser.add_argument('--split', default='train2017',
                        help='Dataset split part, optional: [train2017, val2017]')

    args = parser.parse_args()

    if Path(args.data_path).is_file():
        converter = DARKNET2COCO(args.data_path)
    else:
        converter = YOLO2COCO(args.data_path)
    converter.generate()
