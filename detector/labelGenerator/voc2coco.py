# -*- coding: utf-8 -*-
# ===========================   introduce  ====================================
# Created on Sat Feb 29 21:16:02 2020
# @author: Lim
# 
# English: 1.use this tool to label your own picture:
#            https://link.zhihu.com/?target=https%3A//github.com/chinakook/labelImg2
#          2.use this function PascalVOC2coco to convert all "*.xml" file label produced by 1. 
#            to a .josn file, also is train data of R-CenterNet.  
# 中文：   1.先用这个工具对你自己的数据打标签 
#            https://link.zhihu.com/?target=https%3A//github.com/chinakook/labelImg2
#          2.用PascalVOC2coco将你打完标签的所有.xml文件合并成一个json文件，也就是R-CenterNet需要的
#            训练数据
# =============================================================================
# -*- coding:utf-8 -*-
# !/usr/bin/env python

import json
import cv2
import numpy as np
import glob
import PIL.Image
import os, sys
import math
from tqdm import tqdm  # 导入 tqdm


class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.ob = []
        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(tqdm(self.xml, desc="Converting XML to COCO", unit="file")):
            self.json_file = json_file
            self.num = num
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)
            # print(path)
            with open(json_file, 'r', encoding='UTF-8') as fp:

                flag = 0
                # content = fp.readline()
                for p in fp:
                    if not p.strip():  # 跳过空行
                        continue
                    f_name = 1
                    # print(p)
                    if 'filename' in p:
                        self.filen_ame = p.split('>')[1].split('<')[0]
                        f_name = 0
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(os.path.dirname(current_dir))
                        img_dir = os.path.join(project_root, 'data', 'RoLabelImg_Transform', 'img')
                        self.path = os.path.join(img_dir, self.filen_ame.split('.')[0] + '.jpg')
                    if 'width' in p:
                        # print(p.split('>')[1].split('<')[0])
                        self.width = float(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])
                        self.images.append(self.image())
                    if flag == 1:
                        self.supercategory = self.ob[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)
                        x1 = float(self.ob[1])
                        y1 = float(self.ob[2])
                        w = float(self.ob[3])
                        h = float(self.ob[4])
                        angle = float(self.ob[5]) * 180 / math.pi
                        angle = angle if angle < 180 else angle - 180

                        self.rectangle = [x1, y1, x1 + w, y1 + w]
                        self.bbox = [x1, y1, w, h, angle]  

                        self.annotations.append(self.annotation())
                        self.annID += 1
                        self.ob = []
                        flag = 0
                    elif f_name == 1:
                        key = p.split('>')[0].split('<')[1]
                        # print(key)
                        if key == 'name':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'cx':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'cy':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'w':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'h':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'angle':
                            self.ob.append(p.split('>')[1].split('<')[0])
                            flag = 1

    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1 
        categorie['name'] = self.supercategory
        return categorie

    def annotation(self):
        annotation = {}

        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1

        annotation['bbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self):

        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i;
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i;
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask

            return self.mask2polygons()

        except:
            return [0]

    def mask2polygons(self):
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        os.makedirs(os.path.dirname(self.save_json_path), exist_ok=True)
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
xml_dir = os.path.join(project_root, 'data', 'RoLabelImg_Transform', 'xml', '*.xml')
xml_file = glob.glob(xml_dir)

annotations_dir = os.path.join(project_root, 'detector', 'data', 'annotations')
train_json_path = os.path.join(annotations_dir, 'train.json')

print(f"Found {len(xml_file)} XML files to convert")

PascalVOC2coco(xml_file, train_json_path)