#!/usr/bin/env python3

import os
import numpy as np
import random
import xml.etree.ElementTree as ET

from utils import get_classes


class Annotator(object):
    def __init__(self):
        # 0: get all
        # 1: get ImageSets dir's *.txt
        # 2: get 2007_train.txt, 2007_val.txt
        self.anno_mode = 2

        # only valid when anno_mode = 0|2
        self.classes_path = 'data/voc/classes_voc.txt'

        self.train_ratio = 0.9
        self.val_ratio = 0.1

        self.vocdevit_path = 'data/voc/VOCdevkit/'
        self.vocdevit_sets = [('2007', 'train'),('2007', 'val')]

        self.classes, _ = get_classes(self.classes_path)

        self.sets_num = np.zeros(len(self.vocdevit_sets), dtype=np.int32)
        self.class_nums = np.zeros(len(self.classes), dtype=np.int32)
    
    def convert_anno(self, year, image_id, list_file):
        in_file = open(os.path.join(self.vocdevit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
        tree = ET.parse(in_file)
        root  = tree.getroot()

        for obj in root.iter('object'):
            diff = 0
            if obj.find("difficult") != None:
                diff = obj.find("difficult").text
        
            cls = obj.find('name').text
            if cls not in self.classes or int(diff) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (int(float(xmlbox.find("xmin").text)), int(float(xmlbox.find("ymin").text)),
                 int(float(xmlbox.find("xmax").text)), int(float(xmlbox.find("ymax").text)))
            
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

            self.class_nums[self.classes.index(cls)] =  self.class_nums[self.classes.index(cls)] + 1



if __name__ == '__main__':

    random.seed(0)

    anno = Annotator()
    if ' ' in os.path.abspath(anno.vocdevit_path):
        raise ValueError("please remove empty space in vocdevit_path")
    
    if anno.anno_mode == 0 or anno.anno_mode == 1:
        print("gen txts in ImageSets")
        xmlfile_root = os.path.join(anno.vocdevit_path, 'VOC2007/Annotations')
        saveBase = os.path.join(anno.vocdevit_path, 'VOC2007/ImageSets/Main')
        # TODO gen 
    if anno.anno_mode == 0 or anno.anno_mode == 2:
        print("gen 2007_train.txt, 2007_val.txt")
        type_index = 0
        for year, image_set in anno.vocdevit_sets:
            image_ids = open(os.path.join(anno.vocdevit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)),encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')

            for img_id in image_ids:
                list_file.write("%s/VOC%s/JPEGImages/%s.jpg"%(anno.vocdevit_path, year, img_id))
                anno.convert_anno(year, img_id, list_file)
                list_file.write("\n")
            anno.sets_num[type_index] = len(image_ids)
            type_index = type_index + 1
            list_file.close()
        print("Generated %s_train.txt, %s_val.txt done for train"%(year, year))

        def printTable(l1, l2):
            for i in range(len(l1[0])):
                print("|", end= ' ')
                for j in range(len(l1)):
                    print(l1[j][i].rjust(int(l2[j])), end=' ')
                    print("|", end =' ')
                print()

        str_nums = [str(int(x)) for x in anno.class_nums]
        tableD = [
            anno.classes, str_nums
        ]
        colWidth = [0] * len(tableD)

        len1 = 0

        for i in range(len(tableD)):
            for j in range(len(tableD[i])):
                if len(tableD[i][j])> colWidth[i]:
                    colWidth[i] = len(tableD[i][j])
        printTable(tableD, colWidth)
                            

        if anno.sets_num[0] <= 500:
            print("train set less than 500, notify the epoch(can not be set too big.)")

        if np.sum(anno.class_nums) == 0:
            raise RuntimeError("no class? check this out")

