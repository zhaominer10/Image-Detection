from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOC2012DataSet(Dataset):
    """读取解析PASCAL VOC2012数据集"""

    def __init__(self, voc_root, transforms, train_set=True):
        self.root = voc_root
        self.image_root = os.path.join(self.root, 'JPEGImages')
        self.annotation_root = os.path.join(self.root, 'Annotations')

        if train_set:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        else:
            txt_list = os.path.join(self.)