import os
from lxml import etree
import json


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式， 参考TensorFlow的recursive_parse_xml_to_dict
    """
    if len(xml) == 0:  # 查看当前xml是否有子目录
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


txt_list = os.path.join('../VOC2012', 'ImageSets', 'Main', 'train.txt')
with open(txt_list) as read:
    xml_list = [os.path.join('../VOC2012/Annotations', line.strip() + '.xml') for line in read.readlines()]

idx = 0
xml_path = xml_list[idx]

with open(xml_path) as f:
    xml_str = f.read()

xml = etree.fromstring(xml_str)
data = parse_xml_to_dict(xml)

data = json.dumps(data, indent=4)
print(data)
