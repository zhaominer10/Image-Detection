import os
from lxml import etree
import json
import torch


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
print(xml)

data = parse_xml_to_dict(xml)

# data = json.dumps(data, indent=4)
# print(data)

data = data['annotation']
print(data)

boxes = []
labels = []
iscrowd = []
assert "object" in data, "{} lack of object information.".format(xml_path)
for obj in data["object"]:
    xmin = float(obj["bndbox"]["xmin"])
    xmax = float(obj["bndbox"]["xmax"])
    ymin = float(obj["bndbox"]["ymin"])
    ymax = float(obj["bndbox"]["ymax"])

    # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
    if xmax <= xmin or ymax <= ymin:
        print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
        continue

    boxes.append([xmin, ymin, xmax, ymax])
    labels.append(1)
    if "difficult" in obj:
        iscrowd.append(int(obj["difficult"]))
    else:
        iscrowd.append(0)

# convert everything into a torch.Tensor
boxes = torch.as_tensor(boxes, dtype=torch.float32)
labels = torch.as_tensor(labels, dtype=torch.int64)
iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
image_id = torch.tensor([idx])
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
print(area)
