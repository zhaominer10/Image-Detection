import math
import os
from typing import List, Optional, Dict, Tuple

import torchvision
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torch import Tensor

# sizes = ((32, 64, 128, 256, 512),)
# if not isinstance(sizes[ 0 ], (list, tuple)):
#     sizes = tuple((s,) for s in sizes)
# print(sizes)
#
# aspect_ratios = ((0.5, 1.0, 2.0),)
# if not isinstance(aspect_ratios[ 0 ], (list, tuple)):
#     aspect_ratios = (aspect_ratios,) * len(sizes)
# print(aspect_ratios)
#
#
# def generate_anchors(scales, ratios):
#     # type: # (List[int], List[float]) -> Tensor
#
#     scales = torch.as_tensor(scales)
#     ratios = torch.as_tensor(ratios)
#
#     h_ratios = torch.sqrt(ratios)
#     w_ratios = 1.0 / h_ratios
#
#     a = w_ratios[ :, None ]
#     b = scales[ None, : ]
#     c = a * b
#
#     ws = (w_ratios[ :, None ] * scales[ None, : ]).view(-1)
#     hs = (h_ratios[ :, None ] * scales[ None, : ]).view(-1)
#
#     base_anchors = torch.stack([ -ws, -hs, ws, hs ], dim=1) / 2
#     d = base_anchors.round()
#     print(base_anchors)
#
#
# for i, j in zip(sizes, aspect_ratios):
#     generate_anchors(i, j)

##################################################


# root_1 = "../VOC2012/JPEGImages/2007_000027.jpg"
# root_2 = "../VOC2012/JPEGImages/2007_000032.jpg"
# root_3 = "../VOC2012/JPEGImages/2007_000033.jpg"
# root_4 = "../VOC2012/JPEGImages/2007_000039.jpg"
# image_1 = Image.open(root_1)
# image_2 = Image.open(root_2)
# image_3 = Image.open(root_3)
# image_4 = Image.open(root_4)
# image_1 = F.to_tensor(image_1)
# image_2 = F.to_tensor(image_2)
# image_3 = F.to_tensor(image_3)
# image_4 = F.to_tensor(image_4)
#
# images = image_1
# feature_maps = [image_4]
#
# image_sizes = [ img.shape[ -2: ] for img in images ]
#
# image_sizes_list = torch.jit.annotate(List[ Tuple[ int, int ] ], [ ])
#
# for image_size in image_sizes:
#     assert len(image_size) == 2
#     image_sizes_list.append((image_size[ 0 ], image_size[ 1 ]))
#
# # 将处理后的图像已经图像的长宽信息打包在一起
# # images 是batch images，长宽相同
# # image_sizes_list 是图像normalize后，resize后，batch_image统一长宽之前的图像尺寸
# # targets 是resize后的bbox信息
# from FasterRCNN.network.image_list import ImageList
#
# image_list = ImageList(images, image_sizes_list)
#
# image_hw = image_list.tensors.shape[-2:]
# grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
#
# strides = [[torch.tensor(image_hw[0] // g[0]), torch.tensor(image_hw[1] // g[1])] for g in grid_sizes]

a = [1,2,3]
b = list(a)