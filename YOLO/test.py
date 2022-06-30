from PIL import Image
from torchvision.transforms import functional as F
import torch


# path_1 = "../VOC2012/JPEGImages/2007_000027.jpg"
# path_2 = "../VOC2012/JPEGImages/2007_000033.jpg"
#
# path_3 = "../VOC2012/JPEGImages/2007_000042.jpg"
#
# feature_map_1 = Image.open(path_1)
# feature_map_2 = Image.open(path_2)
# image_list = Image.open(path_3)
#
# feature_map_1 = F.to_tensor(feature_map_1)
# feature_map_2 = F.to_tensor(feature_map_2)
# image_list = F.to_tensor(image_list)
#
# feature_maps = [feature_map_1, feature_map_2]
# image_size = image_list.shape[-2:]
#
# grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
#
# strides = [[torch.tensor(image_size[0] // g[0]), torch.tensor(image_size[1] // g[1])] for g in grid_sizes]
#
# key = str(grid_sizes) + str(strides)
#
# print(grid_sizes)
# print(strides)
# print(key)
#
#
# a = torch.arange(0, 3)*4
# print(a)


import torch
a = torch.arange(0, 10, 2)
print(a)
b = torch.arange(0, 20, 5)
print(b)
x, y = torch.meshgrid(a, b)
print(x)
print(y)
c = x.reshape(-1)
d = y.reshape(-1)
shifts = torch.stack([c, d, c, d], dim=1)
# print(x)
# print(y)
print(c)
print(d)
print(shifts)
print(shifts.shape)
print(shifts.view(-1, 1, 4).shape)