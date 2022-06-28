import os
import datetime

import torch
import torchvision

import transforms

from my_dataset import VOC2012DataSet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device for training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = '../VOC2012'
    aspect_ratio_group_factor = 3
    batch_size = 8
    amp = False  # 是否需要混合精度训练，需要GPU支持

    # check VOC dataset
    if os.path.exists(VOC_root) is False:
        raise FileExistsError("VOC dataset does not exist in path:{}.".format(VOC_root))

    # load train data set
    train_dataset = VOC2012DataSet(voc_root=VOC_root,
                                   transforms=data_transform["train"],
                                   train_set=True)
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减少训练时所需GPU显存，默认使用

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("using {} dataloader workers.".format(nw))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和target，不能直接使用默认的方法合成batch
    # if train_sampler


if __name__ == "__main__":
    main()
