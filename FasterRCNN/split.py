import os
import random

"""
本FasterRCNN项目使用的PASCAL VOC2012数据集，该数据集已分好训练集和验证集。本python脚本是为需要自行分割数据集的情况而提供的。
这里已PASCAL VOC2012 annotation作为示例，进行训练集数据与验证集分割。
"""


def main():
    random.seed(0)

    file_path = "../VOC2012/Annotations"
    assert os.path.exists(file_path), "path: {} does not exist.".format(file_path)

    val_rate = 0.5

    files_name = sorted([file.split('.')[0] for file in os.listdir(file_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num * val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")  # x写模式，新建一个文件，如果该文件已存在则会报错
        val_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        val_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
