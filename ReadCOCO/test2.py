import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

json_path = '../COCO/annotations/instances_val2017.json'
img_path = '../COCO/val2017'

coco = COCO(annotation_file=json_path)

ids = list(sorted(coco.imgs.keys()))
print('number of images: {}'.format(len(ids)))

coco_classes = dict([(v['id'], v['name']) for k, v in coco.cats.items()])
print('yes')

for img_id in ids[:3]:
    # 单张图片的所有标注信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)

    path = coco.loadImgs(img_id)[0]['file_name']

    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)

    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target['category_id']])

    plt.imshow(img)
    plt.show()
