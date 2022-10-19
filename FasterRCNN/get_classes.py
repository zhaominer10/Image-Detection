import json
import os

txt_path = '../VOC2012/ImageSets/Main'
list_classes = os.listdir(txt_path)

classes = []

for i in list_classes:
    cla = i.split('_')

    if len(cla) == 2 and cla[0] not in classes:
        classes.append(cla[0])

classes = sorted(classes)
values = list(range(1, 21, 1))

classes = dict(zip(classes, values))
classes = json.dumps(classes, indent=4)

with open('pascal_voc_classes.json', 'w') as json_file:
    json_file.write(classes)

print("hello")
