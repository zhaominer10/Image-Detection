import json

json_path = '../COCO/annotations/instances_val2017.json'
json_labels = json.load(open(json_path))
print(json_labels['info'])
