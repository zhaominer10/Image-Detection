from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_true = COCO(annotation_file='../COCO/annotations/instances_val2017.json')

# predict_results.json is not provided
coco_pre = coco_true.loadRes('./predict_results.json')

coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType='bbox')
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
