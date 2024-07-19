import json
import os
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_lsj import dataloader
#from ...register_npy_coco import register_npy_coco

# Register cstm dataset

def load_npy_coco(image_dir, annotation_file, dataset_type):
    with open(annotation_file) as f:
        coco_annotations = json.load(f)
    
    dataset_dicts = []
    for idx, item in enumerate(coco_annotations["images"]):
        if dataset_type in item["file_name"]:
            record = {}
            filename = f"raw_pred_{item['file_name'].replace('.jpg', '.npy')}"
            height, width = item["height"], item["width"]
            
            filepath = os.path.join(image_dir, filename)
            record["file_name"] = filepath
            record["image_id"] = item["id"]
            record["height"] = height
            record["width"] = width
            
            annos = [anno for anno in coco_annotations["annotations"] if anno["image_id"] == item["id"]]
            objs = []
            for anno in annos:
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": anno["category_id"],
                    "segmentation": anno["segmentation"],
                    "iscrowd": anno["iscrowd"]
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def register_npy_coco():
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    base_dir = "/data2/COCO_RAW_NPY/COCO_RAW_NPY"
    annotation_train = "/home/yzhang63/COCO2014annotations/instances_train2014.json"
    annotation_val = "/home/yzhang63/COCO2014annotations/instances_val2014.json"
    if "RAW_coco_train" not in DatasetCatalog.list():
        DatasetCatalog.register("RAW_coco_train", lambda: load_npy_coco(base_dir, annotation_train, "train2014"))
        MetadataCatalog.get("RAW_coco_train").set(thing_classes=COCO_CLASSES)
    if "RAW_coco_val" not in DatasetCatalog.list():
        DatasetCatalog.register("RAW_coco_val", lambda: load_npy_coco(base_dir, annotation_val, "val2014"))
        MetadataCatalog.get("RAW_coco_val").set(thing_classes=COCO_CLASSES)

register_npy_coco()


dataloader.train.dataset.names = "RAW_coco_train"
dataloader.test.dataset.names = "RAW_coco_val"
dataloader.train.total_batch_size  = 2
# cstm dataset ends

dataloader.train.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=(256, 384), max_size=640),
    L(T.RandomFlip)()
]



model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model



train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)



train.max_iter = 184375

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer settings
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}


