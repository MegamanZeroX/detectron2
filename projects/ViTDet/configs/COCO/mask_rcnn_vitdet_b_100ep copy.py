from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_lsj import dataloader

import os
from detectron2.data.datasets import register_coco_instances

#cstm_dataset
def cstm_register_coco():
    dataset_path = os.getenv("DETECTRON2_DATASETS", "/data1/")
    train_json = "/home/yzhang63/instances_train2017.json"
    train_images = os.path.join(dataset_path, "coco/images/train2017")
    val_json = os.path.join(dataset_path, "coco/annotations/instances_val2017.json")
    val_images = os.path.join(dataset_path, "coco/images/val2017")

    register_coco_instances("my_coco_train", {}, train_json, train_images)
    register_coco_instances("my_coco_val", {}, val_json, val_images)

    print("Dataset registered successfully.")

cstm_register_coco()
#cstm_dataset ends
dataloader.train.dataset.names = "my_coco_train"
dataloader.test.dataset.names = "my_coco_val"
dataloader.train.total_batch_size  = 2


model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)


# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
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

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
