from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.solver import WarmupMultiStepLR

def get_vitdet_cfg():
    cfg = get_cfg()

    # Load base model configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Model weights
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
    
    # Backbone configuration
    cfg.MODEL.BACKBONE.NAME = "build_vit_fpn_backbone"
    
    # Data augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = (256, 384)
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # Dataset
    cfg.DATASETS.TRAIN = ("RAW_coco_train",)
    cfg.DATASETS.TEST = ("RAW_coco_val",)
    
    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 2

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 184375
    cfg.SOLVER.STEPS = (163889, 177546)
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 250
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.AMP.ENABLED = True  # Enable automatic mixed precision

    # ROI Heads
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Adjust based on your dataset

    # Output directory
    cfg.OUTPUT_DIR = "./output"
    
    return cfg
