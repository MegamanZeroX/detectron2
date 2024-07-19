import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

def main():
    from register_npy_coco import register_npy_coco
    register_npy_coco()

    cfg = get_cfg()
    cfg.merge_from_file("/home/yzhang63/detectron2/projects/ViTDet/configs/COCO/RAW_mask_rcnn_vitdet_b_100ep.yaml")
    cfg.DATASETS.TRAIN = ("RAW_coco_train",)
    cfg.DATASETS.TEST = ("RAW_coco_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Adjust based on your dataset
    cfg.SOLVER.AMP.ENABLED = True  # Enable automatic mixed precision

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
