# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or '/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolov8n.pt'
    data = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/tless_mod_train.yaml"
    device = cfg.device if cfg.device is not None else 'cuda:0'

    last_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/runs/detect/pbr_blk_regen_tless_mod_1500/weights/best.pt"
    args = dict(name = "pbr_blk_regen_tless_mod_1800",
                model=last_path,    #model=model,
                resume=False,    #resume=True,                          
                #weights="/home/ise.ros/akshay_work/NN_Implementations/surfemb/runs/detect/train7/weights/last.pt",
                device=device, 
                data=data,
                batch=32, epochs=300, imgsz=320,
                #   lr0: Initial LR, 
                #   lrf: Multiplier for final LR (Final LR = lr0*lrf)''' 
                lr0=0.0025, lrf=0.0001, cls=0.75,  #cls: CLASS LOSS GAIN
                patience=6000,
                show_labels=True, val=True, #augment=True,
                cache=True, workers=8 
                )
                #save_dir="/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/runs/detect/train")
    if use_python:
        from ultralytics import YOLO
        yolo_model = YOLO(model)
        yolo_model.train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        '''dataset_path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_yolo_train"
        dl = trainer.get_dataloader(dataset_path, batch_size=16, rank=0, mode='train')
        print()'''


if __name__ == '__main__':
    train()