# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.segment import SegmentationPredictor

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr



def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = "yolov8x-seg.pt"
    #data = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/tless_mod_train.yaml"
    data = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/tlessmod_seg.yaml"
    device = cfg.device if cfg.device is not None else 'cuda:0'

    last_path = ""
    args = dict(name = "pbr_blk_tlessmod_segment_500",
                model=model,    #model=last_path,    
                resume=False,    #resume=True,                          
                #weights="/home/ise.ros/akshay_work/NN_Implementations/surfemb/runs/detect/train7/weights/last.pt",
                device=device, 
                data=data,
                batch=32, epochs=500, imgsz=320,
                #   lr0: Initial LR, 
                #   lrf: Multiplier for final LR (Final LR = lr0*lrf)''' 
                lr0=0.0000035, lrf=0.0001, 
                patience=6000,
                show_labels=True, val=True, #augment=True, 
                )
                #save_dir="/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/runs/detect/train")
    if use_python:
        from ultralytics import YOLO
        yolo_model = YOLO(model)
        yolo_model.train(**args)
    else:
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        '''dataset_path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_yolo_train"
        dl = trainer.get_dataloader(dataset_path, batch_size=16, rank=0, mode='train')
        print()'''


if __name__ == '__main__':
    train()