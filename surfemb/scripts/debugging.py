import argparse
from pathlib import Path

import cv2
import torch.utils.data
import numpy as np


ROOT_DIR = "D:\\Akshay_Work\\aks_git_repos\\surfemb"  #os.path.dirname(os.path.abspath(__file__))
MASK_DIR = "D:\\Akshay_Work\\aks_git_repos\\surfemb\\maskrcnn_train"
import sys
sys.path.append(ROOT_DIR)
sys.path.append(MASK_DIR)
from surfemb import utils
from surfemb.data import obj
from surfemb.data.config import config
from surfemb.data import instance
from surfemb.data import detector_crops
from surfemb.data.renderer import ObjCoordRenderer
from surfemb.surface_embedding import SurfaceEmbeddingModel
from surfemb import pose_est
from surfemb import pose_refine


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="D:\\Akshay_Work\\aks_git_repos\\surfemb\\data\\models\\motor-vlyro4oe-500k-steps.ckpt")
parser.add_argument('--real', default=True)   #action='store_false')
parser.add_argument('--detection', default=True)   #action='store_false')
parser.add_argument('--i', type=int, default=987)
parser.add_argument('--device', default='cuda:0')

args = parser.parse_args()
data_i = args.i
device = torch.device(args.device)
model_path = Path(args.model_path)

model = SurfaceEmbeddingModel.load_from_checkpoint(args.model_path)
model.eval()
model.freeze()
model.to(device)

dataset = model_path.name.split('-')[0]
real = args.real
detection = args.detection
root = Path('data/bop') / dataset
cfg = config[dataset]
res_crop = 224

objs, obj_ids = obj.load_objs(root / cfg.model_folder)
renderer = ObjCoordRenderer(objs, res_crop)
assert len(obj_ids) == model.n_objs
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)
auxs = model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=detection)
dataset_args = dict(dataset_root=root, obj_ids=obj_ids, auxs=auxs, cfg=cfg)
if detection:
    assert args.real
    data = detector_crops.DetectorCropDataset(
        **dataset_args, detection_folder=Path("data") / Path(f'detection_results/{dataset}')
    )
else:
    data = instance.BopInstanceDataset(**dataset_args, pbr=not args.real, test=args.real)


inst = data[data_i]
obj_idx = inst['obj_idx']
img = inst['rgb_crop']
K_crop = inst['K_crop']
obj_ = objs[obj_idx]
print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')

cv2.imshow("rgb_crop", img)
cv2.waitKey(0)