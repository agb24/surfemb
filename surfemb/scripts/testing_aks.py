import argparse
from pathlib import Path
import sys

import cv2
import torch.utils.data
import numpy as np


# Using this to fix relative import issues & allow easier debug. 
# Else, must run file as "python -m <file>.py". 
ROOT_DIR = "D:\\Akshay_Work\\aks_git_repos\\surfemb"  #os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from surfemb import utils
from surfemb.data import obj
from surfemb.data.config import config
from surfemb.data import instance
from surfemb.data import detector_crops
from surfemb.data.renderer import ObjCoordRenderer
from surfemb.surface_embedding import SurfaceEmbeddingModel
from surfemb import pose_est
from surfemb import pose_refine


from surfemb.dep.unet import ResNetUNet
from surfemb.dep.siren import Siren
from surfemb.data.obj import Obj
from surfemb.data.tfms import denormalize
from surfemb import utils


# could be extended to allow other mlp architectures
mlp_class_dict = dict(
    siren=Siren
)

class CNN():
    def __init__(self, n_objs: int, emb_dim=12, n_pos=1024, n_neg=1024, lr_cnn=3e-4, lr_mlp=3e-5,
                 mlp_name='siren', mlp_hidden_features=256, mlp_hidden_layers=2,
                 key_noise=1e-3, warmup_steps=2000, separate_decoders=True,
                 **kwargs):
        
        self.n_objs, self.emb_dim = n_objs, emb_dim
        self.n_pos, self.n_neg = n_pos, n_neg
        self.lr_cnn, self.lr_mlp = lr_cnn, lr_mlp
        self.warmup_steps = warmup_steps
        self.key_noise = key_noise
        self.separate_decoders = separate_decoders

        # query model
        self.cnn = ResNetUNet(
            n_class=(emb_dim + 1) if separate_decoders else n_objs * (emb_dim + 1),
            n_decoders=n_objs if separate_decoders else 1,
        )

        # key models
        mlp_class = mlp_class_dict[mlp_name]
        mlp_args = dict(in_features=3, out_features=emb_dim,
                        hidden_features=mlp_hidden_features, hidden_layers=mlp_hidden_layers)
        self.mlps = torch.nn.Sequential(*[mlp_class(**mlp_args) for _ in range(n_objs)])


def run():
    model_path = "D:/Akshay_Work/aks_git_repos/surfemb/data/models/tless-2rs64lwh.compact.ckpt"
    device = "cuda:0"

    # load model
    '''model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path)).eval().to(device)  # type: SurfaceEmbeddingModel
    model.freeze()

    print(model)
    with open("surfemb_model.txt", "w") as f:
        f.write(str(model))
    '''



    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="D:/Akshay_Work/aks_git_repos/surfemb/data/models/tless-2rs64lwh.compact.ckpt")
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--detection', action='store_true')
    parser.add_argument('--i', type=int, default=0)
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

    auxs = model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=detection)
    dataset_args = dict(dataset_root=root, obj_ids=obj_ids, auxs=auxs, cfg=cfg)
    
    data = instance.BopInstanceDataset(**dataset_args, pbr=not args.real, test=args.real)
    print()
    obj_coord = data[0]["obj_coord"]
    
    a = obj_coord
    a = np.interp(a, (a.min(), a.max()), (0, 255))
    cv2.imshow("obj_coord", obj_coord[..., :3]*255)
    cv2.waitKey()


if __name__ == "__main__":
    run()