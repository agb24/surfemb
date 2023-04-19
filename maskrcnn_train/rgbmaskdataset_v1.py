from __future__ import print_function, division
import os, copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
from datetime import datetime as dt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import torchviz_utils.transforms as T

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cpu")

pickle_folder = "D:/Akshay_Work/aks_git_repos/surfemb/maskrcnn_train/pickle_folder"

dataset_stats = {"means":  [104.77679889403292, 106.06202152777777, 97.26200198045268],
                "stdevs":  [60.34385930804503, 58.53414827616004, 61.96202448674674]}

class RgbMasksDataset(Dataset):

    def __init__(self, root=pickle_folder, dir_nums=[0], transforms=None):
        """
        Args:
            root (string): Directory with all the images and masks.
            dir_nums (int): The ID of folders to be used for training.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Return from __getitem__ :
            PIL image (H,W)
            target (dict): Containing these keys:
                [----> N is number of Bounding Boxes / Masks]
                boxes (FloatTensor[N,4])
                labels (Int64Tensor[N])
                image_id (Int64Tensor[1])
                area (Tensor[N])
                iscrowd (UInt8Tensor[N])
                masks (UInt8Tensor[N, H, W])
            REF: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """
        self.root = root; print(root)
        self.transforms = transforms
        # Read the training dataset path & GT BBox details
        # remove any rows which don't have a valid mask
        all_pickle = os.listdir(root)
        all_data_df = pd.DataFrame()
        for i, dir in enumerate(dir_nums):
            file = [a for a in all_pickle if f"{dir:06d}" in a]
            df = pd.read_pickle(os.path.join(root, file[0]))
            if i == 0:
                all_data_df = df
            else:
                all_data_df = pd.concat([all_data_df, df], axis=0, ignore_index=True)
        #all_data_df = all_data_df.reset_index(drop=True)
        
        # ADDITIONAL CONDITION:
        # Isolate the bounding boxes and filter them by removing width < 0 or height < 0 
        bboxes = np.asarray([np.asarray(a) for a in all_data_df["bbox_visib"]])
        ind_mod = np.argwhere((bboxes[:,2] > 0.0) & (bboxes[:,3] > 0.0))
        ind_mod = ind_mod.reshape((-1,)).tolist()
        # Keep only rows which satisfy the condition width > 0 and height > 0
        filter_data_df = all_data_df.filter(items = ind_mod, axis = 0)
        # Filter & remove all invalid rows with zero valid pixel counts
        filter_data_df = filter_data_df[filter_data_df["px_count_visib"] > 0.0]
        self.filter_data_df = filter_data_df

        self.sel_ids = sorted(self.filter_data_df["image_id"].unique())
        self.all_folder_ids = self.filter_data_df["folder_id"].unique()
        

    def __len__(self):
        #return len(self.filter_data_df["image_id"].unique())
        counts_by_folder = self.filter_data_df.groupby("folder_id")["image_id"].nunique()
        return counts_by_folder.sum()


    def __getitem__(self, idx):
        """
        param: idx -> Image ID
        """

        sel_ids = self.sel_ids  #[a for a in range(0, 1000)]
        # Get the data for the images of interest only
        id_sel = int(idx%1000)
        id_fol = self.all_folder_ids[ int(idx/1000)-1 ]
        sel_img_id = "{:06d}".format( int(sel_ids[id_sel]) )
        sel_fol_id = "{:06d}".format( int(id_fol) )
        train_df = self.filter_data_df[(self.filter_data_df["image_id"] == sel_img_id)
                                       & 
                                       (self.filter_data_df["folder_id"] == sel_fol_id)]


        labels = [ int(a) for a in train_df["obj_id"] ]#.tolist() ] ; #print("------------->", [type(a) for a in labels])
        labels = torch.as_tensor(labels, dtype=torch.int64, device=device) ; #print("------------->", train_df.shape)  #.iloc[0]))  ["image_id"]
        image_ids = int(train_df["image_id"].iloc[0]) ; #print("------------->", image_ids)
        image_ids = torch.as_tensor([image_ids], dtype=torch.int64, device=device)
        imgs = train_df["images"].tolist()
        # IMAGE to be returned
        image = Image.open(imgs[0]).convert("RGB")

        crowded_bools = np.asarray(train_df["visib_fract"])#.tolist())
        crowded_bools = crowded_bools < 0.75
        crowded_ints = 1*crowded_bools
        is_crowded = torch.as_tensor(crowded_ints, 
                                    dtype=torch.uint8,
                                    device=device)

        mask_list = [np.asarray( Image.open(a) )
                        for a in train_df["masks"]#.tolist()
                    ]
        mask_list_norm = [(a-a.min())/np.ptp(a) for a in mask_list]
        masks = torch.as_tensor(mask_list_norm,
                                dtype=torch.uint8,
                                device=device
                                )
        # EXTRACT all the required values from the filtered dataframe
        boxes = np.asarray(train_df["bbox_visib"].tolist())   
        #print("-------------> BBox shape", boxes)
        bb = copy.deepcopy(boxes)

        # BBoxes are in format X, Y, Width, Height; where X,Y is Top-Left of BB
        # Converting it to X_min, Y_min, X_max, Y_max 
        # TODO tolist() Change -boxes- and -areas- below
        boxes = [boxes[:,0], boxes[:,1], 
                boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]]
        '''boxes = [boxes[0], boxes[1], 
                boxes[0]+boxes[2], boxes[1]+boxes[3]]'''
        boxes = np.asarray(boxes).T; #print("-------------> BBox shape", boxes.shape)

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]);  #print("------------->Area & shape", areas.shape) 
        '''areas = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])'''

        if(np.any(areas < 0, axis=0)): print(f"ERROR IN Image ID: {image_ids}")  #, Original: {bb}, BBoxes: {boxes}, Areas: {areas}")
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)

        # GROUND TRUTH values to be returned
        #mask = Image.open(self.masks[idx])
        target = {"boxes": boxes,
                  "labels": labels, 
                  "image_id": image_ids,
                  "area": torch.as_tensor(areas, device=device),
                  "iscrowd": is_crowded,
                  "masks": masks, 
                  }
        '''transf = transforms.Compose([transforms.PILToTensor(),
                        transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize(dataset_stats["means"],
                                            dataset_stats["stdevs"])])
        image = transf(image)'''
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


class TensorTransforms(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self,sample):
        image = sample
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image)
        image = image.transpose((2, 0, 1))

        return image

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    transformed_dataset = RgbMasksDataset(pickle_folder, 
                            dir_nums = [i for i in range(0,15) if i not in [1,3]], 
                            transforms=get_transform(train=True))
    t1 = dt.now()
    transformed_dataset.__getitem__(idx=10520)
    print("@@@@@@@@@@@@@@@@@@@@@@ Get item took", dt.now() - t1)
    '''print(len(transformed_dataset))
    rs , gs, bs = [], [], []
    # Get stats of images
    for i in range(0, 100):
        a = transformed_dataset.__getitem__(i)
        print(a[0])
        img = np.asarray(a[0])
        rs.append(img[:,:,0])
        gs.append(img[:,:,1])
        bs.append(img[:,:,2])
        print(i)
    rs = np.asarray(rs); gs = np.asarray(gs); bs = np.asarray(bs)
    means = [rs.mean(), gs.mean(), bs.mean()]   ; print("Means: ", means)
    stds = [rs.std(), gs.std(), bs.std()]   ; print("STDEVs: ", stds)
    print()'''

