from __future__ import print_function, division
import os, copy, time, json, yaml
from pathlib import Path
from datetime import datetime as dt
import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

df_save_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_plan_code/all_pose_data"

# CHANGE OBJECT IDs TO CONTINUOUS INDICES, for eg., in range(0,20)
# IN CASE OBEJCTS ARE REMOVED
yaml_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/tless_mod_train.yaml"
with open(yaml_path, "r") as file:
    read_dict = yaml.safe_load(file)
    yaml_dict = read_dict["names"]


class DatasetCreator(Dataset):

    def __init__(self, 
                root="/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_pbr", 
                dir_nums=[a for a in range(1, 9)], 
                transform=None):
        self.root = root
        self.transform = transform
        self.dir_nums = [f"{dir:06d}" for dir in dir_nums]

    def save_csv(self, path=df_save_path):
        print()
        ctr = 0
        # list folders containing images, masks and json
        for i, folder in enumerate(os.listdir(self.root)):
            if folder not in self.dir_nums:
                continue
            
            self.imgs = np.empty(0)
            self.img_ids = np.empty(0)
            self.masks = np.empty(0)
            self.mask_ids = np.empty(0)

            img_folder = os.path.join(self.root, folder, "rgb")
            mask_folder = os.path.join(self.root, folder, "mask")
            all_mask_names = [a for a in os.listdir(mask_folder)]
            img_ids = []
            #all_mask_names = np.array(all_mask_names)
            # Loop through the images
            for j, img_name in enumerate(os.listdir(img_folder)):
                # The img ID is added to the DF
                this_img_id = img_name.split(".")[0]
                img_ids.append(this_img_id)
                # find names of masks that correspond to the image name
                t = time.time()
                masks_wanted = list(filter(
                                lambda x: x.startswith(img_name.split(".")[0]),
                                all_mask_names
                                ))
                mask_names = [mask_folder+"/"+x for x in masks_wanted]
                #print(time.time() - t)
                if j==0:
                    self.imgs = np.full((len(mask_names), 1), 
                                    os.path.join(img_folder, img_name))
                    self.masks = np.array(mask_names)
                    self.masks = np.reshape(self.masks, (-1, 1))
                    self.img_ids = np.full( (len(mask_names),1), this_img_id )
                else:
                    '''# Accounting for cases where there are less than 20 masks 
                    # (usually it is 19 & missing one image)
                    if len(mask_names) < 20:
                        add = copy.deepcopy(mask_names[-1])
                        for i in range(0, 20-len(mask_names)):
                            mask_names.append(add)'''
                    # concatenate the image name and mask names to their paths
                    self.imgs = np.vstack([self.imgs, 
                                        np.full( (len(mask_names),1), 
                                        os.path.join(img_folder, img_name)
                                        )
                                        ])
                    self.masks = np.vstack([self.masks, 
                                            np.reshape(np.asarray(mask_names),
                                                       (-1, 1))
                                            ])
                    self.img_ids = np.vstack([self.img_ids,
                                            np.full( (len(mask_names),1),
                                            this_img_id
                                            )                                                  
                                            ])

                print(i, " - ", j)

            # Create DF of img_ids, paths to images & masks
            old_df = pd.DataFrame({
                    "folder_id": np.full((self.imgs.shape[0], 1), folder).ravel(),
                    "image_id": self.img_ids.ravel(),
                    "images": self.imgs[:,0].ravel(),
                    "mask_id": [Path(a).stem.split("_")[1] 
                                for a in self.masks.ravel()],
                    "masks": self.masks.ravel(),
                    }
                    )
            #df = old_df.explode("masks")
            df = old_df


            # Get the Ground-Truth OBJ ID & Obj-Cam Rot/Transl Matrix
            # for each image
            gt_json="scene_gt.json"
            with open(os.path.join(self.root, folder, gt_json), "r") as file:
                gt_dict = json.loads(file.read())
            # Get the Ground-Truth obj BBOXES (just 'visib' as well as 'all')
            bb_json="scene_gt_info.json"
            bb_df = pd.DataFrame()
            with open(os.path.join(self.root, folder, bb_json), "r") as ff:
                bb_dict = json.loads(ff.read())
            
            overall_gt_list = []
            # CONCURRENTLY PROCESS both the JSONs
            # Loop through the 1st LEVEL: KEYS are IMAGE_IDS
            for ind,key in enumerate(gt_dict.keys()):
                img_id = key

                # Merge these lists of 2 dictionaries in exact same order
                # Loop through LIST OF DICTs contained in the 2nd LEVEL
                scene_gt_list = gt_dict[key]
                bb_gt_list = bb_dict[key]
                for this_mask_id in range(len(scene_gt_list)):
                    # Get individual dicts with 1. OBJ ID & 2. GT Params
                    this_obj_dict = scene_gt_list[this_mask_id]
                    this_obj_bb_dict = bb_gt_list[this_mask_id]
                    d = {**this_obj_dict, **this_obj_bb_dict}
                    '''{d[key]:value for (key, value) in this_obj_bb_dict.items()}'''
                    
                    # TRUE OBJECT ID
                    d["true_obj_id"] = this_obj_dict["obj_id"]
                    # REASSIGN OBJECT ID TO ITS CONTINUOUS INDEX BASED ON YAML
                    d["obj_id"] = [ i for i in yaml_dict.keys()
                            if yaml_dict[i] == this_obj_dict["obj_id"] 
                                  ][0]

                    d["image_id"] = "{:06d}".format(int(img_id))
                    d["mask_id"] = "{:06d}".format(int(this_mask_id))
                    
                    overall_gt_list.append(d)
            
            gt_df = pd.DataFrame(overall_gt_list)
                
            
            # Merge the DFs based on image IDs
            final_df = df.merge(gt_df, how="inner", on=["image_id", "mask_id"])
            final_df.to_pickle(os.path.join(path, f"{folder}_data.pkl"))

            ctr += 1; print("------------>", final_df.shape)
        #self.sample = {"image": self.imgs, "mask": self.masks}
        #print(self.imgs.shape, self.masks.shape)


if __name__ =="__main__":
    s = DatasetCreator()
    s.save_csv()