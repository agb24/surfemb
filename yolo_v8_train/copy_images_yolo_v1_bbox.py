import os
import numpy as np
import shutil
import pandas as pd

origin = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_pbr"
or_dirs = os.listdir(origin)
sel_fols = [f"{dir:06d}" for dir in range(1, 9)]
or_dirs = [a for a in or_dirs if a in sel_fols]

pickle_dir = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/pickle_folder"

dest = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_yolo_train"

img_size = (360, 640)

for dir in or_dirs:
    pickle_df = pd.read_pickle(os.path.join(pickle_dir, f"{dir}_data.pkl"))


    i = 0
    uniques = pickle_df["image_id"].unique()

    for uniq in uniques:
        df = pickle_df[ pickle_df["image_id"] == uniq ]
        # Save the bounding boxes
        new_name = df["folder_id"].iloc[i] + "_" + df["image_id"].iloc[i] + ".jpg"

        obj_id = df["obj_id"]
        bbox = np.vstack(df["bbox_visib"])
        save_data = np.vstack([obj_id, 
                              (bbox[:,0]+bbox[:,2]/2)/img_size[1],
                              (bbox[:,1]+bbox[:,3]/2)/img_size[0],
                              bbox[:,2]/img_size[1],
                              bbox[:,3]/img_size[0]
                              ],
                            )
        save_data = save_data.T
        save_data = pd.DataFrame(save_data,
                                 columns = ["obj_id", "bb0", "bb1", "bb2", "bb3"],
                                )
        save_data = save_data.astype({"obj_id":"int32",
                                    "bb0":"float16",
                                    "bb1":"float16",
                                    "bb2":"float16",
                                    "bb3":"float16",}) 


        folders = ["train", "test", "val"]
        import random
        randnum = random.uniform(0,1)
        if randnum < 0.7:
            fol = "train"
        elif (randnum>=0.7) and (randnum<0.85):
            fol = "test"
        elif randnum>=0.85:
            fol = "val"

        dest_txt = os.path.join(dest, "labels", fol)
        dest_img = os.path.join(dest, "images", fol)

        save_data.to_csv(os.path.join(dest_txt, new_name[:-4] + ".txt"),
                         index = False,
                         header=False,
                         sep=' ')

        # Copy the images with new names into the destination
        src = os.path.join(origin, dir, "rgb", str(df["image_id"].iloc[i]) + ".jpg")
        dst = os.path.join(dest_img, new_name)
        shutil.copyfile(src, dst)
