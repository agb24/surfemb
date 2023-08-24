import os, cv2
import numpy as np
import shutil
import pandas as pd
from imantics import Polygons, Mask

SAVE = "y"

origin = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_pbr"
or_dirs = os.listdir(origin)
sel_fols = [f"{dir:06d}" for dir in range(1, 9)]
or_dirs = [a for a in or_dirs if a in sel_fols]

pickle_dir = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/yolo_v8_train/pickle_folder"

dest = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_yolo_segment"

img_size = (360, 640)

skip_ct = 0

for dir in or_dirs:
    print(dir)
    pickle_df = pd.read_pickle(os.path.join(pickle_dir, f"{dir}_data.pkl"))

    i = 0
    uniques = pickle_df["image_id"].unique()

    for uniq in uniques:
        # Split the dataset
        folders = ["train", "test", "val"]
        import random
        randnum = random.uniform(0,1)
        if randnum < 0.7:
            fol = "train"
        elif (randnum>=0.7) and (randnum<0.85):
            fol = "test"
        elif randnum>=0.85:
            fol = "val"
        # Setup the folder names
        dest_img = os.path.join(dest, "images", fol)
        dest_txt = os.path.join(dest, "labels_bbox", fol)
        dest_mask = os.path.join(dest, "labels_mask", fol)


        # --------------- Load the existing Pickle DF for the Bboxes & Save it
        df = pickle_df[ pickle_df["image_id"] == uniq ]
        new_name = df["folder_id"].iloc[i] + "_" + df["image_id"].iloc[i] 
        # Save the bounding boxes
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
        # Saving BBox data to labels
        if SAVE == "y":
            save_data.to_csv(os.path.join(dest_txt, new_name + ".txt"),
                            index = False,
                            header= False,
                            sep=' ')
        

        # --------------- Get the Binary Masks (combine multiple single mask images), then
        # then convert to Polygons, then save object ids and polygons in txt
        new_mask_name = os.path.join(dest_mask, new_name + ".txt")
        mask_path = os.path.join(origin, df["folder_id"].iloc[i], "mask")
        this_img_masks = [os.path.join(mask_path, a) for a in os.listdir(mask_path) 
                          if (df["image_id"].iloc[i] + "_" in a)]
        this_img_polygons = []
        for i,f in enumerate(this_img_masks):
            img = cv2.imread(f)
            #array = np.ones((100, 100))
            array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            polygons = Mask(array).polygons()
            try:
                act_polygons = polygons[0]
                normalizer = np.tile(np.asarray([640,360]).astype(np.float64),
                                    int(act_polygons.shape[0]/2)
                                    )
                act_polygons = act_polygons / normalizer
                this_img_polygons.append([obj_id.values[i]] + 
                                            act_polygons.tolist())
            except:
                skip_ct += 1
                if skip_ct % 100 == 0:
                    print(f"Skipped one of the masks in {new_name}")
                    print(skip_ct )
                
        if SAVE == "y":
            with open(new_mask_name, "w") as file:
                for poly in this_img_polygons:
                    for item in poly:
                        file.write(f"{item}" + " ")
                    file.write("\n")


        # --------------- Copy the images with new names into the destination
        new_img_name = new_name + ".jpg"
        src = os.path.join(origin, dir, "rgb", str(df["image_id"].iloc[i]) + ".jpg")
        dst = os.path.join(dest_img, new_img_name)
        if SAVE == "y":
            shutil.copyfile(src, dst)

print(skip_ct)