import os, json
import numpy as np
import pandas as pd
import transforms3d as t3d

# Read the Pose, Obj, BBox data from Pickle Files
'''df_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_planning/all_pose_data"
df_files = [os.path.join(df_path, a) for a in os.listdir(df_path)
            if "6d_pose.pkl" not in a]'''
# Columns: ['folder_id', 'image_id', 'images', 'mask_id', 'masks', 
# 'cam_R_m2c', 'cam_t_m2c', 'obj_id', 'true_obj_id', 'bbox_obj', 'bbox_visib', 
# 'px_count_all', 'px_count_valid', 'px_count_visib', 'visib_fract']

dataset_path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/train_pbr"
folders = os.listdir(dataset_path)
save_df_path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_planning"
save_df_name = "6d_pose.pkl"

yaml_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
5: 5, 6: 6, 7: 10, 8: 11, 9: 12, 10: 13, 
11: 14, 12: 15, 13: 16, 14: 17, 15: 18}

def new():
    root = dataset_path
    save_df = pd.DataFrame()

    overall_gt_list = []

    for fol_id,folder in enumerate(folders):
        # Get the Ground-Truth OBJ ID & Obj-Cam Rot/Transl Matrix
        # for each image
        gt_json="scene_gt.json"
        with open(os.path.join(root, folder, gt_json), "r") as file:
            gt_dict = json.loads(file.read())
        # Get the Ground-Truth obj BBOXES (just 'visib' as well as 'all')
        bb_json="scene_gt_info.json"
        with open(os.path.join(root, folder, bb_json), "r") as ff:
            bb_dict = json.loads(ff.read())
        # Get WORLD to CAMERA orientations
        camera_json="scene_camera.json"
        with open(os.path.join(root, folder, camera_json), "r") as cam_file:
            cam_dict = json.loads(cam_file.read())
        

        # CONCURRENTLY PROCESS both the JSONs
        # Loop through the 1st LEVEL: KEYS are IMAGE_IDS
        for ind,key in enumerate(gt_dict.keys()):
            img_id = key
            # CAMERA_TO_WORLD for each Scene Position
            this_img_cam_dict = cam_dict[key]

            # Merge these lists of 2 dictionaries in exact same order
            # Loop through LIST OF DICTs contained in the 2nd LEVEL
            scene_gt_list = gt_dict[key]
            bb_gt_list = bb_dict[key]
            for this_mask_id in range(len(scene_gt_list)):
                # Get individual dicts with 1. OBJ ID & 2. GT Params
                this_obj_dict = scene_gt_list[this_mask_id]
                this_obj_bb_dict = bb_gt_list[this_mask_id]
                d = {**this_obj_dict, **this_obj_bb_dict,
                     "cam_quat_m2c":[],
                     "folder_name": [],
                     "scene_id": []}
                '''{d[key]:value for (key, value) in this_obj_bb_dict.items()}'''
                
                # TRUE OBJECT ID
                d["true_obj_id"] = this_obj_dict["obj_id"]
                # REASSIGN OBJECT ID TO ITS CONTINUOUS INDEX BASED ON YAML
                d["obj_id"] = [ i for i in yaml_dict.keys()
                        if yaml_dict[i] == this_obj_dict["obj_id"] 
                                ][0]

                d["image_id"] = "{:06d}".format(int(img_id))
                d["mask_id"] = "{:06d}".format(int(this_mask_id))
                
                # Rot Matrix
                rot_mat = this_obj_dict["cam_R_m2c"]
                rot_mat = np.asarray(rot_mat).reshape((3,3))
                # Convert Rotation Matrix to Quaternion: [W X Y Z]
                quat = t3d.quaternions.mat2quat(rot_mat)
                # Store Quaternion Values
                d["cam_quat_m2c"]=quat

                # Save Folder Name
                d["folder_name"] = folder

                # Save Unique Scene ID
                folder_ix = int(d["folder_name"])
                image_ix = int(d["image_id"])
                image_ix = int(image_ix / 25) + 1
                d["scene_id"] = folder_ix * image_ix

                # Save Camera Parameters, and the Cam_to_World Coordinates
                d["cam_K"] = this_img_cam_dict["cam_K"]
                d["cam_R_w2c"] = this_img_cam_dict["cam_R_w2c"]
                d["cam_t_w2c"] = this_img_cam_dict["cam_R_w2c"]

                overall_gt_list.append(d)
    
    print(len(overall_gt_list))
    final_df = pd.DataFrame(overall_gt_list)
    final_df.to_pickle(os.path.join(save_df_path, save_df_name))


def old():
    save_df = pd.DataFrame()
    for i in range(len(df_files)):
        file = df_files[i]
        df = pd.read_pickle(file)
        sub_df = df[['cam_R_m2c', 'cam_t_m2c', 'obj_id', 'true_obj_id',
                    'folder_id', 'image_id',]]
        sub_df = sub_df.reset_index()
        print(sub_df.shape)

        pose_dict = {'cam_quat_m2c':[], 
                    'cam_t_m2c':[],
                    'obj_id':[],
                    'true_obj_id':[],
                    'image_id': [],
                    'folder_id': [],
                    'scene_id': []
                    }
        for ind in range(sub_df.shape[0]):
            # Rot Matrix
            rot_mat = sub_df["cam_R_m2c"].iloc[ind]
            rot_mat = np.asarray(rot_mat).reshape((3,3))
            # Convert Rotation Matrix to Quaternion: [W X Y Z]
            quat = t3d.quaternions.mat2quat(rot_mat)
            # Store Quaternion Values
            pose_dict['cam_quat_m2c'].append(quat)
            # Store the Simulated Scene ID of the Image, 
            # based on unique combo of Folder ID and Image ID
            folder = int(sub_df['folder_id'].iloc[ind])
            image_ix = int((int(sub_df['image_id'].iloc[ind]) + 1) / 25)
            print("---------------------", folder, image_ix)
            scene_id = ( folder * image_ix )
            pose_dict['scene_id'].append(scene_id)

        pose_dict['cam_t_m2c'] = sub_df['cam_t_m2c']
        pose_dict['obj_id'] = sub_df['obj_id']
        pose_dict['true_obj_id'] = sub_df['true_obj_id']
        pose_dict['image_id'] = sub_df['image_id']
        pose_dict['folder_id'] = sub_df['folder_id']

        if i == 0:
            save_df = pd.DataFrame(pose_dict)
        else:
            save_df = pd.concat([save_df, pd.DataFrame(pose_dict)])

    print(save_df.shape, save_df.columns)
    save_df.to_pickle(os.path.join(save_df_path, save_df_name))


if __name__ == "__main__":
    new()


# QUATERNION DISTANCES FORMULA:::
# Cosine Distances (uses the Inner/Dot Product of vectors)
# If a,b are WXYZ Vectors of Quaternions,
# Distance(a, b) = ( 1 - |a . b| )
# --- EXAMPLE: Different Quaternions: 1 - abs(a @ b) == 0.99049   [LARGER DISTANCE]
# --- EXAMPLE: Same Quaternions: 1 - abs(a @ a) == 3.33066 * e-16   [NEARLY ZERO]

# COLUMNS OF THE PICKLE FILE:::::
#    ['cam_R_m2c', 'cam_t_m2c', 'obj_id', 'bbox_obj', 'bbox_visib',
#    'px_count_all', 'px_count_valid', 'px_count_visib', 'visib_fract',
#    'cam_quat_m2c', 'folder_name', 'scene_id', 'true_obj_id', 
#    'image_id', 'mask_id',
#    'cam_K', 'cam_R_w2c', 'cam_R_w2c"]