import json
import os

final_list = []

rootpath = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/bop/tlessmod/test_primesense/"

scene_ids = [1,2,3,4]
filenames = ["000001/scene_gt.json",
             "000002/scene_gt.json",
             "000003/scene_gt.json",
             "000004/scene_gt.json"]

for ind,flnam in enumerate(filenames):

    filename = os.path.join(rootpath, flnam)
    scene_id = scene_ids[ind]

    with open(filename, "r") as f:
        scene_gt = json.load(f)


    for k in scene_gt.keys():
        dict_lists = scene_gt[k]

        img_ids = [k for i in range(len(dict_lists))]
        inst_count = [1 for i in range(len(dict_lists))]

        obj_ids = [a["obj_id"] for a in dict_lists]

        scene_ids = [scene_id for i in range(len(dict_lists))]


        for ct in range(len(dict_lists)):
            out_dict = {}
            out_dict["im_id"] = int(img_ids[ct])
            out_dict["inst_count"] = inst_count[ct]
            out_dict["obj_id"] = obj_ids[ct]
            out_dict["scene_id"] = scene_ids[ct]
            final_list.append(out_dict)
    

with open(f"test_targets_bop19.json", "w") as file:
    json.dump(final_list, file)

print()
