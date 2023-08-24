'''import os, json
import pandas as pd


path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/test_primesense"
folders = os.listdir(path)
filename = "scene_gt_info.json"

overall_ct = 0
valid_ct = 0

for i,f in enumerate(folders):
    file = os.path.join(path,f,filename)
    with open(file) as fff:
        img_dict = json.load(fff)
    for val in img_dict.values():
        for bbox_dict in val:
            overall_ct += 1
            if -1 not in bbox_dict["bbox_visib"]:
                valid_ct += 1

print(overall_ct)
print(valid_ct)'''

