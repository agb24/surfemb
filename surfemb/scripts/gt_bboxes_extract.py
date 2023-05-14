import os
import json
import numpy as np

spath = "/home/ise.ros/akshay_work/datasets/motor/test_primesense"
folders = ["000001", "000002"]
bb_json_path = [os.path.join(spath,f,"scene_gt_info.json")
                for f in folders]
obj_json_path = [os.path.join(spath,f,"scene_gt.json")
                for f in folders]

bboxes = []
obj_ids = []
scene_ids = []
scores = []
times = []
view_ids = []
for i in range(len(bb_json_path)):
    scene_id = i+1

    with open(bb_json_path[i]) as f:
        bb_gt1 = json.load(f)
        print(type(bb_gt1))

    with open(obj_json_path[i]) as f:
        obj_gt1 = json.load(f)
        print(type(obj_gt1))

    for ind, view_id in enumerate(bb_gt1.keys()):
        for j, dic in enumerate(bb_gt1[str(view_id)]):
            a = dic
            if all(ele > 0 for ele in a["bbox_visib"]):
                bboxes += [a["bbox_visib"]]
                obj_ids += [ obj_gt1[str(view_id)][j]["obj_id"] ]
                scene_ids += [scene_id]
                scores += [1]
                times += [2]
                view_ids += [int(view_id)]

path = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/detection_results/motor"
bboxes = np.asarray(bboxes)
save_bboxes = np.zeros(bboxes.shape)
save_bboxes[:,0] = bboxes[:,0]
save_bboxes[:,1] = bboxes[:,1]
save_bboxes[:,2] = bboxes[:,0]+bboxes[:,2]
save_bboxes[:,3] = bboxes[:,1]+bboxes[:,3]

np.save(os.path.join(path, "bboxes.npy"), save_bboxes)
np.save(os.path.join(path, "obj_ids.npy"), np.asarray(obj_ids))
np.save(os.path.join(path, "scene_ids.npy"), np.asarray(scene_ids))
np.save(os.path.join(path, "scores.npy"), np.asarray(scores))
np.save(os.path.join(path, "times.npy"), np.asarray(times))
np.save(os.path.join(path, "view_ids.npy"), np.asarray(view_ids))

