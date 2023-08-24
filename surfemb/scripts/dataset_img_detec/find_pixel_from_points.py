'''
Visualizes Pixel Co-ordinates of all selected points from point cloud.
Gets the Orientation of the particular set of points selected.
'''

import argparse
from pathlib import Path

import math
import cv2
import torch.utils.data
import numpy as np
import open3d as o3d
from matplotlib import cm

ROOT_DIR = "/home/ise.ros/akshay_work/NN_Implementations/surfemb"  #os.path.dirname(os.path.abspath(__file__))
MASK_DIR = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/maskrcnn_train"
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

from sklearn.cluster import KMeans, DBSCAN
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/models/tless_mod-400K-STEPS_2s3iffop.ckpt")
parser.add_argument('--real', default=True)   #action='store_false')
parser.add_argument('--detection', default=True)   #action='store_false')
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
#root = Path("/home/ise.ros/akshay_work/NN_Implementations/surfemb/surfemb/scripts/dataset_img_detec")
root = Path("/home/ise.ros/akshay_work/NN_Implementations/surfemb/data")
cfg = config[dataset]
res_crop = 224


objs, obj_ids = obj.load_objs(root / "bop" / dataset / cfg.model_folder)
renderer = ObjCoordRenderer(objs, res_crop)
assert len(obj_ids) == model.n_objs
surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)
auxs = model.get_infer_auxs(objs=objs, crop_res=res_crop, from_detections=detection)
dataset_args = dict(dataset_root=root / "bop" / dataset, obj_ids=obj_ids, auxs=auxs, cfg=cfg)
if detection:
    assert args.real
    data = detector_crops.DetectorCropDataset(
        **dataset_args, detection_folder= root / Path(f'detection_results') / dataset
    )
else:
    data = instance.BopInstanceDataset(**dataset_args, pbr=not args.real, test=args.real)




def pick_points(geometry, mesh=True):
    if mesh == True:
        pcd = o3d.io.read_point_cloud(geometry)
        pcd = pcd.voxel_down_sample(voxel_size=0.1)  #mesh.sample_points_uniformly(number_of_points=5000)
    elif mesh == False:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(geometry)

    print("----------> Number of Points::: ", len(pcd.points))
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Q ----> Close window + RETURN Curr. pts.")
    print("\n->  Y ----> Align geometry w/ -ve Y-Axis.")
    print("-> K ----> Lock screen ; Go To Select Mode.")
    print("   Mouse Drag ----> Box Selection")
    print("   Ctrl + Clik ----> Polygon Selection")
    print("-> C ----> Crop + select points")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    final_pcd = vis.get_cropped_geometry()    
    vis.destroy_window()
    
    return final_pcd


data_i = 0
while data_i < len(data):
    data_i += 1
    print()
    print('------------ new input -------------')
    inst = data[data_i]
    obj_idx = inst['obj_idx']
    img = inst['rgb_crop']
    K_crop = inst['K_crop']
    obj_ = objs[obj_idx]
    print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')

    geom_path = " "#root / dataset / "models_cad" / f"obj_{obj_ids[obj_idx]:06d}.ply"
    if ".ply" in str(geom_path):
        #geometry = o3d.data.read_triangle_mesh(geom_path)
        final_pcd = pick_points(str(geom_path), mesh=True)
    else:
        geometry = surface_samples[obj_idx]
        final_pcd = pick_points(geometry, mesh=False)
    
    
    print("\n\n\n\n----------> Final Points::: ", len(final_pcd.points))

    # Get "Query" Images and Mask "Logits"
    with utils.timer('forward_cnn'):
        mask_lgts, query_img = model.infer_cnn(img, obj_idx)
    mask_prob = torch.sigmoid(mask_lgts)
    print("query_img,   mask_prob", query_img.shape, mask_prob.shape)
    query_vis = model.get_emb_vis(query_img)
    print("query_vis", query_vis.shape)
    query_norm_img = torch.norm(query_img, dim=-1) * mask_prob
    print("query_norm_img", query_norm_img.shape)
    query_norm_img /= query_norm_img.max()
    print("query_norm_img", query_norm_img.shape)
    cv2.imshow('query_norm', query_norm_img.cpu().numpy())


    # Get the "Keys" of selected vertices 
    final_pcd_np = np.asarray(final_pcd.points)
    verts_norm = (final_pcd_np - obj_.offset) / obj_.scale
    with utils.timer('forward_mlp'):
        keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), obj_idx)  # (N, emb_dim)
    keys_means = keys_verts.mean(dim=0)


    # corr vis
    uv_names = 'xy', 'xz', 'yz'
    uv_slices = slice(1, None, -1), slice(2, None, -2), slice(2, 0, -1)
    uv_uniques = []
    uv_all = ((verts_norm + 1) * (res_crop / 2 - .5)).round().astype(int)
    for uv_name, uv_slice in zip(uv_names, uv_slices):
        view_uvs_unique, view_uvs_unique_inv = np.unique(uv_all[:, uv_slice], axis=0, return_inverse=True)
        uv_uniques.append((view_uvs_unique, view_uvs_unique_inv))
    #print(uv_uniques[0])
    # visualize
    img_vis = img[..., ::-1].astype(np.float32) / 255
    grey = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)


    '''for win_name in (*uv_names, 'dist', 'pose', 'mask_score', 'coord_score', 'keys'):
        cv2.imshow(win_name, np.zeros((res_crop, res_crop)))
    '''


    def get_matching_pixels(query_img):
        # Element-wise multiplication of 'Query' image w/ mask
        query_img = query_img * mask_prob.unsqueeze(-1)
        query_img_flat = query_img.reshape((res_crop*res_crop, 12))
        query_img_flat = query_img_flat.permute((1,0))

        correlations = keys_verts @ query_img_flat
        correlations = torch.softmax(correlations, dim=0)
        maximum = torch.argmax(correlations, dim=1)
        print(maximum.shape)
        
        max_coords = [(math.floor(a.item()/224), a.item()%224) 
                            for a in maximum]

        return max_coords


    def cluster_pts(max_coords):
        #cluster_alg = KMeans(n_clusters=6, random_state=100)
        cluster_alg = DBSCAN(eps=3.5)
        clust_labels = cluster_alg.fit_predict(max_coords)
        return clust_labels

    def show_points(max_coords, clust_labels, uniq_labels, max_cluster):
        colors = cm.rainbow(np.linspace(0,1,len(uniq_labels)))
        colors = colors.astype(np.float16)*255
        colors = colors.astype(np.int16)
        cv2.imshow("IMAGE:", img)
        # Check if label is drawn
        is_lbl = {i:False for i in (uniq_labels)}
        for i,cd in enumerate(max_coords):
            lbl = int(clust_labels[i])
            col = colors[lbl].tolist()[:3]
            if lbl == max_cluster:
                # Y-> Axis 0, X-> Axis 1
                cv2.drawMarker(img, (cd[1],cd[0]), col, cv2.MARKER_CROSS, 3)
                if is_lbl[lbl] == False:
                    cv2.putText(img, str(lbl), (cd[1],cd[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
                    is_lbl[lbl] = True
        cv2.imshow("IMAGE:", img)
        cv2.waitKey(20000)
    
    # Get the max-coords for the Image pixel points
    max_coords = get_matching_pixels(query_img)
    # Get the clusters for each pixel point
    clust_labels = cluster_pts(max_coords)
    # Find Unique Cluster ID with Maximum num of points
    uniq_labels = Counter(clust_labels)    
    max_cluster = [k for k in uniq_labels.keys()
                   if ( uniq_labels[k]==max(uniq_labels.values()) 
                       and k != -1)][0]
    points = [max_coords[i] for i in range(len(max_coords)) 
              if clust_labels[i]==max_cluster]
    
    # Show Pixels
    print("------------> MAX CLUSTER: ", max_cluster)
    show_points(max_coords, clust_labels, uniq_labels, max_cluster)