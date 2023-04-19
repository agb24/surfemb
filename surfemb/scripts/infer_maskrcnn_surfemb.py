# Using this to fix relative import issues & allow easier debug. 
# Else, must run file as "python -m <file>.py". 
ROOT_DIR = "D:\\Akshay_Work\\aks_git_repos\\surfemb"  #os.path.dirname(os.path.abspath(__file__))
MASK_DIR = "D:\\Akshay_Work\\aks_git_repos\\surfemb\\maskrcnn_train"
import sys
sys.path.append(ROOT_DIR)
sys.path.append(MASK_DIR)

# This is for importing the MaskRCNN Trained model
from maskrcnn_train.pred_imgs import predict_masks

import argparse
from pathlib import Path

import cv2, time, os
from PIL import Image
import torch.utils.data
import numpy as np
import torch
import torchvision


from surfemb import utils
from surfemb.data import obj
from surfemb.data.config import config
from surfemb.data import instance
from surfemb.data import detector_crops
from surfemb.data.renderer import ObjCoordRenderer
from surfemb.surface_embedding import SurfaceEmbeddingModel
from surfemb import pose_est
from surfemb import pose_refine


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="D:\\Akshay_Work\\aks_git_repos\\surfemb\\data\\models\\motor-vlyro4oe-500k-steps.ckpt")
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--res-data', type=int, default=256)
parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--max-poses', type=int, default=50000)
parser.add_argument('--max-pose-evaluations', type=int, default=5000)
parser.add_argument('--no-rotation-ensemble', dest='rotation_ensemble', action='store_false')

args = parser.parse_args()
res_crop = args.res_crop
device = torch.device(args.device)
model_path = Path(args.model_path)
assert model_path.is_file()
model_name = model_path.name.split('.')[0]

dataset = model_name.split('-')[0]
root = Path('data/bop') / dataset
cfg = config[dataset]
res_crop = 224







path = "D:\\Akshay_Work\\datasets\\motor\\train_pbr\\000001\\rgb\\000246.jpg"
#"D:\\Akshay_Work\\datasets\\tless\\test_primesense\\000001\\rgb\\000334.png"
#"D:\\Akshay_Work\\datasets\\motor\\train_pbr\\000003\\rgb\\000189.jpg"
#"D:/Akshay_Work/datasets/motor_dataset_trials/test_img1.jpg"
pil_image = Image.open(path).convert("RGB")
use_maskrcnn_preds = False

pil_image_list = [pil_image]


# load SurfEmb model
model = SurfaceEmbeddingModel.load_from_checkpoint(str(model_path))  # type: SurfaceEmbeddingModel
model.eval()
model.freeze()
model.to(device)
print("\n\n\n\n\n Model Variables::", [k for k,v in vars(model).items() if k not in ["cnn", "mlps"] ])
# Initialize the 3D object files & sample Surface Normals for the given dataset
if (dataset == "tless") or (dataset == "motor"):
    objs, obj_ids = obj.load_objs(root / cfg.model_folder)
    renderer = ObjCoordRenderer(objs, res_crop)
    assert len(obj_ids) == model.n_objs
    surface_samples, surface_sample_normals = utils.load_surface_samples(dataset, obj_ids)
else:
    print("Other dataset not implemented :( \n Exiting..............")
    sys.exit()



def rgb_bbox_affine_reshape(im, bbox, crop_res=224):
    # Dealing with Boolean-type masks
    modify_boolean = False
    if im.dtype == torch.bool:
        modify_boolean = True
    
    if modify_boolean == True:
        im = im.long()
        im = im.permute(1,2,0).cpu().numpy().astype(np.uint8)
    else:
        im = im.permute(1,2,0).cpu().numpy()
    
    crop_scale = 1.2
    offset_scale = 1.
    rgb_interpolation=(cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC)
    max_angle = np.pi
    theta = np.random.uniform(max_angle, max_angle)
    S, C = np.sin(theta), np.cos(theta)
    R = np.array((
        (C, -S),
        (S, C),
    ))
    [top, left, bottom, right] = bbox.numpy()

    cy, cx = (top + bottom) / 2, (left + right) / 2
    size = crop_res / max(bottom - top, right - left) / crop_scale
    M = np.concatenate((R, [[-cx], [-cy]]), axis=1) * size
    r = crop_res
    M[:, 2] += r / 2
    offset = (r - r / crop_scale) / 2 * offset_scale
    M[:, 2] += np.random.uniform(-offset, offset, 2)

    #interp = cv2.INTER_LINEAR if im.ndim == 2 else np.random.choice(rgb_interpolation)
    #rgb_crop = cv2.warpAffine(im, M, (r, r), flags=interp)
    
    interp = cv2.INTER_CUBIC
    rgb_crop = cv2.resize(im, (r,r), fx=5, fy=5, interpolation=interp)

    if modify_boolean == True:
        rgb_crop = rgb_crop > 0

    if rgb_crop.ndim == 2:    
        return torch.from_numpy(rgb_crop).unsqueeze(0)

    return torch.from_numpy(rgb_crop).permute(2,0,1)


def run():

    # -------------------------GET MASK-RCNN PREDICTIONS-------------------------
    # Args for "predict_masks": images (a list of PIL images), 
    #                           mask_thresh=0.5 (thresh for visualization), 
    #                           vis=True
    # Returns: return_masks, return_boxes, return_scores, return_labels
    #          Each is a list of results- one element for each image.
    return_masks, return_boxes, return_scores, return_labels, ret_raw_logits = predict_masks(pil_image_list, vis=False)
    # ---------------------------------------------------------------------------


    # Go through each image & begin predictions
    for i, image in enumerate(pil_image_list):
        image = torchvision.transforms.functional.pil_to_tensor(image)
        # Go through each one of the objects
        obj_ct = 0
        while obj_ct < len(return_labels[i]):

            # --------------- CROP THE IMAGE BASED ON BOUNDING BOXES
            # Get the object details & image crops
            this_obj_idx = return_labels[i][obj_ct] - 1

            
            # Get Bounding Box in xywh format
            final_preds = torchvision.ops.box_convert(return_boxes[i], in_fmt="xyxy", out_fmt="xywh")
            # Get the cropped image of the detected mask
            init_crop_img = torchvision.transforms.functional.crop(image, 
                                                            top=final_preds[obj_ct][1].item(),
                                                            left=final_preds[obj_ct][0].item(),
                                                            height=final_preds[obj_ct][3].item(),
                                                            width=final_preds[obj_ct][2].item())
            print("\n\n\n\ncrop_img", init_crop_img.max(), init_crop_img.min(), 
                                        init_crop_img.shape)   # .shape)   #/ 255
            # Resize Image with proper Affine Transform
            crop_img = rgb_bbox_affine_reshape(init_crop_img, return_boxes[i][obj_ct])
            # Resize image to (224,224) & transfer to cuda:0
            #crop_img = torchvision.transforms.Resize(size=(res_crop, res_crop))(init_crop_img)       
            #crop_img = torch.zeros(3, 224,224)
            #crop_img[:, 0:final_preds[obj_ct][3].item(), 0:final_preds[obj_ct][2].item()] = init_crop_img
            print("crop image shape ----------------->", crop_img.shape)
            crop_img = crop_img.to(device)   
            crop_img = crop_img.to(torch.float)
            

            '''# Crop image by using CENTER of BBox ----- instead of cropping by odd-sized rectangular BBox, then resizing
            # (Old approach likely to give bad mask predictions due to distortion)
            this_bbox = return_boxes[i][obj_ct]
            width = this_bbox[2] - this_bbox[0]
            height = this_bbox[3] - this_bbox[1]
            center_point = [this_bbox[0] + width//2, this_bbox[1] + height//2]
            # Get the final crop image based on BBox Center & Crop Size -> "res_crop" 
            crop_img = torchvision.transforms.functional.crop(image, 
                                                            top=center_point[1].item() - res_crop//2,
                                                            left=center_point[0].item() - res_crop//2,
                                                            height=res_crop,
                                                            width=res_crop)
            crop_img = crop_img.to(device)   
            crop_img = crop_img.to(torch.float)'''
            

            # Get the camera intrinsics: [(fx, 0, cx), (0, fy, cy), (0, 0, 1)]
            K_crop = np.asarray([(1075.65091572, 0, 641.068883438),
                                (0, 1073.90347929, 507.72159802),
                                (0, 0, 1)])
            obj_ = objs[obj_ct]
            print(f'obj_id: ', this_obj_idx)    #{obj_ids[this_obj_idx]}')

            
            # ------ SURFEMB::::: INFERENCE FROM CNN ------
            with utils.timer('forward_cnn'):
                mask_lgts, query_img = model.infer_cnn(crop_img, this_obj_idx) #obj_ids[this_obj_idx], rotation_ensemble=False)
                from pytorch_lightning import Trainer
                from pytorch_lightning.loggers import TensorBoardLogger
                logger = TensorBoardLogger("torchlogs/", name="my_model")
                logger.log_graph(model.infer_cnn, [crop_img, this_obj_idx])

            # --------------- Get the mask probability, queries & normalized queries
            # This is from the inbuilt mask detection from surfemb; doesn't give great pred.
            if use_maskrcnn_preds==False:
                mask_prob = torch.sigmoid(mask_lgts)
            # ALT: This is the mask probability from the pre-trained Mask-RCNN model; better mask pred.
            if use_maskrcnn_preds==True:
                mask_prob_overall_img = return_masks[i][obj_ct,:,:]     #[obj_ct]
                raw_logits_overall_img = ret_raw_logits[i][obj_ct,:,:]
                

                # Cropped mask within BBox
                mask_prob = torchvision.transforms.functional.crop(mask_prob_overall_img, 
                                                            top=final_preds[obj_ct][1].item(),
                                                            left=final_preds[obj_ct][0].item(),
                                                            height=final_preds[obj_ct][3].item(),
                                                            width=final_preds[obj_ct][2].item())
                mask_prob = mask_prob.unsqueeze(0)
                #mask_prob = torchvision.transforms.Resize(size=(res_crop, res_crop))(mask_prob)
                # Resize Image with proper Affine Transform
                mask_prob = rgb_bbox_affine_reshape(mask_prob, return_boxes[i][obj_ct])
                
                '''# Get the final crop image based on BBox Center & Crop Size -> "res_crop" 
                mask_prob = torchvision.transforms.functional.crop(mask_prob_overall_img, 
                                                            top=center_point[1].item() - res_crop//2,
                                                            left=center_point[0].item() - res_crop//2,
                                                            height=res_crop,
                                                            width=res_crop)
                mask_prob = torch.unsqueeze(mask_prob, 0)'''
                


                mask_prob = mask_prob.permute(1,2,0)
                mask_prob = mask_prob.to(device) 
                mask_prob = mask_prob.to(torch.float)   



                mask_lgts = torchvision.transforms.functional.crop(raw_logits_overall_img, 
                                                            top=final_preds[obj_ct][1].item(),
                                                            left=final_preds[obj_ct][0].item(),
                                                            height=final_preds[obj_ct][3].item(),
                                                            width=final_preds[obj_ct][2].item())
                
                '''mask_lgts = torchvision.transforms.functional.crop(raw_logits_overall_img, 
                                                            top=center_point[1].item() - res_crop//2,
                                                            left=center_point[0].item() - res_crop//2,
                                                            height=res_crop,
                                                            width=res_crop)'''

                mask_lgts = mask_lgts.unsqueeze(0)
                #mask_lgts = torchvision.transforms.Resize(size=(res_crop, res_crop))(mask_lgts)
                # Resize Image with proper Affine Transform
                mask_lgts = rgb_bbox_affine_reshape(mask_lgts, return_boxes[i][obj_ct])
                mask_lgts = mask_lgts.squeeze()
                mask_lgts = mask_lgts.to(device)
                '''mask_lgts = torch.special.logit(mask_prob)
                mask_lgts = mask_lgts.permute(1,2,0)
                mask_lgts = mask_lgts.squeeze()'''
                print("final mask lgts shape -------------->", mask_lgts.shape)

            # Visualizing the embedding for the query
            query_vis = model.get_emb_vis(query_img)
            if use_maskrcnn_preds==True:
                squeeze_mask_prob = mask_prob.squeeze() 
                query_norm_img = torch.norm(query_img, dim=-1) * squeeze_mask_prob
            else: 
                query_norm_img = torch.norm(query_img, dim=-1) * mask_prob
            query_norm_img /= query_norm_img.max()

            
            # TODO IS THIS NEEDED? MAYBE YOU CAN REMOVE IT
            # TODO This depends later on CosyPose detections.
            dist_img = torch.zeros(res_crop, res_crop, device=model.device)

            # --------------- Get the vertex samples uniformly from the object surface
            verts_np = surface_samples[this_obj_idx]
            verts = torch.from_numpy(verts_np).float().to(device)
            normals = surface_sample_normals[this_obj_idx]
            verts_norm = (verts_np - obj_.offset) / obj_.scale

            # ------ SURFEMB::::: INFERENCE FROM MLP Key Model ------
            with utils.timer('forward_mlp'):
                keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), this_obj_idx)  # (N, emb_dim)
                logger.log_graph(model.infer_mlp, [torch.from_numpy(verts_norm).float().to(model.device), this_obj_idx])
            keys_means = keys_verts.mean(dim=0)  # (emb_dim,)



            # Visualizing the matches between Pose/Query Norm to ----> Views of the part from XY,YZ,ZX planes.
            uv_names = 'xy', 'xz', 'yz'
            uv_slices = slice(1, None, -1), slice(2, None, -2), slice(2, 0, -1)
            uv_uniques = []
            uv_all = ((verts_norm + 1) * (res_crop / 2 - .5)).round().astype(int)
            for uv_name, uv_slice in zip(uv_names, uv_slices):
                view_uvs_unique, view_uvs_unique_inv = np.unique(uv_all[:, uv_slice], axis=0, return_inverse=True)
                uv_uniques.append((view_uvs_unique, view_uvs_unique_inv))




            

            def refine_pose():
                if current_pose is not None:
                    with utils.timer('refinement'):
                        R, t, score_r = pose_refine.refine_pose(
                            R=current_pose[0], t=current_pose[1], query_img=query_img, keys_verts=keys_verts,
                            obj_idx=this_obj_idx, obj_=obj_, K_crop=K_crop, model=model, renderer=renderer,
                        )
                        trace = np.trace(R @ current_pose[0].T)
                        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                        print(f'refinement angle diff: {np.rad2deg(angle):.1f} deg')
                    debug_pose_hypothesis(R, t)


            down_sample_scale = 3
            def debug_pose_hypothesis(R, t, obj_pts=None, img_pts=None):
                global uv_pts_3d, current_pose
                current_pose = R, t
                render = renderer.render(this_obj_idx, K_crop, R, t)
                render_mask = render[..., 3] == 1.
                pose_img = img_vis.copy()
                pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 + 0.25

                if obj_pts is not None:
                    colors = np.eye(3)[::-1]
                    for (x, y), c in zip(img_pts.astype(int), colors):
                        cv2.drawMarker(pose_img, (x, y), tuple(c), cv2.MARKER_CROSS, 10)
                    uv_pts_3d = obj_pts
                    '''mouse_cb(None, *last_mouse_pos)'''

                cv2.imshow('pose', pose_img)

                poses = np.eye(4)
                poses[:3, :3] = R
                poses[:3, 3:] = t
                pose_est.estimate_pose(
                    mask_lgts=mask_lgts, query_img=query_img.detach(),
                    obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
                    obj_diameter=obj_.diameter, K=K_crop, down_sample_scale=down_sample_scale,
                    visualize=True, poses=poses[None], avg_queries=False
                )


            def estimate_pose():
                print()
                with utils.timer('pnp ransac'):
                    R, t, scores, mask_scores, coord_scores, dist_2d, size_mask, normals_mask = pose_est.estimate_pose(
                        mask_lgts=mask_lgts, query_img=query_img.detach(), down_sample_scale=down_sample_scale,
                        obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
                        obj_diameter=obj_.diameter, K=K_crop,
                        avg_queries=False,  visualize=False
                    )
                if not len(scores):
                    print('no pose')
                    return None
                else:
                    R, t, scores, mask_scores, coord_scores = [a.detach().cpu().numpy() for a in
                                                            (R, t, scores, mask_scores, coord_scores)]
                    best_pose_idx = np.argmax(scores)
                    R_, t_ = R[best_pose_idx], t[best_pose_idx, :, None]
                    debug_pose_hypothesis(R_, t_)
                    return R_, t_
            





            # --------------- Visualization: Cropped Image
            np_img = crop_img.cpu().numpy()
            #np_img = torch.reshape(crop_img, (res_crop, res_crop, 3)).cpu().numpy()
            img_vis = np_img.transpose(1, 2, 0).astype(np.float32)  /  255
            # Show the image, mask & query
            show_time = 10000    ;  print("img_vis shape: ", img_vis.shape)
            cv2.imshow('img', img_vis)
            cv2.waitKey(100)

            if use_maskrcnn_preds==True:
                cv2.imshow('mask_est', torch.sigmoid(mask_lgts).cpu().numpy()) #mask_prob.cpu().detach().numpy()) 
            else:
                cv2.imshow('mask_est', mask_prob.unsqueeze(2).cpu().detach().numpy())
                
            cv2.waitKey(100)
            cv2.imshow('query_vis', query_vis.cpu().numpy())
            cv2.waitKey(100)
            cv2.imshow('query_norm_img', query_norm_img.cpu().detach().numpy())
            cv2.waitKey(100)
            '''for win_name in ["xy", "yz", "zx"]:
                cv2.imshow(win_name, np.zeros((res_crop, res_crop)))'''
            estimate_pose()
            # Formatting OpenCV Windows
            cols = 4
            window_names = ["img", "mask_est", "query_vis", "query_norm_img",
                           'pose', 'mask_score', 'coord_score', 'query_norm' ]  # 'dist', 'xy', 'xz', 'yz',  
            for j, name in enumerate(window_names):
                row = j // cols
                col = j % cols
                cv2.moveWindow(name, 50 + 300 * col, 100 + 300 * row)
            cv2.waitKey(int(show_time/4))

            # Refine the pose
            print("Refining the pose of the aligned part...")
            refine_pose()
            print("Done !!!")
            cv2.waitKey(show_time)

            #time.sleep(30)
            obj_ct += 1



if __name__ == "__main__":
    run()
        


