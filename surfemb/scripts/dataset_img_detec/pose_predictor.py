ROOT_DIR = "/home/ise.ros/akshay_work/NN_Implementations/surfemb"  #os.path.dirname(os.path.abspath(__file__))
MASK_DIR = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/maskrcnn_train"
ROS_PY_DIR = "/opt/ros/noetic/lib/python3/dist-packages"
import sys
sys.path.append(ROOT_DIR)
sys.path.append(MASK_DIR)
sys.path.append(ROS_PY_DIR)

import argparse
from pathlib import Path

import time, os, random
import yaml 
import cv2
import torch.utils.data
import numpy as np

import rospy
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import Vector3, Quaternion
from geometry_msgs.msg import Transform, TransformStamped, WrenchStamped
import tf2_ros
from tf2_msgs.msg import TFMessage
import transforms3d as t3d

from surfemb import utils
from surfemb.data import obj
from surfemb.data.config import config
from surfemb.data import instance
from surfemb.data import detector_crops
from surfemb.data.renderer import ObjCoordRenderer
from surfemb.surface_embedding import SurfaceEmbeddingModel
from surfemb import pose_est
from surfemb import pose_refine
from maskrcnn_train.pred_imgs_saver import run_and_save as maskrcnn_run_save


current_pose = None

class PosePredictor():
    def __init__(self, dataset_name="tless"):
        self.dataset_name = dataset_name
        if self.dataset_name == "tless":
            model_name = "tless-2rs64lwh.compact.ckpt"
        elif self.dataset_name == "motor":
            model_name = "motor-vlyro4oe-500k-steps.ckpt"
        else:
            print("DATASET NOT IMPLEMENTED. EXITING ........")
            quit()
        
        # Initializations
        model_root = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/data/models"
        self.model_path = Path(model_root) / model_name
        # Get the name of the dataset
        self.dataset = self.model_path.name.split('-')[0]
        # Other args
        self.detection = True
        self.real = True
        self.device = torch.device("cuda:0")
        
        # load the model
        self.model = SurfaceEmbeddingModel.load_from_checkpoint(Path(self.model_path))
        self.model.eval()
        self.model.freeze()
        self.model.to(self.device)
        # ROOT OF THE DATASET IMAGES AND RESULTS FOLDER
        self.root = Path("/home/ise.ros/akshay_work/NN_Implementations/surfemb/surfemb/scripts/dataset_img_detec")
        self.cfg = config[self.dataset]
        self.res_crop = 224

        # Load objects, renderer
        self.objs, self.obj_ids = obj.load_objs(self.root / self.dataset / self.cfg.model_folder)
        self.renderer = ObjCoordRenderer(self.objs, self.res_crop)
        assert len(self.obj_ids) == self.model.n_objs
    

    def get_pose(self):
        return_dict_list = []
        res_crop = self.res_crop
        model = self.model
        objs = self.objs

        device = self.device
        detection = self.detection
        renderer = self.renderer
        obj_ids = self.obj_ids
        root = self.root
        data_i = 0
        

        # -------- Run the Mask-RCNN Predictor --------
        bop_test_path= self.root / f"{self.dataset}/test_primesense"
        detec_save_path= self.root / f"detection_results/{self.dataset}"
        boxes, obj_ids = maskrcnn_run_save(str(bop_test_path), str(detec_save_path))
        # -------- Run the Mask-RCNN Predictor --------

        # -------- Init dataloader-image loader
        self.surface_samples, self.surface_sample_normals = utils.load_surface_samples(self.dataset, obj_ids)
        auxs = self.model.get_infer_auxs(objs=objs, crop_res=self.res_crop, from_detections=self.detection)
        dataset_args = dict(dataset_root=self.root / self.dataset, obj_ids=obj_ids, auxs=auxs, cfg=self.cfg)
        if self.detection:
            self.data = detector_crops.DetectorCropDataset(
                **dataset_args, detection_folder= self.root / Path(f'detection_results') / self.dataset
            )
        else:
            self.data = instance.BopInstanceDataset(**dataset_args, pbr=not self.real, test=self.real)
        data = self.data
        surface_samples, surface_sample_normals = self.surface_samples, self.surface_sample_normals


        # initialize opencv windows
        cols = 4
        window_names = 'img', 'mask_est', \
                        'query_norm', 'keys', 'pose', \
                        'mask_score', 'coord_score'
        for j, name in enumerate(window_names):
            row = j // cols
            col = j % cols
            cv2.imshow(name, np.zeros((res_crop, res_crop)))
            cv2.moveWindow(name, 100 + 300 * col, 100 + 300 * row)

        print()
        print('With an opencv window active:')
        print("press 'd' for next prediction")
        print("press 'q' to quit.")
        while data_i < len(data):
            print()
            print('------------ new input -------------')
            inst = data[data_i]
            obj_idx = inst['obj_idx']
            img = inst['rgb_crop']
            K_crop = inst['K_crop']
            obj_ = objs[obj_idx]
            print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')

            with utils.timer('forward_cnn'):
                mask_lgts, query_img = model.infer_cnn(img, obj_idx)

            mask_prob = torch.sigmoid(mask_lgts)
            print("query_img,   mask_prob", query_img.shape, mask_prob.shape)
            query_vis = self.model.get_emb_vis(query_img)
            print("query_vis", query_vis.shape)
            query_norm_img = torch.norm(query_img, dim=-1) * mask_prob
            print("query_norm_img", query_norm_img.shape)
            query_norm_img /= query_norm_img.max()
            print("query_norm_img", query_norm_img.shape)
            cv2.imshow('query_norm', query_norm_img.cpu().numpy())

            dist_img = torch.zeros(res_crop, res_crop, device=model.device)

            verts_np = surface_samples[obj_idx]
            verts = torch.from_numpy(verts_np).float().to(device)
            normals = surface_sample_normals[obj_idx]
            verts_norm = (verts_np - obj_.offset) / obj_.scale
            with utils.timer('forward_mlp'):
                keys_verts = model.infer_mlp(torch.from_numpy(verts_norm).float().to(model.device), obj_idx)  # (N, emb_dim)
            keys_means = keys_verts.mean(dim=0)  # (emb_dim,)

            if not detection:
                coord_img = torch.from_numpy(inst['obj_coord']).to(device)
                key_img = model.infer_mlp(coord_img[..., :3], obj_idx)
                key_mask = coord_img[..., 3] == 1
                keys = key_img[key_mask]  # (N, emb_dim)
                key_vis = model.get_emb_vis(key_img, mask=key_mask, demean=keys_means)

            # visualize
            img_vis = img[..., ::-1].astype(np.float32) / 255
            grey = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)

            for win_name in ('pose', 'mask_score', 'keys'):
                cv2.imshow(win_name, np.zeros((res_crop, res_crop)))

            cv2.imshow('img', img_vis)
            cv2.imshow('mask_est', torch.sigmoid(mask_lgts).cpu().numpy())

            global current_pose
            current_pose = None
            down_sample_scale = 3
            def debug_pose_hypothesis(R, t, obj_pts=None, img_pts=None):
                global current_pose
                current_pose = R, t
                render = renderer.render(obj_idx, K_crop, R, t)
                render_mask = render[..., 3] == 1.
                pose_img = img_vis.copy()
                pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 + 0.25

                cv2.imshow('pose', pose_img)

                poses = np.eye(4)
                poses[:3, :3] = R
                poses[:3, 3:] = t
                pose_est.estimate_pose(
                    mask_lgts=mask_lgts, query_img=query_img,
                    obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
                    obj_diameter=obj_.diameter, K=K_crop, down_sample_scale=down_sample_scale,
                    visualize=True, poses=poses[None],
                )


            def estimate_pose():
                print()
                with utils.timer('pnp ransac'):
                    R, t, scores, mask_scores, coord_scores, dist_2d, size_mask, normals_mask = pose_est.estimate_pose(
                        mask_lgts=mask_lgts, query_img=query_img, down_sample_scale=down_sample_scale,
                        obj_pts=verts, obj_normals=normals, obj_keys=keys_verts,
                        obj_diameter=obj_.diameter, K=K_crop,
                    )
                if not len(scores):
                    print('no pose')
                    return None
                else:
                    R, t, scores, mask_scores, coord_scores = [a.cpu().numpy() for a in
                                                            (R, t, scores, mask_scores, coord_scores)]
                    best_pose_idx = np.argmax(scores)
                    R_, t_ = R[best_pose_idx], t[best_pose_idx, :, None]
                    debug_pose_hypothesis(R_, t_)
                    return R_, t_

            # ESTIMATING THE POSE
            print("Estimating Pose ::: ")
            append_pose = estimate_pose()
            print("Pose done!!!")
            # REFINING IMAGE
            if current_pose is not None:
                print("Refining Pose ::: ")
                with utils.timer('refinement'):
                    R, t, score_r = pose_refine.refine_pose(
                        R=current_pose[0], t=current_pose[1], query_img=query_img, keys_verts=keys_verts,
                        obj_idx=obj_idx, obj_=obj_, K_crop=K_crop, model=model, renderer=renderer,
                    )
                    trace = np.trace(R @ current_pose[0].T)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    print(f'refinement angle diff: {np.rad2deg(angle):.1f} deg')
                debug_pose_hypothesis(R, t)
                append_pose = (R, t)
                print("Refinement done!!!")

            return_dict_list.append(
                            {"data_id": data_i,
                            "obj_id": obj_ids[obj_idx],
                            "pose": append_pose
                            })
            data_i = data_i + 1
            #time.sleep(3)
            cv2.waitKey(1000)

            '''while True:
                key = cv2.waitKey()
                if key == ord("q"):
                    # DELETE ALL THE RECORDED DATA
                    folders = ["detection_results/motor", 
                                "detection_results/tless",
                                "motor/test_primesense/000001/rgb", 
                                "tless/test_primesense/000001/rgb"]
                    folders = [root / f for f in folders]
                    for folder in folders:
                        for file in Path.iterdir(folder):
                            Path.unlink(folder / file)
                    return return_dict_list
                    #quit()

                if key == ord("d"):
                    data_i = (data_i + 1) % len(data)
                    break'''
        


        # DELETE ALL THE RECORDED DATA
        folders = ["detection_results/motor", 
                    "detection_results/tless",
                    "motor/test_primesense/000001/rgb", 
                    "tless/test_primesense/000001/rgb"]
        folders = [root / f for f in folders]
        for folder in folders:
            for file in Path.iterdir(folder):
                os.remove(str(folder / file))  #Path.unlink(folder / file)
        #time.sleep(2)
        return return_dict_list, boxes
            
        

class StartListener():
    def __init__(self):
        self.cb_signal = False
    def callback(self, msg):
        self.cb_signal = msg.data
    def listen(self):
        rospy.Subscriber('pose_est_start', Bool, self.callback)
        rate = rospy.Rate(100)
        print("Waiting for signal...")
        while not rospy.is_shutdown():
            if self.cb_signal == True:
                print("RECEIVED SIGNAL!!!")
                self.cb_signal = False
                return True 
            rate.sleep()


def projection_matrix(K, w, h, near=10., far=10000.):  # 1 cm to 10 m
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    #view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[:2, :3] = K[:2, :3]
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    def orthographic_matrix(left, right, bottom, top, near, far):
        return np.array((
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ))
    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:
    orth = orthographic_matrix(-.5, w - .5, -.5, h - .5, near, far)
    return orth @ persp @ view


def publish_transform(transform_dict, bbox):
    # "transform_dict" has keys::: data_id, obj_id, pose
    obj_id = transform_dict["obj_id"]
    obj_frame_name = f"obj_{obj_id}"
    (w,d) = (bbox[2]-bbox[0])/2, (bbox[3]-bbox[1])/2
    bbox_center_pos = ( bbox[0]+w, bbox[1]+d);  #print(bbox_center_pos)

    '''
    Applying the camera projection matrix:::: get translation in world coords --->
    THESE ARE THE CAMERA PARAMETERS::::
    -----------------VV IMPORTANT::::: pixel_size == 2e-6 m!!!! (0.002 mm)-------------------
    -----------------This is used to convert the translation from pixel units to m!!!!-----------------
    distortion_model: "plumb_bob"
    D: [0.0, 0.0, 0.0, 0.0, 0.0]
    K: [264.546630859375, 0.0, 319.4079895019531, 0.0, 264.546630859375, 178.61087036132812, 0.0, 0.0, 1.0]
    R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    P: [264.546630859375, 0.0, 319.4079895019531, 0.0, 0.0, 264.546630859375, 178.61087036132812, 0.0, 0.0, 0.0, 1.0, 0.0]
    '''
    transl_est = transform_dict["pose"][1];    #print("-----------------> EST TRANSLATION", transl_est)
    
    '''# ADDING THE BBOX CENTER VALUES to the translations
    transl_est[0] = bbox_center_pos[0] * 0.117 / (367-232)
    transl_est[1] = bbox_center_pos[1] * 0.167 / (352-233)'''
    
    transl_est = [a/1000 for a in transl_est]

    rot_matrix = transform_dict["pose"][0]
    np_rot_matrix = np.asarray(rot_matrix)         
    rot_est_quat = t3d.quaternions.mat2quat(rot_matrix)
    rot_est_euler = t3d.euler.quat2euler(rot_est_quat);   #print("-----------------> EST EULER ROT", rot_est_euler)

    pub = rospy.Publisher("/tf", TFMessage, queue_size = 10)
    rate = rospy.Rate(100)
    ctr = 0
    while ctr < 2:      #not rospy.is_shutdown():
        obj_transf = TFMessage([TransformStamped(header=Header(stamp=rospy.Time.now(),
                                                            frame_id="zed2i_left_camera_optical_frame"),
                                            child_frame_id=obj_frame_name,
                                            transform=Transform(translation=Vector3(*[a for a in transl_est]),
                                                                rotation=Quaternion(rot_est_quat[1],
                                                                                    rot_est_quat[2],
                                                                                    rot_est_quat[3],
                                                                                    rot_est_quat[0]
                                                                                    ))
                                                )
                            ])
        pub.publish(obj_transf)
        rate.sleep()
        ctr += 1


if __name__ == "__main__":
    # Initialize the Pose Predictor
    predictor = PosePredictor()
    rospy.init_node('pose_listener', anonymous=True)
    # Init and start the Listener
    listener = StartListener()
    while not rospy.is_shutdown():
        listener.listen()
        # Predict the pose of the objects in the saved images
        ret_dict_list, boxes = predictor.get_pose()
        # Say that the predictions are done
        c = 0
        while c < 5:
            start_pub = rospy.Publisher("/pose_est_start", Bool, queue_size=10)
            start_pub.publish(Bool(data=False))
            rospy.sleep(0.5)
            c += 1
        # Publish the object TF positions
        print([(d['data_id'],d['obj_id']) for d in ret_dict_list])     
        if ret_dict_list != []:
            ctr = 0
            while ctr < 300:
                #print(ctr)
                for i, dicter in enumerate(ret_dict_list):
                    publish_transform(dicter, boxes[i])
                ctr += 1;    #print(ctr)
            ret_dict_list = []
        else:
            print("Done publishing.")