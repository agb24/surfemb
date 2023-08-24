import os, copy
import numpy as np
import pandas as pd
import open3d as o3d

import pytransform3d.rotations as py3drot
import pytransform3d.transformations as py3dtr
import pytransform3d.transform_manager as py3d_transf_mgr
import pytransform3d.visualizer as py3dviz
import pytransform3d.coordinates as py3dcoord
from pytransform3d.batch_rotations import norm_vectors

SIZE = 50

def get_viz_frames(frame, label, size=SIZE, order=[0,1,2], extrinsic=True):
    """
    frame: dict with {"xyz":[], "rpy":[]}
    Returns: 4x4 Rotation matrix and the 'Frame' object for viz
    """
    frame_3x3 = py3drot.matrix_from_euler(frame["rpy"],
                                              i=order[0],j=order[1],k=order[2],
                                              extrinsic=extrinsic)
    frame_4x4 = np.zeros((4,4))
    frame_4x4[:3,:3] += frame_3x3
    try:
        frame_4x4[:,3] += np.asarray(frame["xyz"]+[1])
    except:
        pass
    
    frame_quat = py3drot.quaternion_from_matrix(frame_3x3)
    frame_viz = py3dviz.Frame(frame_4x4, label=label, s=size)

    return frame_4x4, frame_quat, frame_viz



def test(ind = 2590):
    pi = np.pi
    # CREATE FIGURE
    fig = py3dviz.Figure("viz_win", width=1080, height=720)
    # TRANSFORM MANAGER
    transf_mgr = py3d_transf_mgr.TransformManager()
    
    
    # Read pose samples from PKL ::::
    # COLUMNS ARE ::::
    #   ['cam_R_m2c', 'cam_t_m2c', 'obj_id', 'bbox_obj', 'bbox_visib',
    #    'px_count_all', 'px_count_valid', 'px_count_visib', 'visib_fract',
    #    'cam_quat_m2c', 'folder_name', 'scene_id', 'true_obj_id', 
    #    'image_id', 'mask_id',
    #    'cam_K', 'cam_R_w2c', 'cam_R_w2c']
    root_path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/models_cad"        
    pose_df = pd.read_pickle("/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_planning/6d_pose.pkl")
    filter_pose_df = pose_df.iloc[ind]


    # CREATE BASE FRAME + Visualizer
    base_frame = {"xyz":[0,0,0], "rpy":[0,0,0]}
    # Get 'Frame' for visualization
    base_frame_4x4, base_frame_quat, base_frame_viz = get_viz_frames(base_frame,
                                                                     "base_frame")
    fig.add_geometry(base_frame_viz.geometries[0])
    # Create Base Plane
    fig.plot_plane(normal=np.asarray([0,0,1]), 
                   point_in_plane=np.asarray([0,0,0]),
                   s=1000,
                   c=[0.5,0,0.5])


    # CREATE CAMERA
    # Set the POSE of the Camera in world co-ordinates
    world_to_cam_pose_3x3 = np.asarray(filter_pose_df["cam_R_w2c"]).reshape((3,3))
    world_to_cam_transl = np.asarray(filter_pose_df["cam_t_w2c"]).reshape((3,))
    world_to_cam_4x4 = np.eye(4)
    world_to_cam_4x4[:3,:3] = world_to_cam_pose_3x3
    world_to_cam_4x4[:3,3] = world_to_cam_transl
    transf_mgr.add_transform("world", "camera", world_to_cam_4x4)
    # Reverse transform lookup
    cam2world_4x4 = transf_mgr.get_transform("camera", "world")
    intrinsic = np.asarray([264.546630859375, 0.0, 319.4079895019531,
                            0.0, 264.546630859375, 178.61087036132812,
                            0.0, 0.0, 1.0]).reshape((3,3))
    # Create Camera Object + Frame Object
    camera = py3dviz.Camera(M=intrinsic, 
                            cam2world=cam2world_4x4,
                            virtual_image_distance=SIZE,
                            sensor_size=[640,480])
    cam_frame = py3dviz.Frame(cam2world_4x4, label="camera_frame", s=SIZE)
    # Add Camera to Figure
    fig.add_geometry(camera.geometries[0])
    fig.add_geometry(cam_frame.geometries[0])
    

    # CREATE MESH AND REFERENCE AXIS
    # Read the Mesh
    obj_id = filter_pose_df["true_obj_id"]
    mesh_path = os.path.join(root_path, f"obj_{obj_id:06d}.ply")
    # Get Transformation Matrix for this Object
    mesh_to_cam_pose_3x3 = np.asarray(filter_pose_df["cam_R_m2c"]).reshape((3,3))
    mesh_to_cam_pose_transf = np.asarray(filter_pose_df["cam_t_m2c"]).reshape((3,))
    mesh2cam_4x4 = np.eye(4)
    mesh2cam_4x4[:3,:3] = mesh_to_cam_pose_3x3
    mesh2cam_4x4[:3,3] = mesh_to_cam_pose_transf 
    # Add Transformation
    transf_mgr.add_transform("mesh", "camera", mesh2cam_4x4)
    # Add the Mesh Geometry to the Figure
    world2mesh_A2B = transf_mgr.get_transform("world","mesh")
    mesh_geom = py3dviz.Mesh(mesh_path,
                             A2B=world2mesh_A2B, 
                             s=[1,1,1], c=[0,0.5,0.5])
    mesh_frame = py3dviz.Frame(world2mesh_A2B, label="mesh_frame", s=SIZE)
    fig.add_geometry(mesh_geom.geometries[0])
    fig.add_geometry(mesh_frame.geometries[0])
    BBOX_TOP = mesh_geom.geometries[0].get_axis_aligned_bounding_box().max_bound[-1]


    # Orient Z+ve Axis of the Gripper
    gripper_frame = {"xyz":[world2mesh_A2B[0,3],world2mesh_A2B[1,3],BBOX_TOP+50], 
                     "rpy":[pi,0,0]}
    gripper_frame4x4, gripper_frame_quat, \
                            gripper_frame_viz = get_viz_frames(gripper_frame,
                                                               "gripper")
    fig.add_geometry(gripper_frame_viz.geometries[0])
    


    def animate_callback(step, gripper_frame_viz, z_flattened, x_flattened):
        z_rot = z_flattened[step]
        x_rot = x_flattened[step]
        
        # Set the Frame orientation
        new_frame = {"xyz":[world2mesh_A2B[0,3],world2mesh_A2B[1,3],BBOX_TOP+50], 
                     "rpy":[x_rot, 0, z_rot]}
        new_frame4x4, new_frame_quat, \
                    new_frame_viz = get_viz_frames(new_frame,
                                                   "new_frame", 
                                                   order=[0,1,2],
                                                   extrinsic=False)
        gripper_frame_viz.set_data(new_frame4x4)
        import time; time.sleep(0.005)
        return gripper_frame_viz


    # Sample Rotations every 10 Degrees, for a range of 180 deg. 
    z_rotations = np.linspace(-np.pi/2, np.pi/2, 18)
    # Sample Rotations every 5 Degrees, for a range of 90 deg.
    x_rotations = np.linspace(np.pi/2, np.pi, 18)
    z_samples, x_samples = np.meshgrid(z_rotations, x_rotations)
    z_flattened = z_samples.flatten()
    x_flattened = x_samples.flatten()

    fig.animate(animate_callback, 18*18,
                fargs=(gripper_frame_viz, z_flattened, x_flattened), 
                loop=False)
    fig.view_init(azim=45, elev=45) 
    fig.set_zoom(10)
    #fig.show()



if __name__ == "__main__":
    for i in range(900,1200):
        test(i)