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

def get_viz_frames(frame, label, size=100, order=[0,1,2], extrinsic=True):
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


def get_frames_on_sphere(radius=1):
    thetas, phis = np.meshgrid(np.linspace(0, np.pi/2, 5),
                               np.linspace(-np.pi/2, np.pi/2, 5))
    rhos = np.ones_like(thetas) * radius
    spherical_grid = np.column_stack((rhos.reshape(-1), 
                                      thetas.reshape(-1), 
                                      phis.reshape(-1)))
    cartesian_grid = py3dcoord.cartesian_from_spherical(spherical_grid)

    frames_rotated = [] 
    for i,sph in enumerate(spherical_grid[:,:3]):
        fr = {"rpy": [sph[1],0,sph[2]]}
        fr_3x3 = py3drot.matrix_from_euler(fr["rpy"],
                                              i=0,j=1,k=2,
                                              extrinsic=True)
        fr_4x4 = np.zeros((4,4))
        fr_4x4[:3,:3] += fr_3x3
        fr_4x4[:3,3] += cartesian_grid[i,:3]
        fr_4x4[3,3] = 1
        fr_viz = py3dviz.Frame(fr_4x4, label=str(i), s=0.2)
        frames_rotated.append(fr_viz)
    return frames_rotated



def test():
    pi = np.pi
    # CREATE FIGURE
    fig = py3dviz.Figure("viz_win", width=1080, height=720)
    # TRANSFORM MANAGER
    transf_mgr = py3d_transf_mgr.TransformManager()
    

    # Create Base Frame + Visualizer
    base_frame = {"xyz":[0,0,0], "rpy":[0,0,0]}
    # Get 'Frame' for visualization
    base_frame_4x4, base_frame_quat, base_frame_viz = get_viz_frames(base_frame,
                                                                     "base_frame")
    fig.add_geometry(base_frame_viz.geometries[0])


    # Orient Z+ve Axis of the Gripper
    gripper_frame = {"xyz":[0,0,150], "rpy":[pi,0,0]}
    gripper_frame4x4, gripper_frame_quat, \
                            gripper_frame_viz = get_viz_frames(gripper_frame,
                                                               "gripper")
    fig.add_geometry(gripper_frame_viz.geometries[0])    

    '''# Sample Multiple Frames
    spherical_frames = get_frames_on_sphere()
    for sph_fr in spherical_frames:
        fig.add_geometry(sph_fr.geometries[0])'''
    

    # CREATE MESH AND REFERENCE AXIS
    # Read pose samples from PKL
    ind = 100
    root_path = "/home/ise.ros/akshay_work/datasets/tless_mod/tless_mod/models_cad"        
    pose_df = pd.read_pickle("/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_planning/permutandis/permutandis/custom_problem/6d_pose.pkl")
    # Read the mesh
    obj_id = pose_df["true_obj_id"].iloc[ind]
    mesh_path = os.path.join(root_path, f"obj_{obj_id:06d}.ply")
    # Get Transformation Matrix for this Object
    mesh_to_cam_pose_3x3 = np.asarray(pose_df["cam_R_m2c"].iloc[ind]).reshape((3,3))
    mesh_to_cam_pose_transf = np.asarray(pose_df["cam_t_m2c"].iloc[ind]).reshape((3,))
    mesh2cam_4x4 = np.eye(4)
    mesh2cam_4x4[:3,:3] = mesh_to_cam_pose_3x3
    mesh2cam_4x4[:3,3] = mesh_to_cam_pose_transf
    # Add the Mesh Geometry to the Figure
    mesh_geom = py3dviz.Mesh(mesh_path, s=[1,1,1], c=[0,0.5,0])
    fig.add_geometry(mesh_geom.geometries[0])


    # CREATE CAMERA
    # Set the POSE of the Mesh in world co-ordinates
    rot_mat = mesh_to_cam_pose_3x3.T
    cam2mesh_4x4 = np.eye(4)
    cam2mesh_4x4[:3,:3] = rot_mat 
    cam2mesh_4x4[:3,3] = mesh_to_cam_pose_transf
    intrinsic = np.asarray([264.546630859375, 0.0, 319.4079895019531,
                            0.0, 264.546630859375, 178.61087036132812,
                            0.0, 0.0, 1.0]).reshape((3,3))
    camera = py3dviz.Camera(M=intrinsic, cam2world=cam2mesh_4x4)
    fig.add_geometry(camera.geometries[0])

    # Add Transformation
    transf_mgr.add_transform("camera", "world", cam2mesh_4x4)


    def animate_callback(step, gripper_frame_viz, z_flattened, x_flattened,
                         mesh_geom):
        z_rot = z_flattened[step]
        x_rot = x_flattened[step]
        
        # Set the Frame orientation
        new_frame = {"xyz":[0,0,150], "rpy":[x_rot, 0, z_rot]}
        new_frame4x4, new_frame_quat, \
                    new_frame_viz = get_viz_frames(new_frame,
                                                   "new_frame", 
                                                   order=[0,1,2],
                                                   extrinsic=False)
        gripper_frame_viz.set_data(new_frame4x4)
        import time; time.sleep(0.05)
        return gripper_frame_viz, mesh_geom


    # Sample Rotations every 10 Degrees, for a range of 180 deg. 
    z_rotations = np.linspace(-np.pi/2, np.pi/2, 18)
    # Sample Rotations every 5 Degrees, for a range of 90 deg.
    x_rotations = np.linspace(np.pi/2, np.pi, 18)
    z_samples, x_samples = np.meshgrid(z_rotations, x_rotations)
    
    z_flattened = z_samples.flatten()
    x_flattened = x_samples.flatten()

    fig.view_init(azim=75, elev=50) 
    fig.set_zoom(1.2)
    fig.animate(animate_callback, 18*18,
                fargs=(gripper_frame_viz, z_flattened, x_flattened,
                       mesh_geom), 
                loop=True)
    fig.show()



if __name__ == "__main__":
    test()