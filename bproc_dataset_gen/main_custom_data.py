import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy


#import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2000, help="How many scenes with 25 images each to generate")
parser.add_argument('--dataset_name', type=str, default="tless_mod", help="Name of dataset to generate the training data for")
args = parser.parse_args()

bproc.init()

# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.dataset_name), 
                                             model_type = 'cad', 
                                             mm2m = True)
print("----------> Target bop objects", [vars(a) for a in target_bop_objs])


# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.dataset_name))

# set shading and hide objects
for obj in (target_bop_objs): # + itodd_dist_bop_objs):  #hb_dist_bop_objs):    #ycbv_dist_bop_objs  
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(100)

# load cc_textures
orig_cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
reqd_mat_names = ["concrete", "wood", "metal"]       #["metalplates", "metal", "paintedmetal"]   
reqd_materials = []
for mat in bpy.data.materials.keys():
    for req_mat in reqd_mat_names:
        if req_mat in mat.lower():
            #print("----------------------------------> ", mat)
            reqd_materials.append(bpy.data.materials[mat])
cc_textures = [a for a in orig_cc_textures if a.blender_obj in reqd_materials]
#print(cc_textures)
#print([a for a in cc_textures if a.blender_obj in [bpy.data.materials[""]]])


# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.2], [0.3, 0.3, 0.2])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def sample_pose_func_alt(obj: bproc.types.MeshObject):
    min = np.random.uniform([0, 0, 0], [0, 0, 0])
    max = np.random.uniform([0.05, 0.05, 0.05], [0.05, 0.05, 0.05])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

for i in range(args.num_scenes):

    # Sample bop objects for a scene
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=10, replace=False))
    
    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs):  
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless', 'distractor']:
            #grey_col = np.random.uniform(0.1, 0.9)   
            #mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
            mat.set_principled_shader_value("Base Color", [0.0, 0.0, 0.0, 1])

        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))
        if obj.get_cp("bop_dataset_name") == 'itodd':  
            mat.set_principled_shader_value("Metallic", np.random.uniform(0.5, 1.0))
        if obj.get_cp("bop_dataset_name") == 'tless':
            mat.set_principled_shader_value("Specular", np.random.uniform(0.3, 1.0))
            #mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.5))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.1))
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 0.7, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs)

    cam_poses = 0
    while cam_poses < 25:
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, #+ sampled_distractor_bop_objs, 
                                                        size=10, replace=False))
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.35,
                                radius_max = 0.6,
                                #azimuth_min=-180,
                                #azimuth_max=180,
                                elevation_min = 5,
                                elevation_max = 85)
        # Set camera resolution Identical to ZED 2i CAMERA
        bproc.camera.set_resolution(image_width=640, image_height=360)
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,   
                                                                 inplane_rot=np.random.uniform(-3.14159, 3.14159)
                                                                )
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.2}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = args.dataset_name,
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in (sampled_target_bop_objs): #+ sampled_distractor_bop_objs):      
        obj.disable_rigidbody()
        obj.hide(True)

