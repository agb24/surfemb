from pytransform3d.urdf import UrdfTransformManager as py3d_urdf
from pytransform3d.editor import TransformEditor
import matplotlib.pyplot as plt

urdf_file = "/home/ise.ros/akshay_work/NN_Implementations/surfemb/task_planning/permutandis/permutandis/custom_problem/ur5.urdf"
with open(urdf_file, "r") as file:
    urdf_data = file.read()

transform_mgr = py3d_urdf()
urdf = transform_mgr.load_urdf(urdf_data)
print(transform_mgr.transforms)

ax = transform_mgr.plot_frames_in("world", s=0.2)
ax = transform_mgr.plot_connections_in("world", ax=ax)
ax.set_xlim((-0.2, 0.8))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((-0.2, 0.8))
plt.show()