import mujoco.viewer

import numpy as np

# include parent folder
import os
import sys

# include parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)
# include parent folder end

import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene_builder as s3d_scene_builder
import synreal_mujoco.s3d_scene_stepper as s3d_scene_stepper
import synreal_mujoco._mj_data_helper as _mj_data_helper
from synreal_mujoco._deformable_data_helper import load_tetrahedrons, compute_boundary_faces

from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

curr_folder = Path(__file__).parent
login_file = curr_folder.parent.parent / 'simulation_login.json'
s3d_mj.log_in_simulation(login_file=login_file) # this line is optional, but a login prompt will pop up latter

######### rigidbodies
s3d_scene_builder = s3d_scene_builder.s3d_scene_builder()
s3d_scene_builder.add_mjcf_rigidbodies(curr_folder/'xml_projects/TactiSim/DexHand_dfm_finger.xml')

######### finger tip
dfm_file = curr_folder/'xml_projects/TactiSim/meshes/dfm_fingertip.vtk'
dfm_attrib = s3d_scene_builder.add_deformable_body_by_file(dfm_file)
dfm_attrib.attrib.youngsModulus = 1e7
#dfm_attrib.get_rest_pos = lambda  x: x # alter rest pos
dfm_attrib.get_pos = lambda  x: x # alter current pos
_, dfm_tets = load_tetrahedrons(dfm_file)
dfm_faces, _ = compute_boundary_faces(dfm_tets)

########## tets
#dfm_attrib = s3d_scene_builder.add_deformable_body_by_file(curr_folder/'xml_projects/piper_secription/tets1.vtk')
#dfm_attrib.attrib.youngsModulus = 1e6
##dfm_attrib.get_rest_pos = lambda  x: x # alter rest pos
#dfm_attrib.get_pos = lambda  x: x + np.array([0,0.2,0.6]) # alter current pos

########### cloth
##cloth_builder = s3d_scene_builder.add_cloth_by_file( curr_folder / 'xml_projects' / 'clothes'/ '50k_plane.obj')
##cloth_builder = s3d_scene_builder.add_cloth_by_file( curr_folder / 'xml_projects' / 'TactiSim'/ 'meshes' / 'my_cylinder.obj')
#cloth_builder.translate = np.array([-0.8, -2.0, 0.25])
#cloth_builder.quat = np.array([1,0,0,0])

######### connects
s3d_scene_builder.add_connect(curr_folder/'xml_projects/TactiSim/meshes/connect_finger_tip.json')
#s3d_scene_builder.add_connect(curr_folder/'xml_projects/TactiSim/meshes/connect_tet.json')

m, d, s = s3d_scene_builder.build()
l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

################################################
class controller:

    def __init__(self):
        # 目标位置关节角度
        self.target_qpos = np.array([0.514,0.686,0.539,  0.664,0.948,0.795,  0.748,0.997,0.847,  0.687,0.916,0.762,  0.416,0.924,0.77])
        self.current_qpos = np.zeros((15,))

        # 参数配置
        self.step_ratio = 0.01
        self.max_iterations = 1000
        self.tolerance = 0.001          # 允许的位置误差

        self.ConcateFlag = False


    def set_to_final(self, m, d):
        for name, value in (
            ("joint1", 0.05),
            ("joint2", -1.5708 - 0.28),
            ("joint5", 0.28),
            ("joint6", 1.5708),
            ("l_f_joint1_1", 1.5708),
        ):
            joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(m, d)


    def set_ctrl(self,name, value, Model, Data):
        id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        Data.ctrl[id] = value

    def process(self, frame, Model, Data):
        if  frame == 0:
            for name, value in (
                ("mot_joint1", 0.05),
                ("mot_joint2", -1.5708 - 0.28),
                ("mot_joint5", 0.28),
                ("mot_joint6", 1.5708),
                ("act_joint1_1", 1.5708),
            ):
                self.set_ctrl(name, value, Model,Data)

        elif frame == 100:
            self.set_ctrl("mot_joint1", 0.00, Model, Data)
            self.ConcateFlag = True

        if self.ConcateFlag:
            all_reached = True
            for i in range(len(self.current_qpos)):
                target = self.target_qpos[i]

                # 计算剩余距离
                remaining = target - self.current_qpos[i]
                if abs(remaining) <= self.tolerance:
                    self.current_qpos[i] = target
                    continue
                else:
                    all_reached = False

                # 动态步长
                step = remaining * self.step_ratio
                self.current_qpos[i] += step

            Data.ctrl[6:21] = self.current_qpos
            if all_reached:
                print("所有关节已到达目标位置")
                self.ConcateFlag = False



class stress_viewer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(1,1, subplot_kw={'projection': '3d'})
        plt.show()

        self.dfm_scatter = None

    def update(self,stepper):
        dfm_x = stepper.get_deformable_body_positions_in_rigidbody_frame('dfm_fingertip','l_f_link5_4/l_f_link5_4')
        dfm_stress = stepper.get_deformable_body_stress('dfm_fingertip')

        dfm_x = np.asarray(dfm_x)
        dfm_color = np.asarray(dfm_stress)
        if dfm_color.ndim > 1:
            dfm_color = np.linalg.norm(dfm_color.reshape(dfm_color.shape[0], -1), axis=1)
        else:
            dfm_color = dfm_color.ravel()

        #dfm_cmin = float(dfm_color.min())
        #dfm_cmax = float(dfm_color.max())
        if self.dfm_scatter is None:
            dfm_min = dfm_x.min(axis=0)
            dfm_max = dfm_x.max(axis=0)
            dfm_center = (dfm_min + dfm_max) * 0.5
            dfm_radius = max(float((dfm_max - dfm_min).max()) * 0.6, 1e-6)
            self.ax.set_xlim(dfm_center[0] - dfm_radius, dfm_center[0] + dfm_radius)
            self.ax.set_ylim(dfm_center[1] - dfm_radius, dfm_center[1] + dfm_radius)
            self.ax.set_zlim(dfm_center[2] - dfm_radius, dfm_center[2] + dfm_radius)
            self.ax.set_box_aspect((1, 1, 1))
            self.dfm_scatter = self.ax.scatter(dfm_x[:, 0], dfm_x[:, 1], dfm_x[:, 2],
                                     c=dfm_color, cmap='viridis', s=8,
                                     linewidths=0, depthshade=False)
        else:
            self.dfm_scatter._offsets3d = (dfm_x[:, 0], dfm_x[:, 1], dfm_x[:, 2])
            self.dfm_scatter.set_array(dfm_color)
        #self.dfm_scatter.set_clim(dfm_cmin, dfm_cmax)
        self.dfm_scatter.set_clim(0, 1e4)
        
        self.fig.canvas.draw()  # Redraw the figure
        self.fig.canvas.flush_events()


def export_scene_to_obj():
    obj_path = curr_folder / f'deformable_finger_scene_{int(time.time())}.obj'
    vertex_offset = 0
    with open(obj_path, 'w') as f:
        def write_mesh(name, verts, faces):
            nonlocal vertex_offset
            if len(verts) == 0 or len(faces) == 0:
                return
            safe_name = str(name).replace(' ', '_').replace('/', '_')
            f.write(f'o {safe_name}\n')
            for v in verts:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face in faces:
                a, b, c = face + vertex_offset + 1
                f.write(f'f {a} {b} {c}\n')
            vertex_offset += len(verts)

        def write_rigid_mesh(rigid_i, x, t, geo_mat, geo_pos, collision_mask, collision_group):
            verts = x @ geo_mat.reshape(3, 3).T + geo_pos
            write_mesh(f'rigid_{rigid_i}', verts, t)

        _mj_data_helper.for_each_rigid_meshes(m, d, write_rigid_mesh)

        for name, dfm_body, used_verts in zip(s.deformable_body_names, s.deformable_bodies, s.used_vert_of_deformable_body_collision_faces):
            write_mesh(name, dfm_body.get_positions()[used_verts], dfm_faces)

    print(f'exported scene to {obj_path}')

def key_callback(keycode):
    if keycode == ord('1'):
        export_scene_to_obj()


################################################

ctrller = controller()

ctrller.set_to_final(m,d)

l_s3d_scene_stepper.reset_deformable_body_to_connected_pos('l_f_link5_4/l_f_link5_4','dfm_fingertip')


s_viewer = stress_viewer()

with mujoco.viewer.launch_passive(m, d, key_callback = key_callback) as viewer:

    fi = 0

    while viewer.is_running():

        ctrller.process(fi, m, d)

        mujoco.mj_step(m, d)

        l_s3d_scene_stepper.set_rigidbody_pos_mj_2_s3d()
        l_s3d_scene_stepper.step_s3d()
        l_s3d_scene_stepper.set_cloth_pos_s3d_2_mj()
        
        s_viewer.update(l_s3d_scene_stepper)

        viewer.sync()

        fi += 1

