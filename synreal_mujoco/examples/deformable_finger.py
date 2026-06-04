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

from pathlib import Path
import json

curr_folder = Path(__file__).parent
login_file = curr_folder.parent.parent / 'simulation_login.json'
s3d_mj.log_in_simulation(login_file=login_file) # this line is optional, but a login prompt will pop up latter

s3d_scene_builder = s3d_scene_builder.s3d_scene_builder()
s3d_scene_builder.add_mjcf_rigidbodies(curr_folder/'xml_projects/TactiSim/DexHand.xml')

########## tets
#dfm_attrib = s3d_scene_builder.add_deformable_body_by_file(curr_folder/'xml_projects/deformable_finger/assets/tets1.vtk')
#dfm_attrib.attrib.youngsModulus = 1e5
##dfm_attrib.get_rest_pos = lambda  x: x # alter rest pos
#dfm_attrib.get_pos = lambda  x: x + np.array([0,0,0.3]) # alter current pos

m,d,s = s3d_scene_builder.build()

l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

# 目标位置关节角度
target_qpos = np.array([0.514,0.686,0.539,  0.664,0.948,0.795,  0.748,0.997,0.847,  0.687,0.916,0.762,  0.416,0.924,0.77])
current_qpos = np.zeros((15,))

# 参数配置
step_ratio = 0.01
max_iterations = 1000
tolerance = 0.001          # 允许的位置误差

ConcateFlag= False

def set_ctrl(name, value, Model,Data):
    id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    Data.ctrl[id] = value

def process(frame,Model,Data):
    global ConcateFlag

    if  frame == 10:
        set_ctrl("mot_joint1", 0.05, Model,Data)
        set_ctrl("mot_joint2", -1.5708-0.28, Model,Data)
        set_ctrl("mot_joint5", 0.28, Model,Data)
        set_ctrl("mot_joint6", 1.5708, Model,Data)
        set_ctrl("act_joint1_1", 1.5708, Model,Data)

    elif frame == 1000:
        set_ctrl("mot_joint1", 0.00, Model,Data)
        ConcateFlag = True

    if ConcateFlag:
        all_reached = True
        for i in range(len(current_qpos)):
            target = target_qpos[i]

            # 计算剩余距离
            remaining = target - current_qpos[i]
            if abs(remaining) <= tolerance:
                current_qpos[i] = target
                continue
            else:
                all_reached = False

            # 动态步长
            step = remaining * step_ratio
            current_qpos[i] += step

        Data.ctrl[6:21] = current_qpos
        if all_reached:
            print("所有关节已到达目标位置")
            ConcateFlag = False

with mujoco.viewer.launch_passive(m, d) as viewer:

    fi = 0

    while viewer.is_running():

        process(fi, m,d)

        mujoco.mj_step(m, d)

        l_s3d_scene_stepper.set_rigidbody_pos_mj_2_s3d()
        l_s3d_scene_stepper.step_s3d()
        l_s3d_scene_stepper.set_cloth_pos_s3d_2_mj()

        viewer.sync()

        fi += 1

