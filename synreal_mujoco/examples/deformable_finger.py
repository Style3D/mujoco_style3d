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

######### rigidbodies
s3d_scene_builder = s3d_scene_builder.s3d_scene_builder()
s3d_scene_builder.add_mjcf_rigidbodies(curr_folder/'xml_projects/TactiSim/DexHand_dfm_finger.xml')

######### tets
dfm_attrib = s3d_scene_builder.add_deformable_body_by_file(curr_folder/'xml_projects/TactiSim/meshes/dfm_fingertip.vtk')
dfm_attrib.attrib.youngsModulus = 1e6
#dfm_attrib.get_rest_pos = lambda  x: x # alter rest pos
dfm_attrib.get_pos = lambda  x: x # alter current pos

######### connects
s3d_scene_builder.add_connect(curr_folder/'xml_projects/TactiSim/meshes/connect_finger_tip.json')

m, d, s = s3d_scene_builder.build()

##reset to key frame
#initial_key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "initial")
#mujoco.mj_resetDataKeyframe(m, d, initial_key_id)
#mujoco.mj_forward(m, d)

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


l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

# 目标位置关节角度
target_qpos = np.array([0.514,0.686,0.539,  0.664,0.948,0.795,  0.748,0.997,0.847,  0.687,0.916,0.762,  0.416,0.924,0.77])
current_qpos = np.zeros((15,))

# 参数配置
step_ratio = 0.01
max_iterations = 1000
tolerance = 0.001          # 允许的位置误差

ConcateFlag = False

def set_ctrl(name, value, Model,Data):
    id = mujoco.mj_name2id(Model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    Data.ctrl[id] = value

def process(frame,Model,Data):
    global ConcateFlag

    if  frame == 0:
        for name, value in (
            ("mot_joint1", 0.05),
            ("mot_joint2", -1.5708 - 0.28),
            ("mot_joint5", 0.28),
            ("mot_joint6", 1.5708),
            ("act_joint1_1", 1.5708),
        ):
            set_ctrl(name, value, Model,Data)
        pass

    elif frame == 100:
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

        process(fi, m, d)

        mujoco.mj_step(m, d)

        l_s3d_scene_stepper.set_rigidbody_pos_mj_2_s3d()
        l_s3d_scene_stepper.step_s3d()
        l_s3d_scene_stepper.set_cloth_pos_s3d_2_mj()

#TODO: output the whole scene to .obj when I press '1' 

        viewer.sync()

        fi += 1

