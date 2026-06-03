# include parent folder
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)
# include parent folder end

import time
import mujoco.viewer
import numpy as np


import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene as s3d_scene
import synreal_mujoco.s3d_scene_stepper as s3d_scene_stepper
import synreal_mujoco.data_classes as dc

import json
from pathlib import Path

def rigid_body_property_fn(geo_name, rb_builder:dc.rigid_body_builder):
    if  geo_name == 'table_box' or geo_name == 'table_mesh':
        print(f' set {geo_name} rigid body property')
        rb_builder.attrib. dynamic_friction = 0.007
        rb_builder.attrib. static_friction = 0.007
        rb_builder.attrib. mass = 3e-2
    else:
        rb_builder.attrib. dynamic_friction = 0.03
        rb_builder.attrib. static_friction = 0.03
        rb_builder.attrib. mass = 3e-2


curr_folder = Path(__file__).parent
login_file = curr_folder.parent.parent / 'simulation_login.json'
s3d_mj.log_in_simulation(login_file = login_file) # this line is optional, but a login prompt will pop up latter


#mjcf_file = 'xml_projects/zjrx_lefthand/left_hand.xml'
s3d_scene_builder = s3d_scene.s3d_scene_builder()

s3d_scene_builder.add_mjcf_rigidbodies(curr_folder / 'xml_projects/wonik_allegro/left_hand.xml', rigidbody_builder_fn = rigid_body_property_fn )

######### cloth
cloth_builder = s3d_scene_builder.add_cloth_by_file( curr_folder /'xml_projects'/'wonik_allegro'/'assets'/ '50k_plane.obj')
cloth_builder.translate = np.array([-0.3, -1.5, -0.04])
cloth_builder.quat = np.array([1,0,0,0])
cloth_builder.rgba = np.array([1, 0.8, 0.0, 1])
#cloth_builder.attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)

m,d,s = s3d_scene_builder.build()

l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

with open(curr_folder /'xml_projects'/'wonik_allegro'/'trajectory_param.json', 'r') as fin:
    data = json.load(fin)

drop_rate = data['drop_rate']
hand_z_min = data['hand_z_min']

with mujoco.viewer.launch_passive(m, d) as viewer:

    fi = 0

    while viewer.is_running():

        begin0_t = time.time()

        x = 0.004 * fi

        if  x < 1.2:

            z = np. clip( 0.3 - drop_rate * float(fi), hand_z_min , 1 )

            l_s3d_scene_stepper.set_mocap_pos('palm', np.array([ x , 0.5 , z]))

        l_s3d_scene_stepper.update_force_2_mujoco()

        mujoco.mj_step(m, d)

        l_s3d_scene_stepper.set_rigidbody_pos_mj_2_s3d()
        l_s3d_scene_stepper.step_s3d()
        l_s3d_scene_stepper.set_cloth_pos_s3d_2_mj()

        viewer.sync()

        fi += 1

        end0_t = time.time()
        duration0 = end0_t - begin0_t
        print(f'fps: {1. / duration0:.2f} ')
