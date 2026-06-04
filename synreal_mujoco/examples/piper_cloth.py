# include parent folder
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)
# include parent folder end

import mujoco.viewer
import synreal_sim as sim
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene_builder as s3d_scene_builder
import synreal_mujoco.s3d_scene_stepper as s3d_scene_stepper
import synreal_mujoco.data_classes as dc
from pathlib import Path
import numpy as np

curr_folder = Path(__file__).parent
login_file = curr_folder.parent.parent / 'simulation_login.json'
s3d_mj.log_in_simulation(login_file = login_file) # this line is optional, but a login prompt will pop up latter


asset_dir = Path(__file__).parent.resolve() / 'xml_projects'
s3d_scene_builder = s3d_scene_builder.s3d_scene_builder()

def rb_builder(name,rb_builder):
    if name == 'link8': #TODO: link8/0 should be ok
        rb_builder.attrib.mass = 3e-2


s3d_scene_builder.add_mjcf_rigidbodies( asset_dir/ 'piper_secription'/'piper_description.xml', rigidbody_builder_fn = rb_builder)

######### cloth
cloth_builder = s3d_scene_builder.add_cloth_by_file( asset_dir / 'clothes'/ '50k_plane.obj')
cloth_builder.translate = np.array([-0.8, -2.0, 0.15])
cloth_builder.quat = np.array([1,0,0,0])
cloth_builder.attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)

m,d,s = s3d_scene_builder.build()

l_s3d_scene_stepper = s3d_scene_stepper.s3d_scene_stepper(m,d,s)

with mujoco.viewer.launch_passive(m, d) as viewer:

    while viewer.is_running():

        mujoco. mj_step(m, d)

        l_s3d_scene_stepper.set_rigidbody_pos_mj_2_s3d()
        l_s3d_scene_stepper.step_s3d()
        l_s3d_scene_stepper.set_cloth_pos_s3d_2_mj()

        viewer.sync()

