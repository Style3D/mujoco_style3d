
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene_builder as s3d_scene_builder
import synreal_mujoco._mj_data_helper as _mj_data_helper
import synreal_mujoco.smj as smj

class s3d_scene_stepper:
    def __init__(self,mujoco_model,mujoco_data,scene):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.scene : s3d_scene_builder.s3d_scene = scene

    def set_mocap_pos(self, mocap_name, pos):
        smj.set_mocap_pos(self.mujoco_model, self.mujoco_data, mocap_name, pos)

    def update_force_2_mujoco(self):
        smj.update_rigidbody_cloth_collision_force(self.mujoco_model, self.mujoco_data, self.scene.mapper)
        smj.apply_collision_force_to_rigidbody(self.mujoco_model, self.mujoco_data, self.scene.mapper) ## cloth affacts rigid body

    def set_rigidbody_pos_mj_2_s3d(self):
        s3d_mj.set_rigid_body_pos_to_sim(self.mujoco_model, self.mujoco_data, self.scene.rigid_bodies)
        s3d_scene_stepper._update_connects(self.mujoco_model, self.mujoco_data, self.scene)

    def step_s3d(self):
        self.scene. world. step_sim()


    def set_cloth_pos_s3d_2_mj(self):
        self.scene.world. fetch_sim(0)

        for cloth, cloth_name in zip(self.scene.sim_cloth, self.scene.cloth_names):
            x = cloth.get_positions()
            _mj_data_helper.set_cloth_positions(self.mujoco_model, self.mujoco_data, cloth_name, x)


        for dfm, dfm_name, used_verts in zip(self.scene.deformable_bodies, self.scene.deformable_body_names, self.scene.used_vert_of_deformable_body_collision_faces):
            x = dfm.get_positions()[used_verts]
            _mj_data_helper.set_cloth_positions(self.mujoco_model, self.mujoco_data, dfm_name, x)


    @staticmethod
    def _update_connects(mujoco_model, mujoco_data, scene):
        for connect_info in scene.connect_infos:
            if connect_info.object_type0 == 'rigid_body' and connect_info.object_type1 == 'deformable_body':

                dfm_name = connect_info.object1
                dfm_body = scene.deformable_bodies[scene.deformable_body_names.index(dfm_name)]
                rb_id = connect_info.rb_id
                rb_pos = mujoco_data.geom_xpos[rb_id]
                rb_mat = mujoco_data.geom_xmat[rb_id].reshape(3, 3)
                pos = connect_info.data0 @ rb_mat.T + rb_pos
                indices = connect_info.data1
                dfm_body.set_positions(pos, indices)
