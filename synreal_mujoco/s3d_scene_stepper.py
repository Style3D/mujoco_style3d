
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.s3d_scene_builder as s3d_scene_builder
import synreal_mujoco._mj_data_helper as _mj_data_helper
import synreal_mujoco.smj as smj
import synreal_sim as sim
import numpy as np

class s3d_scene_stepper:
    def __init__(self,mujoco_model,mujoco_data,scene):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.scene : s3d_scene_builder.s3d_scene = scene
        self._stress_mesh_collections = {}


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


    def set_cloth_pos_s3d_2_mj(self ):
        self.scene.world.fetch_sim(0, [
                sim.OutputVars.Positions,
                sim.OutputVars.Transforms,
                sim.OutputVars.CollisionForceRigidBoydCloth,
                sim.OutputVars.AvatarStressMapObstacle,
                sim.OutputVars.AvatarStressMapCloth
            ])

        for cloth, cloth_name in zip(self.scene.sim_cloth, self.scene.cloth_names):
            x = cloth.get_positions()
            _mj_data_helper.set_flex_positions(self.mujoco_model, self.mujoco_data, cloth_name, x)

        for dfm, dfm_name, used_verts in zip(self.scene.deformable_bodies, self.scene.deformable_body_names, self.scene.used_vert_of_deformable_body_collision_faces):
            x = dfm.get_positions()
            x = x[used_verts]
            name = s3d_scene_builder._name_2_xml_name(s3d_scene_builder.xml_prefix_deformable_body, dfm_name)
            _mj_data_helper.set_flex_positions(self.mujoco_model, self.mujoco_data, name, x)

    def get_deformable_body_positions_in_rigidbody_frame(self, dfm_name, rigidbody_name):

        dfm = self.scene.deformable_bodies[self.scene.deformable_body_names.index(dfm_name)]

        x = dfm.get_positions()

        geom_id = self.scene.mj_geom_index[self.scene.rigid_body_names.index(rigidbody_name)]
        if geom_id is not None:
            x = _mj_data_helper.get_local_coodinate(x, self.mujoco_data.geom_xmat[geom_id].reshape(3, 3), self.mujoco_data.geom_xpos[geom_id])
        return x 

    def get_deformable_body_stress(self, dfm_name):
        dfm = self.scene.deformable_bodies[self.scene.deformable_body_names.index(dfm_name)]
        return dfm.get_stress_map()

    def reset_deformable_body_to_connected_pos(self, rigid_body_name, dfm_name ):
        for connect_info in self.scene.connect_infos:
            if connect_info.object0==rigid_body_name and connect_info.object1 == dfm_name:
                dfm_body = self.scene.deformable_bodies[self.scene.deformable_body_names.index(dfm_name)]

                geom_id = self.scene.mj_geom_index[self.scene.rigid_body_names.index(rigid_body_name)]
                geom_pos = self.mujoco_data.geom_xpos[geom_id]
                geom_mat = self.mujoco_data.geom_xmat[geom_id].reshape(3, 3)
                pos = _mj_data_helper.get_world_coodinate(connect_info.data0, geom_mat, geom_pos)
                dfm_body.set_positions(pos, np.arange(len(pos)))

    @staticmethod
    def _update_connects(mujoco_model, mujoco_data, scene):
        for connect_info in scene.connect_infos:
            if connect_info.is_deformable_body_attatch_to_rigid_body():

                dfm_name = connect_info.object1
                dfm_body = scene.deformable_bodies[scene.deformable_body_names.index(dfm_name)]

                geom_name = connect_info.object0
                local_coords = connect_info.data0

                geom_id = scene.mj_geom_index[scene.rigid_body_names.index(geom_name)]
                geom_pos = mujoco_data.geom_xpos[geom_id]
                geom_mat = mujoco_data.geom_xmat[geom_id].reshape(3, 3)

                pos = _mj_data_helper.get_world_coodinate(local_coords, geom_mat, geom_pos)
                indices = connect_info.data1
                dfm_body.set_positions(pos[indices], indices)
