from copy import deepcopy
from dataclasses import dataclass
from typing import List
from typing import Callable
from typing import Dict

import numpy as np
import synreal_sim as sim
from synreal_mujoco._deformable_data_helper import *
import synreal_mujoco.s3d_mj as s3d_mj
import synreal_mujoco.smj as smj
from synreal_mujoco import cloth_property
import synreal_mujoco.data_classes as dc

import xml.etree.ElementTree as ET
from pathlib import Path
import os

from synreal_mujoco.utility import kwargs_helper


class s3d_scene_builder:
    def __init__(self  ):

        # deformable body
        self.deformable_body_name_prefix = 'dfm'
        self.deformable_body_files : List[str] = []
        self.deformable_body_buidlers : List[dc.deformable_body_builder] = []
        self._temp_files: List[str] = []

        #rigid body
        self.mjcf_file =''
        self.flexed_mjcf_file = ''
        self.rigidbody_builder_fn : Callable[[str],dc.rigid_body_builder]

        #cloth
        self.cloth_name_prefix = 'cloth'
        self.cloth_files = []
        self.cloth_builder_map : Dict[str, dc.cloth_builder] = {}


    # mujoco mjcf

    # attrib_setter : lambda (rigidbody_name) -> rigidbody_attrib
    # note: set attrib with setter is for performance reason, so that the mjcf file can be loaded later instead of load here right away
    #def add_mjcf_rigidbodies( self, filename, rigidbody_builder : Callable[[str, dc.rigid_body_builder],None ]= None ):
    def add_mjcf_rigidbodies( self, filename, **kwargs ):
        self.mjcf_file = filename
        #TODO: fix this, runtime error
        with kwargs_helper(kwargs) as kw_helper:
            self.rigidbody_builder_fn = kw_helper.get('rigidbody_builder_fn', lambda name, rigidbody_builder : None)

    # clothes
    def add_cloth_by_file(self, filename ):
        self.cloth_files.append(filename)
        builder = dc.cloth_builder()
        name = s3d_scene_builder. _get_cloth_name_frome_file(self.cloth_name_prefix, filename)
        self.cloth_builder_map[name] = builder
        return builder

    # deformable body
    def add_deformable_body_by_file(self, filename ):
        self.deformable_body_files.append(filename)
        dfm_builder = dc.deformable_body_builder()
        dfm_builder.get_pos = lambda x : x
        dfm_builder.get_rest_pos = lambda x : x
        self.deformable_body_buidlers.append(dfm_builder)
        return dfm_builder


    @staticmethod
    def _add_rigid_body_to_scene(s : dc.s3d_scene, m, d, rigidbody_builder_fn : Callable[[str,dc.rigid_body_builder], None ]):
        s.rigid_bodies, s.mj_index = s3d_mj._add_rigid_body_to_sim(m, d, s.world, rigidbody_builder_fn)


    @staticmethod
    def _get_cloth_name_frome_file(prefix, obj_file):
        file_base_name = str(Path(obj_file).stem)
        return prefix +'_' + file_base_name

    @staticmethod
    def _add_cloth_to_scene(s : dc.s3d_scene, m, d , attrib_map, name_start_with_will_considered_cloth):

        def __attrib_getter (name ):
           return attrib_map[name].attrib

        s.sim_cloth, s.cloth_names = s3d_mj._add_cloth_to_sim_2( m, d, s.world,  __attrib_getter , name_start_with_will_considered_cloth )


    def _add_deformable_body_to_scene(self, scene : dc.s3d_scene, dfm_body_params):
        scene.deformable_bodies = []
        for dfm in dfm_body_params:
            obj = sim.DeformableBody(dfm.pos , dfm.collision_faces, dfm.tets, dfm.rest_pos)
            obj.set_attrib(dfm.attrib)
            scene.deformable_bodies.append(obj)
            obj.attach(scene.world)


    @staticmethod
    def _export_surface_to_obj(pos, faces, obj_path: str) -> None:
        with open(obj_path, 'w') as f:
            for v in pos:
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for face in faces:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

    @staticmethod
    def _add_flexcomp_to_worldbody(tree: ET.ElementTree, name:str, file: str, pos,quat,rgba, **attribs) -> None:
        """Inserts a <flexcomp> with the given file into <worldbody>. Extra keyword
        arguments are added as XML attributes (e.g. name, type, pos, radius, dim)."""
        worldbody = tree.getroot().find('worldbody')
        if worldbody is None:
            raise ValueError("No <worldbody> element found in the XML tree")

        attrs = {
            'name': name,
            'type': 'mesh',
            'pos': f'{pos[0]} {pos[1]} {pos[2]}',
            'quat': f'{quat[0]} {quat[1]} {quat[2]} {quat[3]}',
            'radius': '0.0005',
            'dim': '2',
            'rgba': f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}',
            'custom': 'true',
            'file': str(file)
        }
        attrs.update(attribs)
        elem = ET.SubElement(worldbody, 'flexcomp', attrs)
        elem.text = '\n        '  # forces explicit </flexcomp> closing tag instead of />
        elem.tail = '\n\n    '    # newline between </flexcomp> and </worldbody>

    def _add_flex_cloth(self,tree):
        for i,cloth_file in enumerate(self.cloth_files):
            name = s3d_scene_builder._get_cloth_name_frome_file(self.cloth_name_prefix, cloth_file)
            cloth_builder = self.cloth_builder_map[name]
            s3d_scene_builder._add_flexcomp_to_worldbody(tree, name, cloth_file,cloth_builder.translate,cloth_builder.quat,cloth_builder.rgba)

    def _add_flex_deformable_body(self,tree, mjcf_name, s : dc.s3d_scene):
        deformable_bodies_param=[]
        s.deformable_body_names =[]
        for i, dfm_file in enumerate(self.deformable_body_files):
            dfm_builder = self.deformable_body_buidlers[i]
            pos, tets = load_tetrahedrons(dfm_file)
            if dfm_builder.collision_faces is None:
                faces = compute_boundary_faces(tets)

            rest_pos = deepcopy(dfm_builder.get_pos(pos))
            curr_pos = deepcopy(dfm_builder.get_pos(pos))

            temp_obj_path = mjcf_name + f'_{self.deformable_body_name_prefix}_{i}.obj'
            temp_obj_path = Path(temp_obj_path).as_posix()  
            s3d_scene_builder._export_surface_to_obj(curr_pos, faces, temp_obj_path)  # export before offset mutates pos
            self._temp_files.append(temp_obj_path)

            name = self.deformable_body_name_prefix +'_' + str(i)
            #s3d_scene_builder._add_flexcomp_to_worldbody(
            #    tree, name, os.path.basename(temp_obj_path), np.array([0,0,0]), np.array([1,0,0,0]),np.array([1,0.6,0.8,1]))

            s3d_scene_builder._add_flexcomp_to_worldbody(
                tree, name, temp_obj_path, np.array([0,0,0]), np.array([1,0,0,0]),np.array([1,0.6,0.8,1]))

            s.deformable_body_names.append(name)
            deformable_bodies_param.append(dc.deformable_body_constructor_param(curr_pos, rest_pos, tets, faces,dfm_builder.attrib))
        return deformable_bodies_param


    def _add_flex_to_mjcf(self, s: dc.s3d_scene):
        tree = ET.parse(self.mjcf_file)
        base, ext = os.path.splitext(self.mjcf_file)

        # cloth
        self._add_flex_cloth(tree)

        # deformable body
        deformable_bodies_param = self._add_flex_deformable_body( tree, base,s)

        # write .xml
        out_path = base + '_flex' + ext
        tree.write(out_path)
        self._temp_files.append(out_path)
        self.flexed_mjcf_file = out_path

        return deformable_bodies_param


    # build
    def build(self ):

        scene = dc.s3d_scene()

        dfm_bodies_param = self._add_flex_to_mjcf(scene)

        m, d = s3d_mj.load_data(self.flexed_mjcf_file)

        for path in self._temp_files:
            os.remove(path)
        self._temp_files.clear()

        scene.world = s3d_mj.get_a_sim_world(m)

        s3d_scene_builder._add_rigid_body_to_scene(scene, m, d, self.rigidbody_builder_fn )

        s3d_scene_builder._add_cloth_to_scene(scene, m, d, self.cloth_builder_map, self.cloth_name_prefix)

        self._add_deformable_body_to_scene(scene, dfm_bodies_param)

        collision_force = []
        scene.mapper = smj.s3d_mj_mapper (
            scene.world,
            scene.sim_cloth,
            scene.cloth_names,
            scene.rigid_bodies,
            scene.mj_index,
            collision_force
        )

        return m, d, scene