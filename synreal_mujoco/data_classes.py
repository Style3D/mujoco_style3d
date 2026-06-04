
from dataclasses import dataclass, field
from typing import List
from typing import Callable

import numpy as np
import synreal_sim as sim
import synreal_mujoco.smj as smj
from synreal_mujoco import cloth_property

@dataclass
class s3d_scene:
    world: sim.World = None
    deformable_bodies: List[ sim.DeformableBody] = field(default_factory=list)
    deformable_body_names : List[str] = field(default_factory=list)

    rigid_bodies: List[ sim.RigidBody] = field(default_factory=list)
    mj_index: List[int] = field(default_factory=list) # index of rigid body in mujoco, the order is the same as rigid_bodies    
    mapper : smj.s3d_mj_mapper = None

    sim_cloth: List[sim.Cloth] = field(default_factory=list)
    cloth_names: List[str] = field(default_factory=list)


@dataclass
class rigid_body_builder:
    with_convex_hull : bool = False
    is_fixed: bool = True
    attrib: sim.RigidBodyAttrib = cloth_property.get_rigid_body_property_default()


@dataclass
class cloth_builder:
    translate = np.array([0,0,0])
    quat = np.array([1,0,0, 0])
    rgba =  np.array([1,0.6,0.8,1])
    attrib: sim.ClothAttrib = cloth_property.get_cloth_property_default()


@dataclass
class deformable_body_builder:
    attrib = sim.DeformableBodyAttrib()
    collision_faces = None
    get_pos = None  # lambda : positions -> f(positions)
    get_rest_pos = None # lambda : positions -> f(positions)


class deformable_body_constructor_param:
    def __init__(self, pos, rest_pos, tets, collision_faces,attrib):
        self.pos = pos
        self.rest_pos = rest_pos
        self.tets = tets
        self.collision_faces = collision_faces
        self.attrib = attrib
