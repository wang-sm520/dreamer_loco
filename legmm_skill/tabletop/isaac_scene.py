"""A2 scene with actual per-episode primitive geometry. Import after AppLauncher.

One cohort is one scene lifetime: each new cohort authors fresh dimensions before
PhysX parsing. No runtime visual-only scaling and no finite shape catalogue.
"""
from pathlib import Path
from pxr import UsdPhysics,PhysxSchema
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.asset.importer.mjcf')
import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg,AssetBaseCfg,RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from .spec import OUTPUT,STOW
from .collision import excluded_pair

LEG_ORDER=[f'{leg}_{j}_joint' for leg in ('FL','FR','RL','RR') for j in ('hip','thigh','calf')]
HOME_Q=[-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8]
ARM_BODIES=[f'link{i}' for i in range(1,7)]+['left','right']
DOG_BODIES=['base_link']+[f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf')]


@sim.clone
def spawn_robot(prim_path,cfg,translation=None,orientation=None):
    prim=sim.spawn_from_mjcf(prim_path,cfg.replace(articulation_props=None),translation,orientation)
    stage=prim.GetStage()
    world=stage.GetPrimAtPath(prim_path+'/worldBody')
    if world.IsValid():
        world.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        world.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    sim.modify_articulation_root_properties(prim_path+'/base_link/base_link',
        sim.ArticulationRootPropertiesCfg(enabled_self_collisions=bool(cfg.self_collision),solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,sleep_threshold=0.,stabilization_threshold=0.))
    # Preserve W2's no internal leg contact interaction, but enable arm-vs-dog.
    # Arm adjacent overlap belongs to the rigid joint design, not grasp failure.
    names=DOG_BODIES+['arm_base']+ARM_BODIES+['payload']
    for i,a in enumerate(names):
        pa=stage.GetPrimAtPath(f'{prim_path}/base_link/{a}')
        if not pa.IsValid(): continue
        relation=UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel()
        for b in names[i+1:]:
            if excluded_pair(a,b,DOG_BODIES):
                relation.AddTarget(f'{prim_path}/base_link/{b}')
    return prim


def robot_config(model):
    init={n:v for n,v in zip(LEG_ORDER,HOME_Q)}
    init.update({f'joint{i+1}':v for i,v in enumerate(STOW)})
    init.update(endleft=.036,endright=-.036)
    return ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',articulation_root_prim_path='/base_link/base_link',
        spawn=sim.MjcfFileCfg(func=spawn_robot,asset_path=str(model),usd_dir=str(OUTPUT/'generated_usd_mesh_shoulder_v2'),
            usd_file_name='a2_tabletop.usd',fix_base=False,import_sites=False,make_instanceable=False,
            self_collision=True,
            activate_contact_sensors=True,rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,
                retain_accelerations=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.4),joint_pos=init),actuators={
            'hip_thigh':ImplicitActuatorCfg(joint_names_expr=['.*_hip_joint','.*_thigh_joint'],stiffness=100.,damping=4.2,effort_limit_sim=120.,armature=.03,friction=0.),
            'calf':ImplicitActuatorCfg(joint_names_expr=['.*_calf_joint'],stiffness=150.,damping=7.,effort_limit_sim=180.,armature=.03,friction=0.),
            'shoulder':ImplicitActuatorCfg(joint_names_expr=['joint[1-3]'],stiffness=1000.,damping=40.2,effort_limit_sim=24.,friction=0.),
            'wrist':ImplicitActuatorCfg(joint_names_expr=['joint[4-6]'],stiffness=80.,damping=5.1,effort_limit_sim=8.,friction=0.),
            'fingers':ImplicitActuatorCfg(joint_names_expr=['end.*'],stiffness=1000.,damping=21.,effort_limit_sim=15.,friction=0.)})


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    robot:ArticulationCfg=None
    table:RigidObjectCfg=None
    target:RigidObjectCfg=None
    floor=AssetBaseCfg(prim_path='/World/floor',spawn=sim.CuboidCfg(size=(300.,300.,.2),
        rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),collision_props=sim.CollisionPropertiesCfg(),
        physics_material=sim.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.,friction_combine_mode='max')),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.1)))
    light=AssetBaseCfg(prim_path='/World/light',spawn=sim.DomeLightCfg(intensity=1500.))


def make_scene(states,model,legacy_rigid_solver=False):
    cfg=TabletopSceneCfg(num_envs=len(states),env_spacing=5.,replicate_physics=False)
    cfg.robot=robot_config(model)
    solver={} if legacy_rigid_solver else dict(solver_position_iteration_count=4,solver_velocity_iteration_count=0)
    if not legacy_rigid_solver:
        for rigid in (cfg.robot.spawn.rigid_props,cfg.floor.spawn.rigid_props):
            rigid.solver_position_iteration_count=4;rigid.solver_velocity_iteration_count=0
    tables=[];objects=[]
    for s in states:
        material=sim.RigidBodyMaterialCfg(static_friction=s.friction,dynamic_friction=s.friction,restitution=0.)
        tables.append(sim.CuboidCfg(size=s.table_size,rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,**solver),
            collision_props=sim.CollisionPropertiesCfg(),physics_material=material,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.45,.3,.2))))
        common=dict(rigid_props=sim.RigidBodyPropertiesCfg(**solver),collision_props=sim.CollisionPropertiesCfg(),
            mass_props=sim.MassPropertiesCfg(mass=s.mass),physics_material=material,activate_contact_sensors=True,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.85,.55,.08)))
        objects.append(sim.CuboidCfg(size=s.size,**common) if s.shape=='box' else
                       sim.CylinderCfg(radius=s.size[0]/2,height=s.size[2],**common))
    cfg.table=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Table',spawn=sim.MultiAssetSpawnerCfg(assets_cfg=tables,random_choice=False))
    cfg.target=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Object',spawn=sim.MultiAssetSpawnerCfg(assets_cfg=objects,random_choice=False))
    for body in DOG_BODIES+ARM_BODIES:
        filters=['{ENV_REGEX_NS}/Table','{ENV_REGEX_NS}/Object','/World/floor']
        # Explicit pair columns, so a legal finger/object contact cannot hide a
        # simultaneous forbidden contact behind a net-force cancellation.
        if body in ARM_BODIES:
            filters += [f'{{ENV_REGEX_NS}}/Robot/base_link/{b}' for b in DOG_BODIES]
            filters += [f'{{ENV_REGEX_NS}}/Robot/base_link/{b}' for b in ['arm_base']+ARM_BODIES
                        if b != body and not excluded_pair(body,b,DOG_BODIES)]
        setattr(cfg,'contact_'+body,ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/base_link/{body}',
            filter_prim_paths_expr=filters,history_length=1,track_contact_points=True,max_contact_data_count_per_prim=64))
    return cfg
