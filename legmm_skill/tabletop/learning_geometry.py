"""Cache-only adaptation for >32 live training scenes, without changing geometry.

The production planner keeps 32 compiled models. A larger visual cohort would
otherwise compile the same per-env model again on every teacher query. This
outer LRU reuses those exact (model,data) pairs; the qualified configure method
still performs every full pose/joint reset and geometric calculation.
"""
from collections import OrderedDict
from .geometry import GeometryPlanner


class LearningGeometryPlanner(GeometryPlanner):
    def __init__(self,robot_path,max_models):
        super().__init__(robot_path,cache_models=True)
        self.max_models=max(32,int(max_models));self.full_cache=OrderedDict()

    def configure(self,state,base_pose,object_pose=None,joint_state=None):
        key=(state.shape,tuple(state.size),tuple(state.table_size),state.mass)
        cached=self.full_cache.pop(key,None)
        if cached is not None:self.model_cache[key]=cached
        super().configure(state,base_pose,object_pose,joint_state)
        self.full_cache[key]=(self.m,self.d)
        while len(self.full_cache)>self.max_models:self.full_cache.popitem(last=False)
