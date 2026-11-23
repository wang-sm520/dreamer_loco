"""Prepare only the explicit new loader branch; no active/formal adoption."""
import ast,difflib

def recovery_source(source):
    source=source.replace('allow_closed_collection=False,expected_num_envs=64):','allow_closed_collection=False,expected_num_envs=64,allow_resource_abort=False):',1)
    source=source.replace('(allow_partial or allow_closed_collection)', '(allow_partial or allow_closed_collection or allow_resource_abort)',1)
    source=source.replace('(allow_partial or allow_closed_collection):', '(allow_partial or allow_closed_collection or allow_resource_abort):',1)
    needle='        elif allow_closed_collection:'
    assert needle in source;source=source.replace(needle,"        elif allow_resource_abort:\n            from .resource_abort_recovery import resource_abort_plan\n            payload['cursor']['resource_abort_plan']=resource_abort_plan(payload,updates_per_round,expected_num_envs)\n"+needle,1)
    ast.parse(source);return source

def patch(source):return ''.join(difflib.unified_diff(source.splitlines(True),recovery_source(source).splitlines(True),fromfile='a/legmm_skill/tabletop/dagger_recovery.py',tofile='b/legmm_skill/tabletop/dagger_recovery.py'))
