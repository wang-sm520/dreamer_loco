import ast
from pathlib import Path


def test_learning_uses_same_physics_step_and_contact_verdict_as_production():
    root=Path(__file__).resolve().parents[1]
    production=ast.parse((root/'isaac_probe.py').read_text())
    learning=ast.parse((root/'learning_episode.py').read_text())
    def step(tree):
        return next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='step')
    # Critical sensor filtering, command delay, standing control and per-step
    # success accumulation must remain the audited production implementation.
    assert ast.dump(step(production),include_attributes=False)==ast.dump(step(learning),include_attributes=False)
    ppo=ast.parse((root/'ppo_episode.py').read_text())
    assert ast.dump(step(production),include_attributes=False)==ast.dump(step(ppo),include_attributes=False)
