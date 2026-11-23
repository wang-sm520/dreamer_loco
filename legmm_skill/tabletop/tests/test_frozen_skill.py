import hashlib,json
import pytest
pytest.importorskip('rsl_rl')
import torch
from tabletop.student import Student
from tabletop.spec import STOW
from tabletop.frozen_skill import FrozenW8Skill


def test_frozen_student_exact_parity_checksum_and_no_truth_argument(tmp_path):
    torch.set_num_threads(2);student=Student('cpu',11)
    path=tmp_path/'student.pt'
    torch.save(dict(model=student.model.state_dict(),model_config=student.model.config,
        normalization=dict(mean=student.mean,scale=student.scale)),path)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(format='w8_frozen_skill_v1',source_sha256={},
        weights=dict(student=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))))
    frozen=FrozenW8Skill(manifest)
    images=torch.zeros(1,2,2,128,128);proprio=torch.tensor([[*STOW,.072]])
    with pytest.raises(RuntimeError):frozen.act_depth(images,proprio)
    frozen.reset(1,proprio)
    expected=student.act(images,proprio,proprio)[:,:4]
    assert torch.equal(frozen.act_depth(images,proprio),expected)
    assert all(not parameter.requires_grad for parameter in frozen.student.model.parameters())
    with pytest.raises(TypeError):frozen.act_depth(images,proprio,torch.zeros(1,3))
    from tabletop.residual_ppo import build_residual_ppo
    algorithm=build_residual_ppo(dict(policy=torch.zeros(1,547),critic=torch.zeros(1,5)))
    residual=tmp_path/'residual.pt';torch.save(dict(actor_state_dict=algorithm.actor.state_dict()),residual)
    data=json.loads(manifest.read_text())
    data['weights']['residual']=dict(file=residual.name,sha256=hashlib.sha256(residual.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(data));combined=FrozenW8Skill(manifest);combined.reset(1,proprio)
    assert torch.equal(combined.act_depth(images,proprio),expected)
    # Concurrent training/evaluation feature wrappers must not retain each
    # other's forwards (a full validation otherwise keeps every GPU feature).
    from tabletop.residual_ppo import FrozenStudentFeatures
    wrapper=FrozenStudentFeatures(combined.student)
    saved=len(combined.features.parts)
    for _ in range(3):wrapper.observe(images,proprio,proprio)
    assert len(combined.features.parts)==saved and len(wrapper.parts)==1
    wrapper.close()
    with path.open('ab') as file:file.write(b'changed')
    with pytest.raises(ValueError,match='Frozen weight changed'):FrozenW8Skill(manifest)
