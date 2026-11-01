"""The merged legmm config fragments load, and survive the size-preset regexes."""
import pytest

from drone_signature import load_config


@pytest.mark.parametrize('presets', [
    ['legmm'],
    ['legmm', 'size1m', 'legmm_debug'],
    ['legmm', 'legmm_wm'],
    ['legmm', 'legmm_bc'],
    ['legmm', 'legmm_rl'],
    ['legmm', 'legmm_online'],
    ['legmm', 'size100m', 'legmm_online'],
])
def test_legmm_presets(presets):
  c = load_config(presets)
  assert c.task == 'legmm_mm'
  assert c.env.legmm.obs_depth_dtype == 'uint8'
  assert c.env.legmm.domain_rand.cam_noise_a == (0.001, 0.004)
  assert c.env.legmm.camera_wrist.mount_xyz == (-0.08, 0.0, 0.06)
  assert c.agent.rec_scales == 'depth_head=1.0,depth_wrist=1.0,proprio=2.0,goal=2.0'
  assert c.agent.label_keys == ('label_aff', 'label_aff_mask', 'label_phase')
  assert c.agent.skill.enabled and c.agent.aff_head.enabled
  assert c.agent.switch_cost == 0.2
  phase = presets[1] if len(presets) > 1 and presets[1].startswith('legmm_') else None
  expected = {'legmm_wm': 'wm', 'legmm_bc': 'bc', 'legmm_rl': 'rl'}.get(
      phase or presets[-1], 'online')
  assert c.agent.train_phase == expected
  if 'legmm_debug' in presets:
    assert c.agent.imag_length == 8 and c.run.envs == 2
    assert c.agent.dyn.rssm.deter == 512
  else:
    assert c.agent.imag_length == 30
  if 'legmm_rl' in presets:
    assert c.agent.bc.scale_end == pytest.approx(0.1)
    assert c.agent.skill_prior.anneal_steps == 200000


@pytest.mark.parametrize('preset', ['drone_nav', 'drone_multi'])
def test_drone_presets_keep_new_switches_off(preset):
  c = load_config([preset])
  a = c.agent
  assert a.train_phase == 'online' and a.imag_enabled
  assert not a.skill.enabled and not a.aff_head.enabled
  assert not a.phase_head.enabled and not a.deliberation.enabled
  assert not a.rec_per_pixel_norm and a.rec_scales == ''
  assert a.bc.scale == 0.0 and a.skill_prior.scale == 0.0
  assert a.switch_cost == 0.0
  assert c.offline.dir == '' and c.eval.episodes == 200
