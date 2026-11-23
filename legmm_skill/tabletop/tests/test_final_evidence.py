import hashlib,json,shutil
from types import SimpleNamespace
import pytest
from tabletop.final_test import frozen_samples


def test_fixed_independent_manifest_rejects_missing_or_changed_states(tmp_path):
    files={}
    for seed in range(20000,20256,32):
        path=tmp_path/f'sampling_{seed}.json'
        path.write_text(json.dumps([dict(accepted=True,state=dict(seed=i)) for i in range(seed,seed+32)]))
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'fixed_manifest.json').write_text(json.dumps(dict(completed=True,seed_range=[20000,20255],files=files)))
    assert len(frozen_samples(tmp_path))==64
    (tmp_path/'sampling_20000.json').write_text('[]')
    with pytest.raises(ValueError,match='states changed'):frozen_samples(tmp_path)


def test_actual_pose_audit_distinguishes_side_grasp_top_grasp_and_wrist_heading():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tabletop.final_pose_audit import pose_angles
    obj=Rotation.from_euler('z',90,degrees=True).as_matrix()
    shoulder=np.array([0.,0.,.6]);position=np.array([.7,0.,.7])
    side=pose_angles(np.eye(3),obj,shoulder,position)
    assert side['horizontal_approach_tilt_deg']==pytest.approx(0)
    assert side['closing_axis_error_deg']==pytest.approx(0)
    assert side['wrist_heading_compensation_deg']==pytest.approx(0)
    yaw=pose_angles(Rotation.from_euler('z',20,degrees=True).as_matrix(),obj,shoulder,position)
    assert yaw['wrist_heading_compensation_deg']==pytest.approx(20)
    top=pose_angles(Rotation.from_euler('y',-90,degrees=True).as_matrix(),obj,shoulder,position)
    assert top['horizontal_approach_tilt_deg']==pytest.approx(90)
    assert top['approach_normal_error_deg']==pytest.approx(90)
    assert top['wrist_heading_compensation_deg'] is None


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='Local ffmpeg unavailable')
def test_video_encodes_already_acquired_frames_without_second_observation(tmp_path):
    import torch
    from tabletop.video_evidence import RecordingVisual
    calls=[]
    images=torch.zeros(2,2,2,128,128);rgb=torch.ones(2,270,480,3,dtype=torch.uint8)*100
    def observe(scene):calls.append(1);return images,rgb,None
    guard=SimpleNamespace(check=lambda size:calls.append(size))
    recording=RecordingVisual(SimpleNamespace(observe=observe),guard);recording.begin(tmp_path,'synthetic_unit',1)
    for _ in range(2):
        actual=recording.observe(None)
        assert actual[0] is images
    recording.finish(dict(episodes=[dict(seed=1,success=True,reason='success',steps=32),dict(seed=2,success=False,reason='timeout',steps=32)]))
    assert calls==[1,1,20_000_000,20_000_000]
    assert len(list(tmp_path.glob('*.mp4')))==2
    assert not recording.frames
