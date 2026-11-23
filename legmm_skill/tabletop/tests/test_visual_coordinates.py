import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tabletop.visual_diagnostics import camera_optical_point,require_fresh_frame


def test_camera_world_and_optical_axes_preserve_metric_units():
    origin=np.array([1.,2.,3.]);q=np.array([1.,0.,0.,0.])
    assert camera_optical_point(origin+[.071,0,0],origin,q)==pytest.approx([0,0,.071])
    assert camera_optical_point(origin+[0,.2,.3],origin,q)==pytest.approx([-.2,-.3,0])
    turn=Rotation.from_euler('z',90,degrees=True).as_quat();q=np.r_[turn[3],turn[:3]]
    assert camera_optical_point(origin+[0,.4,0],origin,q)==pytest.approx([0,0,.4],abs=1e-12)


def test_stale_render_frame_is_rejected():
    require_fresh_frame(None,1);require_fresh_frame(1,2)
    with pytest.raises(RuntimeError):require_fresh_frame(2,2)
    with pytest.raises(RuntimeError):require_fresh_frame(2,1)
