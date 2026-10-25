"""Independent small-scene checks of Gaussian geometry and occlusion."""
import numpy as np
import pytest
from scenequery.gaussian import read_ply, render_gaussians


def gaussian_data(z, rgb, alpha):
    count=len(z)
    data={'x':np.zeros(count),'y':np.zeros(count),'z':np.array(z),'opacity':np.log(np.array(alpha)/(1-np.array(alpha)))}
    for i in range(3):
        data[f'scale_{i}']=np.full(count,np.log(.1))
        data[f'f_dc_{i}']=(np.array(rgb)[:,i]-.5)/.28209479177387814
    for i in range(4):data[f'rot_{i}']=np.ones(count) if i==0 else np.zeros(count)
    return data


def test_expected_depth_includes_uncovered_probability():
    result=render_gaussians(gaussian_data([2],[[.25,.5,.75]],[.8]),np.eye(4),1,1)
    np.testing.assert_allclose(result['rgb'][0,0],[.2,.4,.6],atol=2e-6)
    assert result['depth'][0,0]==pytest.approx(2.3,abs=2e-6)
    assert result['opacity'][0,0]==pytest.approx(.8,abs=2e-6)


def test_near_surface_occludes_far_surface_independent_of_input_order():
    data=gaussian_data([2,1],[[0,0,1],[1,0,0]],[.5,.8])
    result=render_gaussians(data,np.eye(4),1,1)
    np.testing.assert_allclose(result['rgb'][0,0],[.8,0,.1],atol=2e-6)
    assert result['depth'][0,0]==pytest.approx(1.35,abs=2e-6)


def test_truncated_ply_is_rejected(tmp_path):
    path=tmp_path/'truncated.ply'
    path.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nend_header\n'+np.array([1],dtype='<f4').tobytes())
    with pytest.raises(ValueError,match='length mismatch'):read_ply(path)
