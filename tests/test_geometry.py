import numpy as np
from scenequery.geometry import K,unproject,project,visibility


def test_camera_projection_and_pose_direction():
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[1.2,-.4,3]
    pixels=np.array([[0,0],[319.5,239.5],[639,479]],dtype=float);depth=np.array([.5,1,2])
    points=unproject(pixels,depth,pose);uv,z=project(points,pose)
    np.testing.assert_allclose(uv,pixels,atol=1e-8);np.testing.assert_allclose(z,depth)
    np.testing.assert_allclose(points[1],[1.2,-.4,4])


def test_visibility_rejects_occluded_missing_and_behind_camera():
    depth=np.ones((480,640));depth[100,100]=0
    uv=np.array([[320,240],[320,240],[100,100],[320,240]])
    points=unproject(uv,np.array([1,1.3,1,-1]),np.eye(4))
    _,seen=visibility(points,np.eye(4),depth)
    assert seen.tolist()==[True,False,False,False]
