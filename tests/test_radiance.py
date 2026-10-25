import numpy as np
import torch
from scenequery.radiance import RadianceField, camera_rays, render_rays, composite, sample_planes
from scenequery.geometry import project


def test_ray_camera_roundtrip_at_scaled_pixel_centers():
    pose=np.eye(4);pose[:3,3]=[.2,-.1,.5]
    origins,directions=camera_rays(pose,16,12)
    xyz=(origins+directions*1.4).numpy()
    uv,z=project(xyz,pose)
    y,x=np.mgrid[:12,:16]
    expected=np.c_[((x+.5)*640/16-.5).ravel(),((y+.5)*480/12-.5).ravel()]
    assert np.max(np.abs(uv-expected))<1e-4
    assert np.max(np.abs(z-1.4))<1e-6


def test_compositing_matches_constant_density_slab():
    sigma=torch.ones(1,100)*2
    z=(torch.arange(100)[None]+.5)/100
    color=torch.ones(1,100,3)*.7
    result=composite(sigma,color,z,torch.tensor([[0.,0.,1.]]),torch.ones(1))
    expected=1-np.exp(-2*.995)
    assert abs(result['opacity'].item()-expected)<1e-6
    assert abs(result['rgb'][0,0].item()-.7*expected)<1e-6
    assert torch.all(result['weights']>=0)


def test_density_and_appearance_receive_photometric_gradients():
    torch.manual_seed(10)
    model=RadianceField([[-2,-2,-2],[2,2,2]],resolutions=[8],channels=2,hidden=16)
    origins,directions=camera_rays(np.eye(4),4,3)
    result=render_rays(model,origins,directions,samples=12)
    (result['rgb']-.2).square().mean().backward()
    for module in (model.planes,model.trunk,model.density,model.color):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters())>0
    assert torch.isfinite(result['depth']).all()


def test_portable_sampler_matches_reference_values_and_gradients():
    torch.manual_seed(3)
    p=torch.randn(3,4,8,8,requires_grad=True)
    q=p.detach().clone().requires_grad_(True)
    grid=torch.randn(3,31,2)*1.2
    out=sample_planes(p,grid)
    ref=torch.nn.functional.grid_sample(q,grid[:,None],align_corners=True,padding_mode='border')
    ref=ref[:,:,0].permute(2,0,1).reshape(31,-1)
    assert torch.allclose(out,ref,atol=1e-6)
    out.square().sum().backward();ref.square().sum().backward()
    assert torch.allclose(p.grad,q.grad,atol=2e-5)


def test_checkpoint_loader_preserves_density_parameterization(tmp_path):
    from scenequery.radiance_log import LogDensityField, load_model
    model=LogDensityField([[-2,-2,-2],[2,2,2]],resolutions=[8],channels=2,hidden=16)
    config={'resolutions':[8],'channels':2,'hidden':16,'density_activation':'exp'}
    path=tmp_path/'model.pt'
    torch.save({'identity':{'config':config},'model':model.state_dict(),'step':0},path)
    restored,_=load_model(path,'cpu')
    assert isinstance(restored,LogDensityField)
    origins,directions=camera_rays(np.eye(4),4,3)
    with torch.no_grad():
        expected=render_rays(model,origins,directions,12)
        actual=render_rays(restored,origins,directions,12)
    assert torch.equal(expected['rgb'],actual['rgb'])
    assert torch.equal(expected['depth'],actual['depth'])
