"""Optimize surfel XYZ/RGB against posed training RGB-D; save resumable Adam state."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
import time
import resource
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scenequery.common import ROOT, DATA, ART, atomic_json, read_json, sha256
from scenequery.geometry import K, unproject, visibility


def initialize(frames, voxel, max_points):
    points=[];colors=[]
    y,x=np.mgrid[0:480:3,0:640:3];uv=np.c_[x.ravel(),y.ravel()]
    for f in frames:
        d=np.asarray(Image.open(DATA/f['depth']),dtype=np.float32)/5000
        c=np.asarray(Image.open(DATA/f['rgb']),dtype=np.float32)/255
        z=d[y,x].ravel();good=(z>.4)&(z<3.5)
        points.append(unproject(uv[good],z[good],np.array(f['c2w'])))
        colors.append(c[y,x].reshape(-1,3)[good])
    xyz=np.concatenate(points).astype('float32');rgb=np.concatenate(colors)
    # First actual observation in each voxel; no artificial perturbation.
    _,i=np.unique(np.floor(xyz/voxel).astype('int32'),axis=0,return_index=True)
    i=np.sort(i)
    if len(i)>max_points:i=np.sort(np.random.default_rng(42).choice(i,max_points,replace=False))
    return xyz[i],rgb[i]


def atomic_checkpoint(path,value):
    torch.save(value,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-points',type=int,default=120000)
    parser.add_argument('--stop-after',type=int,default=0,help='Simulate interruption after this step; preserve the full run configuration.')
    parser.add_argument('--run-name',default=None,help='Optional project-local artifact subdirectory for recovery checks.')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    start=time.perf_counter();device='cpu'
    # CPU grid_sample backward is deterministic and fast for this small fit.
    frames=[f for f in read_json(ROOT/'manifests/frames.json')['frames'] if f['split']=='train']
    if args.pilot:frames=frames[:8]
    run_name=args.run_name or ('pilot' if args.pilot else 'scene')
    if '/' in run_name or run_name in ('.','..'):raise ValueError('Use a simple run name')
    run=ART/run_name;run.mkdir(exist_ok=True)
    config={'method':'RGB-D photometric/depth surfel optimization','steps':args.steps,'seed':42,'voxel_m':.008,'max_points':args.max_points,'xyz_lr':.0003,'rgb_lr':.01,'device':device,'batch_views':4,'depth_gate_m':.06,'max_offset_m':.025,'frames_manifest_sha256':sha256(ROOT/'manifests/frames.json'),'train_frame_ids':[f['id'] for f in frames]}
    xyz0,rgb0=initialize(frames,.008,args.max_points)
    np.savez_compressed(run/'initial.npz',xyz=xyz0,rgb=rgb0)
    xyz=torch.nn.Parameter(torch.tensor(xyz0,device=device));rgb=torch.nn.Parameter(torch.tensor(rgb0,device=device))
    anchor=torch.tensor(xyz0,device=device)
    optimizer=torch.optim.Adam([{'params':[xyz],'lr':config['xyz_lr']},{'params':[rgb],'lr':config['rgb_lr']}])
    images=[];poses=[]
    for f in frames:
        color=np.asarray(Image.open(DATA/f['rgb']),dtype=np.float32)/255
        depth=np.asarray(Image.open(DATA/f['depth']),dtype=np.float32)/5000
        images.append(torch.tensor(np.concatenate([color,depth[:,:,None]],axis=-1).transpose(2,0,1)))
        poses.append(np.array(f['c2w'],dtype='float32'))
    images=torch.stack(images).to(device);poses=torch.tensor(np.stack(poses),device=device)
    checkpoint=run/'checkpoint.pt';history=[];first=0
    if args.resume and checkpoint.exists():
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        if state['config']!=config:raise ValueError('Resume configuration differs')
        with torch.no_grad():xyz.copy_(state['xyz']);rgb.copy_(state['rgb'])
        optimizer.load_state_dict(state['optimizer']);history=state['history'];first=state['step']
        torch.set_rng_state(state['rng'])
    for step in range(first,args.steps):
        vi=torch.randperm(len(frames))[:4];p=poses[vi]
        camera=torch.einsum('bni,bij->bnj',xyz[None]-p[:,None,:3,3],p[:,:3,:3])
        z=camera[:,:,2];uv=camera[:,:,:2]/z.clamp_min(.01)[:,:,None]*525+torch.tensor([319.5,239.5])
        grid=(uv/torch.tensor([639.,479.])*2-1)[:,:,None,:]
        samples=F.grid_sample(images[vi],grid,align_corners=True,padding_mode='zeros').squeeze(-1).transpose(1,2)
        depth=samples[:,:,3];target=samples[:,:,:3]
        mask=((z>.4)&(z<3.5)&(grid[:,:,0].abs().max(-1).values<.98)&(depth>.4)&((z-depth).abs()<.06)).detach()
        depth_loss=(z[mask]-depth[mask]).abs().mean()
        photo_loss=(rgb[None].expand(len(vi),-1,-1)[mask]-target[mask]).abs().mean()
        anchor_loss=((xyz-anchor)**2).mean()
        loss=depth_loss+photo_loss*.1+anchor_loss*10
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            xyz.copy_(torch.maximum(torch.minimum(xyz,anchor+.025),anchor-.025));rgb.clamp_(0,1)
        if step%10==0 or step==args.steps-1:
            row={'step':step+1,'loss':float(loss.detach()),'depth_l1_m':float(depth_loss.detach()),'rgb_l1':float(photo_loss.detach()),'visible_samples':int(mask.sum()),'elapsed_seconds':time.perf_counter()-start}
            history.append(row);print(row,flush=True)
        if (step+1)%20==0 or step==args.steps-1:
            atomic_checkpoint(checkpoint,{'xyz':xyz.detach(),'rgb':rgb.detach(),'optimizer':optimizer.state_dict(),'step':step+1,'rng':torch.get_rng_state(),'history':history,'config':config})
            if args.stop_after and step+1>=args.stop_after:
                print('Stopped after atomic checkpoint for recovery check.',flush=True)
                return
    outxyz=xyz.detach().cpu().numpy();outrgb=rgb.detach().cpu().numpy()
    with open(run/'fitted.npz.tmp','wb') as h:np.savez_compressed(h,xyz=outxyz,rgb=outrgb)
    os.replace(run/'fitted.npz.tmp',run/'fitted.npz')
    report={'config':config,'points':len(xyz0),'seconds':time.perf_counter()-start,'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,'xyz_mean_change_m':float(np.linalg.norm(outxyz-xyz0,axis=1).mean()),'rgb_mean_change':float(np.abs(outrgb-rgb0).mean()),'history':history,'fitted_sha256':sha256(run/'fitted.npz'),'initialization':'deterministic first RGB-D observation per 8 mm voxel, capped by seed-42 sample','training':'Adam gradients into XYZ via depth and RGB projection/sampling, and into RGB via photometric loss; cameras fixed; no semantic gradients'}
    report_name=f'training_{args.run_name}.json' if args.run_name else ('training_pilot.json' if args.pilot else 'training.json')
    atomic_json(ROOT/'reports'/report_name,report)
    atomic_json(run/'config.json',config)
    print('Saved',run,flush=True)


if __name__=='__main__':main()
