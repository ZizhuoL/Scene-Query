"""Export a neural-field surface proxy in the original metric world coordinates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import os
import numpy as np
import torch
from scenequery.common import ROOT, ART, read_json, sha256, atomic_json
from scenequery.radiance import camera_rays, render_image
from scenequery.radiance_log import load_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--device', default='mps')
    args=p.parse_args()
    checkpoint=ART/'neural'/args.run/'checkpoint.pt'
    model,state=load_model(checkpoint,args.device)
    torch.set_num_threads(4)
    frames=[f for f in read_json(ROOT/'manifests/frames.json')['frames'] if f['split']=='train']
    xyz=[];rgb=[];start=time.perf_counter()
    for f in frames:
        pred=render_image(model,f['c2w'],160,120,192,512)
        origins,directions=camera_rays(f['c2w'],160,120)
        valid=(pred['opacity'].ravel()>.98)&(pred['depth'].ravel()>.4)&(pred['depth'].ravel()<3.5)
        xyz.append((origins.numpy()+directions.numpy()*pred['depth'].reshape(-1,1))[valid])
        rgb.append(pred['rgb'].reshape(-1,3)[valid])
    xyz=np.concatenate(xyz);rgb=np.concatenate(rgb)
    _,indices=np.unique(np.floor(xyz/.008).astype('int32'),axis=0,return_index=True)
    if len(indices)>220000:
        indices=np.random.default_rng(42).choice(indices,220000,replace=False)
    indices=np.sort(indices)
    target=ART/'neural'/args.run/'surface.npz'
    with open(str(target)+'.tmp','wb') as f:
        np.savez_compressed(f,xyz=xyz[indices],rgb=rgb[indices])
    os.replace(str(target)+'.tmp',target)
    atomic_json(ROOT/'reports/neural'/args.run/'surface.json',{
        'checkpoint_sha256':sha256(checkpoint),'surface_sha256':sha256(target),
        'points':len(indices),'seconds':time.perf_counter()-start,
        'method':'Expected termination depth from camera-ray neural rendering; opacity > 0.98; 8 mm voxel deduplication; deterministic point cap.',
        'input':'Trained radiance field and 48 training camera poses only. No source RGB, source depth or surfel positions used.',
        'limitation':'A view-sampled surface proxy, not an isosurface mesh. RGB can depend on view. Search selections remain the original independently fitted scene points in the same metric coordinates.'})
    print('Exported',len(indices),'neural points',flush=True)


if __name__=='__main__':
    main()
