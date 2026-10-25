"""Bounded uninterrupted versus interrupted/resumed GPU training check."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import subprocess
import argparse
import numpy as np
import torch
from scenequery.common import ROOT, ART, read_json, atomic_json
from scenequery.radiance import camera_rays, render_rays
from scenequery.radiance_log import load_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/neural/rgbd_s42.json')
    parser.add_argument('--trainer',default='train_radiance.py')
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if '/' in args.trainer or '/' in args.tag:raise ValueError('Simple names required')
    suffix='_'+args.tag if args.tag else ''
    base=read_json(ROOT/args.config)
    base.update(steps=40,rays=128,samples=32,resolutions=[16,32],channels=4,hidden=32,
                log_every=10,save_every=20,retain_steps=[],pilot_render_frame=None)
    for label in ['recovery_full','recovery_resumed']:
        name=label+suffix
        config={**base,'name':name}
        path=ROOT/'config/neural'/(name+'.json')
        atomic_json(path,config)
        command=[sys.executable,str(ROOT/'scripts'/args.trainer),'--config',str(path)]
        if label=='recovery_resumed':
            subprocess.run(command+['--stop-after','20'],check=True,cwd=ROOT)
            subprocess.run(command+['--resume'],check=True,cwd=ROOT)
        else:
            subprocess.run(command,check=True,cwd=ROOT)
    a,sa=load_model(ART/'neural'/('recovery_full'+suffix)/'checkpoint.pt','cpu')
    b,sb=load_model(ART/'neural'/('recovery_resumed'+suffix)/'checkpoint.pt','cpu')
    maximum=max(float((sa['model'][k]-sb['model'][k]).abs().max()) for k in sa['model'])
    pose=read_json(ROOT/'manifests/frames.json')['frames'][27]['c2w']
    origins,directions=camera_rays(pose,32,24)
    with torch.no_grad():
        ra=render_rays(a,origins,directions,64)
        rb=render_rays(b,origins,directions,64)
    rgb_max=float((ra['rgb']-rb['rgb']).abs().max())
    depth_max=float((ra['depth']-rb['depth']).abs().max())
    atomic_json(ROOT/'reports/neural'/('recovery'+suffix+'.json'),{
        'trainer':args.trainer,'steps':40,'interruption_step':20,'device':'mps','parameter_max_abs_difference':maximum,
        'rgb_max_abs_difference':rgb_max,'depth_max_abs_difference_m':depth_max,
        'rng_equal':bool(torch.equal(sa['rng'],sb['rng'])),
        'pass':bool(rgb_max<.002 and depth_max<.002 and torch.equal(sa['rng'],sb['rng'])),
        'tolerance_declared_in_code_before_run':'Rendered RGB and camera-z depth max difference < 0.002; RNG state exactly equal.',
        'note':'MPS indexed gradient accumulation may be nondeterministic; recovery restores model, optimizer, schedule and sample RNG, not a guarantee of bitwise GPU identity.'})
    print('Recovery differences:',maximum,rgb_max,depth_max)


if __name__=='__main__':main()
