"""Bundle preserved scientific experiments and the selected production scene."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scenequery.common import ROOT,ART,read_json,atomic_json,sha256
from scenequery.gaussian import read_ply,render_gaussians
from scenequery.radiance import camera_rays


def main():
    neural=read_json(ROOT/'reports/neural/delivery.json')
    production=read_json(ROOT/'reports/brush/dense/acceptance.json')
    selection=production['selection'];run=selection['run'];checkpoint=selection['checkpoint']
    path=ART/'brush'/run/checkpoint
    if sha256(path)!=selection['checkpoint_sha256']:raise ValueError('Production checkpoint changed')
    data=read_ply(path)
    frames=read_json(ROOT/'manifests/frames.json')['frames']
    positions=[];colors=[]
    for frame in [f for f in frames if f['split']=='train']:
        prediction=render_gaussians(data,frame['c2w'],320,240)
        origin,direction=camera_rays(frame['c2w'],320,240)
        depth=prediction['depth'].reshape(-1)
        valid=(prediction['opacity'].reshape(-1)>.98)&(depth>.3)&(depth<3.5)
        positions.append(origin.numpy()[valid]+direction.numpy()[valid]*depth[valid,None])
        colors.append(prediction['rgb'].reshape(-1,3)[valid])
    xyz=np.concatenate(positions);rgb=np.concatenate(colors)
    _,indices=np.unique(np.floor(xyz/.008).astype('int32'),axis=0,return_index=True)
    indices=np.sort(indices)
    if len(indices)>220000:indices=np.sort(np.random.default_rng(42).choice(indices,220000,replace=False))
    output=ART/'production';output.mkdir(exist_ok=True)
    with (output/'surface.npz.tmp').open('wb') as handle:np.savez_compressed(handle,xyz=xyz[indices].astype('float32'),rgb=rgb[indices].astype('float32'))
    (output/'surface.npz.tmp').replace(output/'surface.npz')
    atomic_json(ROOT/'reports/brush/dense/surface.json',{'count':len(indices),'checkpoint_sha256':sha256(path),
        'surface_sha256':sha256(output/'surface.npz'),'render_camera_ids':[f['id'] for f in frames if f['split']=='train'],
        'method':'Camera-only expected-depth/RGB rendering at 48 training poses; opacity >.98; 8 mm voxel; cap 220k, seed 42. No observed RGB/depth read by export.',
        'limitation':'Interactive point proxy of the learned Gaussian scene; not a mesh or live Gaussian rasterizer. SH view dependence is reduced to sampled colors.'})
    runs={}
    for name,r in neural['runs'].items():
        runs[name]={**r,'family':'neural field','train_views':48,'steps':6000,
                    'artifact_directory':f'artifacts/neural/{name}/checkpoint_final'}
    brush_studies=[('', 'full_s42','Gaussian / RGB only',48),('opaque/','opaque_s42','Gaussian / opaque surfaces',48),('dense/','dense_s42','Gaussian / dense observations',525)]
    for folder,name,label,count in brush_studies:
        study=read_json(ROOT/f'reports/brush/{folder}acceptance.json')
        train=read_json(ROOT/f'reports/brush/{name}/training.json')
        export=study['selection']['checkpoint'];steps=int(Path(export).stem.split('_')[-1])
        runs[name]={**study['final'],'label':label,'family':'Gaussian','train_views':count,'steps':steps,
                    'seconds':train['wall_seconds'],'history':[], 'quality_pass':study['pass'],
                    'artifact_directory':f'artifacts/brush/{name}/{Path(export).stem}_final'}
    prep=read_json(ROOT/'reports/brush/preparation_dense.json')
    delivery={**neural,'selected_run':run,'runs':runs,'parameter_count':production['final']['gaussian_count']*38,
              'gaussian_count':production['final']['gaussian_count'],'steps':runs[run]['steps'],'train_views':prep['train_views'],
              'surface_points':len(indices),'surface_asset':'artifacts/production/surface.npz',
              'display_method':'Trained Gaussian surface proxy','numerical_acceptance_pass':production['pass'],
              'controlled_curve_runs':list(neural['runs']),'controlled_curve_steps':6000,
              'custom_field_quality_pass':False,'default_frame_id':37,
              'limitations':'The interactive scene is a point proxy sampled from a locally trained Gaussian model, not live Gaussian rasterization. Original search anchors use 48 RGB-D training views. The production reconstruction uses 525 observations outside a ±0.10 s guard around all held-out images; it tests dense scene interpolation, not new-scene or wide-baseline generalization. The original random-initialized neural-field study failed its color floors and remains visible. Gaussian runs differ in opacity supervision and input density; only the original field loss ablation has matched architecture, data and ray budgets. Repeated evaluation and all plan deviations are disclosed. Similarity is not confidence.'}
    atomic_json(ROOT/'reports/reconstruction_delivery.json',delivery)
    # Preserve a standalone scientific figure, with data density and selection explicit.
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for folder,name,label,count in brush_studies:
        sel=read_json(ROOT/f'reports/brush/{folder}selection.json')
        items=sorted(sel['development_summaries'].items(),key=lambda x:int(Path(x[0]).stem.split('_')[-1]))
        step=[int(Path(k).stem.split('_')[-1]) for k,_ in items]
        axes[0].plot(step,[r['psnr_db']['mean'] for _,r in items],label=f'{label} ({count} views)')
        axes[1].plot(step,[100*r['depth_mae_m']['mean'] for _,r in items],label=label)
    axes[0].set(xlabel='Optimization step',ylabel='Development mean PSNR / dB',title='Color prediction on 3 development cameras')
    axes[1].set(xlabel='Optimization step',ylabel='Development mean depth MAE / cm',title='Expected-depth prediction')
    for axis in axes:axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(output/'development.png',dpi=180);fig.savefig(output/'development.pdf');plt.close(fig)
    print('Delivered production surface:',len(indices),'points; quality pass:',production['pass'],flush=True)


if __name__=='__main__':main()
