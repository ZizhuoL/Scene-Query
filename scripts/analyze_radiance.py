"""Aggregate fixed experiments without changing selection or acceptance criteria."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'cache/matplotlib'))
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scenequery.common import ROOT, ART, read_json, atomic_json, sha256


RUNS={'rgbd_s42':'RGB + depth / seed 42','rgb_s42':'RGB loss only / seed 42','rgbd_s43':'RGB + depth / seed 43'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study');args=parser.parse_args()
    root=ROOT/'reports/neural'
    study=read_json(ROOT/'config/neural'/f'study_{args.study}.json') if args.study else None
    study_dir=root/'studies'/args.study if args.study else root
    run_names=study['runs'] if study else RUNS
    paired_name=study['paired_full'] if study else 'rgbd_s42'
    control_name=study['control'] if study else 'rgb_s42'
    second_name=study['second_seed'] if study else 'rgbd_s43'
    protocol=read_json(ROOT/'config/neural/protocol.json')
    selection=read_json(study_dir/'selection.json')
    selected=selection['selected_run']
    runs={}
    for name,label in run_names.items():
        report=read_json(root/name/'checkpoint_final.json')
        train=read_json(root/name/'training.json')
        runs[name]={**report,'label':label,'seconds':train['seconds'],'history':train['history'],
                    'peak_rss_gb':train['peak_rss_gb'],'sampled_peak_mps_driver_gb':train['sampled_peak_mps_driver_gb']}
    full=runs[selected]
    initial=read_json(root/selected/'initial_final.json')
    early=read_json(root/selected/'step_000500_final.json')
    middle=read_json(root/selected/'step_002000_final.json')
    reference=read_json(root/'surfel_reference.json')
    train=read_json(root/selected/'training.json')
    surface=read_json(root/selected/'surface.json')
    control=runs[control_name];paired=runs[paired_name]
    delta={k:[a[k]-b[k] for a,b in zip(paired['frames'],control['frames'])]
           for k in ['psnr_db','ssim','depth_mae_m','depth_within_5cm']}
    seed_delta={k:[a[k]-b[k] for a,b in zip(runs[second_name]['frames'],paired['frames'])]
                for k in ['psnr_db','ssim','depth_mae_m']}
    summarize=lambda d:{k:{'mean':float(np.mean(v)),'median':float(np.median(v)),
                           'min':float(min(v)),'max':float(max(v)),'per_view':v} for k,v in d.items()}
    criterion=protocol['acceptance'];summary=full['summary']
    gates={
        'heldout_median_psnr':summary['psnr_db']['median']>=criterion['heldout_median_psnr_min_db'],
        'heldout_worst_psnr':summary['psnr_db']['min']>=criterion['heldout_worst_psnr_min_db'],
        'heldout_median_depth_mae':summary['depth_mae_m']['median']<=criterion['heldout_median_depth_mae_max_m'],
        'psnr_gain_over_random':summary['psnr_db']['median']-initial['summary']['psnr_db']['median']>=criterion['heldout_median_psnr_gain_over_random_min_db'],
        'all_runs_substantive_training':all(read_json(root/name/'training.json')['steps']>=3000 for name in run_names),
        'recovery':read_json(root/(study.get('recovery_report','recovery.json') if study else 'recovery.json'))['pass']}
    a=torch.load(ART/'neural'/paired_name/'initial.pt',map_location='cpu',weights_only=True)['model']
    b=torch.load(ART/'neural'/control_name/'initial.pt',map_location='cpu',weights_only=True)['model']
    gates['ablation_identical_initial_parameters']=all(torch.equal(a[k],b[k]) for k in a)
    for name in run_names:
        gradient=read_json(root/name/'gradients_0.json')
        changes=read_json(root/name/'training.json')['parameter_mean_absolute_changes']
        gates[name+'_finite_nonzero_learning']=all(np.isfinite(v) and v>0 for v in gradient.values()) and all(np.isfinite(v) and v>0 for v in changes.values())
    result={'selected_run':selected,'selection':selection,'gates':gates,'numerical_acceptance_pass':all(gates.values()),
            'paired_rgbd_minus_rgb':summarize(delta),'seed43_minus_seed42':summarize(seed_delta),
            'initial_median_psnr_db':initial['summary']['psnr_db']['median'],
            'early_median_psnr_db':early['summary']['psnr_db']['median'],
            'middle_median_psnr_db':middle['summary']['psnr_db']['median'],
            'final_median_psnr_db':summary['psnr_db']['median'],
            'surfel_reference':{k:float(np.median([r[k] for r in reference['frames']])) for k in ['psnr_db','ssim','depth_mae_m']},
            'worst_psnr_frame':min(full['frames'],key=lambda r:r['psnr_db'])['frame_id'],
            'worst_depth_frame':max(full['frames'],key=lambda r:r['depth_mae_m'])['frame_id'],
            'total_full_training_seconds':sum(r['seconds'] for r in runs.values()),
            'protocol_sha256':sha256(ROOT/'config/neural/protocol.json')}
    result['study']=args.study or 'v1'
    atomic_json(study_dir/'analysis.json',result)
    if study:atomic_json(root/'analysis.json',result)
    figures=ART/'neural/figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#22865a','#c9791f','#4c6fc0']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for (name,r),color in zip(runs.items(),colors):
        axes[0,0].plot([h['step'] for h in r['history']],[h['batch_psnr'] for h in r['history']],label=r['label'],color=color)
        axes[0,1].plot(protocol['final_ids'],[x['psnr_db'] for x in r['frames']],'o-',label=r['label'],color=color)
        axes[1,0].plot(protocol['final_ids'],[x['depth_mae_m']*100 for x in r['frames']],'o-',color=color)
    axes[0,0].set(title='Training batches (not held-out metrics)',xlabel='Optimization step',ylabel='Batch PSNR / dB')
    axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='All final held-out views',xlabel='Frame ID',ylabel='PSNR / dB')
    axes[1,0].set(title='All valid reference depths; no prediction gating',xlabel='Frame ID',ylabel='Depth MAE / cm')
    axes[1,1].plot([0,500,2000,train['steps']],[result['initial_median_psnr_db'],result['early_median_psnr_db'],result['middle_median_psnr_db'],result['final_median_psnr_db']],'o-',color='#22865a')
    axes[1,1].set(title=f'Selected {selected}: retained checkpoints',xlabel='Optimization step',ylabel='Held-out median PSNR / dB')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle('SceneQuery | Random-initialized radiance-field training on M2 Max',fontsize=14)
    fig.savefig(figures/'experiments.png',dpi=160);fig.savefig(figures/'experiments.pdf');plt.close(fig)
    # Save a concise UI payload with observed values, not hand-entered metrics.
    delivery={'study':args.study or 'v1','selected_run':selected,'parameter_count':train['parameter_count'],'steps':train['steps'],
              'train_views':48,'development_ids':protocol['development_ids'],'final_ids':protocol['final_ids'],
              'surface_points':surface['points'],'default_frame_id':22,'runs':runs,
              'limitations':'One fitted scene with supplied poses and RGB-D supervision. Neighboring held-out views are correlated. Fine details are blurred; background and occlusion edges can have large depth errors. The RGB-loss-only control shares depth-derived scene bounds. Two full seeds and one control seed do not establish general statistical superiority. Search uses frozen OWLv2 and original 3D anchors; it was not retrained. The displayed neural surface is a view-sampled proxy, not a mesh.',
              'recovery_report':study.get('recovery_report','recovery.json') if study else 'recovery.json','checkpoint_sha256':train['checkpoint_sha256'],'numerical_acceptance_pass':result['numerical_acceptance_pass']}
    atomic_json(study_dir/'delivery.json',delivery)
    if study:atomic_json(root/'delivery.json',delivery)
    print(result)


if __name__=='__main__':main()
