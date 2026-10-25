"""Development selection, then frozen final Gaussian camera-only evaluation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import numpy as np
from PIL import Image, ImageDraw
from scenequery.common import ROOT, ART, read_json, atomic_json, sha256
from scenequery.gaussian import read_ply, render_gaussians
from scripts.evaluate_radiance import compare


def evaluate(run, checkpoint, split, save_images):
    protocol = read_json(ROOT/'config/neural/protocol.json')
    frames = read_json(ROOT/'manifests/frames.json')['frames']
    path = ART/'brush'/run/checkpoint
    out = path.parent/(path.stem+'_'+split)
    out.mkdir(exist_ok=True)
    report_path = ROOT/'reports/brush'/run/(out.name+'.json')
    identity = {'checkpoint_sha256': sha256(path), 'renderer_sha256': sha256(ROOT/'scenequery/gaussian.py'),
                'compositor_sha256': sha256(ROOT/'scenequery/gaussian_blend.cpp'),
                'evaluator_sha256': sha256(Path(__file__)), 'width': 320, 'height': 240,
                'ids': protocol[split+'_ids'], 'protocol_sha256': sha256(ROOT/'config/neural/protocol.json')}
    if report_path.exists():
        previous = read_json(report_path)
        if previous['identity'] != identity: raise ValueError('Changed frozen evaluation identity')
        return previous
    data = read_ply(path); rows = []
    for fid in identity['ids']:
        start = time.perf_counter()
        prediction = render_gaussians(data, frames[fid]['c2w'])
        row, rgb, depth = compare(prediction, frames[fid], 320, 240)
        row['render_seconds'] = time.perf_counter()-start
        if split == 'development':
            native = np.asarray(Image.open(path.parent/f'eval_{int(path.stem.split("_")[-1])}'/f'frame_{fid:02}.png'), dtype='float32')/255
            difference = prediction['rgb']-native
            row['native_rgb_mae'] = float(abs(difference).mean())
            row['native_vs_reference_psnr_db'] = float(-10*np.log10((difference*difference).mean()))
            if row['native_rgb_mae'] > .005 or row['native_vs_reference_psnr_db'] < 35:
                raise ValueError('Reference renderer differs materially from native Brush')
        rows.append(row)
        if save_images:
            np.savez_compressed(out/f'frame_{fid:02}.npz', **prediction)
            Image.fromarray((prediction['rgb']*255).astype('uint8')).save(out/f'render_{fid:02}.png')
            error = abs(prediction['rgb']-rgb).mean(-1)
            de = abs(prediction['depth']-depth); de[(depth <= .4) | (depth >= 3.5)] = 0
            heat = lambda v: np.stack([np.clip(v*3,0,1),np.clip(v*3-1,0,1),np.clip(v*3-2,0,1)],-1)
            parts = [rgb,prediction['rgb'],heat(error*3),heat(de/.25)]
            labels = ['Held-out original',f'Gaussian / {row["psnr_db"]:.2f} dB','RGB absolute error x3','Depth error / 0.25 m scale']
            sheet = Image.new('RGB',(1280,272),(20,28,24));draw=ImageDraw.Draw(sheet)
            for i,(part,label) in enumerate(zip(parts,labels)):
                sheet.paste(Image.fromarray((part.clip(0,1)*255).astype('uint8')),(i*320,0))
                draw.text((i*320+5,248),label,fill='white')
            sheet.save(out/f'frame_{fid:02}.jpg',quality=94)
        print(checkpoint,split,row,flush=True)
    metrics = ['psnr_db','ssim','depth_mae_m','depth_median_m','depth_p90_m','depth_within_5cm','opacity_coverage']
    summary = {k:{'mean':float(np.mean([r[k] for r in rows])),
                  'median':float(np.median([r[k] for r in rows])),
                  'min':float(min(r[k] for r in rows)), 'max':float(max(r[k] for r in rows))} for k in metrics}
    report = {'identity':identity,'frames':rows,'summary':summary,'gaussian_count':len(data['x']),
              'depth_definition':'Sum of alpha-compositing weights times Gaussian-center camera z; residual transmittance at 3.5 m. All reference-valid pixels enter depth error.'}
    atomic_json(report_path,report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='opaque_s42');parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    if not args.run.replace('_','').isalnum(): raise ValueError('Invalid run')
    freeze=read_json(ROOT/'reports/brush/opaque/experiment_freeze.json')
    for path,digest in freeze['files'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen input changed: '+path)
    reports={}
    for path in sorted((ART/'brush'/args.run).glob('export_*.ply')):
        reports[path.name]=evaluate(args.run,path.name,'development',False)
    candidates=[name for name,r in reports.items() if int(Path(name).stem.split('_')[-1])>=3000 and r['summary']['depth_mae_m']['mean']<=.12]
    if not candidates:raise RuntimeError('No full checkpoint satisfies development depth rule')
    selected=max(candidates,key=lambda name:reports[name]['summary']['psnr_db']['mean'])
    selection={'run':args.run,'checkpoint':selected,'checkpoint_sha256':sha256(ART/'brush'/args.run/selected),
               'rule':freeze['selection'],'development_summaries':{name:r['summary'] for name,r in reports.items()},
               'declared_before_final_render':True,'protocol_deviation':'One explicit opacity-supervision revision after the RGB-only Gaussian failure; all earlier failures retained.'}
    target=ROOT/'reports/brush/opaque/selection.json'
    if target.exists() and read_json(target)!=selection:raise ValueError('Cannot change frozen selection')
    atomic_json(target,selection)
    if args.final:
        result=evaluate(args.run,selected,'final',True)
        s=result['summary'];a=read_json(ROOT/'config/neural/protocol.json')['acceptance']
        gates={'median_psnr':s['psnr_db']['median']>=a['heldout_median_psnr_min_db'],
               'worst_psnr':s['psnr_db']['min']>=a['heldout_worst_psnr_min_db'],
               'median_depth_mae':s['depth_mae_m']['median']<=a['heldout_median_depth_mae_max_m']}
        atomic_json(ROOT/'reports/brush/opaque/acceptance.json',{'selection':selection,'gates':gates,'pass':all(gates.values()),'final':result})
        print('Final gates',gates,flush=True)


if __name__=='__main__':main()
