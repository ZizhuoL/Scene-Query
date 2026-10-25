"""Frozen camera-only prediction; original RGB/depth enter metrics afterward."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scenequery.common import ROOT, DATA, ART, read_json, sha256, atomic_json
from scenequery.radiance import RadianceField, render_image


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = state['identity']['config']
    model = RadianceField(state['model']['bounds'], config['resolutions'], config['channels'], config['hidden'])
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state


def ssim(a, b):
    """Gaussian SSIM (sigma 1.5, 11-pixel window), channel mean, valid interior."""
    blur = lambda x: gaussian_filter(x, sigma=(1.5, 1.5, 0), truncate=3.3333333333)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    score = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return float(score[5:-5, 5:-5].mean())


def compare(prediction, frame, width, height):
    rgb = np.asarray(Image.open(DATA/frame['rgb']).resize((width, height), Image.Resampling.BOX), dtype='float32')/255
    depth = np.asarray(Image.open(DATA/frame['depth']).resize((width, height), Image.Resampling.NEAREST), dtype='float32')/5000
    valid = (depth > .4) & (depth < 3.5)
    error = np.abs(prediction['depth'][valid]-depth[valid])
    return {'frame_id': frame['id'], 'psnr_db': float(-10*np.log10(np.mean((prediction['rgb']-rgb)**2))),
            'ssim': ssim(prediction['rgb'], rgb), 'depth_mae_m': float(error.mean()),
            'depth_median_m': float(np.median(error)), 'depth_p90_m': float(np.quantile(error, .9)),
            'depth_within_5cm': float((error < .05).mean()), 'reference_depth_coverage': float(valid.mean()),
            'opacity_coverage': float((prediction['opacity'] > .95).mean())}, rgb, depth


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--checkpoint', default='checkpoint.pt')
    p.add_argument('--split', choices=['development', 'final'], default='development')
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if '/' in args.run or '/' in args.checkpoint:
        raise ValueError('Simple artifact names required')
    protocol = read_json(ROOT/'config/neural/protocol.json')
    ids = protocol[args.split+'_ids']
    frames = read_json(ROOT/'manifests/frames.json')['frames']
    checkpoint = ART/'neural'/args.run/args.checkpoint
    model, state = load_model(checkpoint, args.device)
    torch.set_num_threads(4)
    key = args.checkpoint.removesuffix('.pt')+'_'+args.split
    out = ART/'neural'/args.run/key
    out.mkdir(exist_ok=True)
    report_path = ROOT/'reports/neural'/args.run/(key+'.json')
    identity = {'checkpoint_sha256': sha256(checkpoint), 'protocol_sha256': sha256(ROOT/'config/neural/protocol.json'),
                'renderer_sha256': sha256(ROOT/'scenequery/radiance.py'), 'evaluator_sha256': sha256(Path(__file__)),
                'width': 320, 'height': 240, 'samples': 192, 'ids': ids}
    if report_path.exists():
        if read_json(report_path)['identity'] != identity:
            raise ValueError('Refusing to overwrite changed frozen evaluation')
        print('Verified existing report', report_path)
        return
    rows = []
    start = time.perf_counter()
    for fid in ids:
        frame = frames[fid]
        render_start = time.perf_counter()
        prediction = render_image(model, frame['c2w'], 320, 240, 192, 512)
        row, rgb, depth = compare(prediction, frame, 320, 240)
        row['render_seconds'] = time.perf_counter()-render_start
        rows.append(row)
        np.savez_compressed(out/f'frame_{fid:02}.npz', **prediction)
        # Original, neural prediction, RGB absolute error, depth absolute error.
        error = np.abs(prediction['rgb']-rgb).mean(-1)
        de = np.abs(prediction['depth']-depth)
        de[(depth <= .4) | (depth >= 3.5)] = 0
        heat = lambda v: np.stack([np.clip(v*3, 0, 1), np.clip(v*3-1, 0, 1), np.clip(v*3-2, 0, 1)], -1)
        parts = [rgb, prediction['rgb'], heat(error*3), heat(de/.25)]
        labels = ['Held-out original', f'Neural / {row["psnr_db"]:.2f} dB', 'RGB absolute error x3', 'Depth error / 0.25 m scale']
        sheet = Image.new('RGB', (1280, 272), (20, 28, 24))
        draw = ImageDraw.Draw(sheet)
        for i, (part, label) in enumerate(zip(parts, labels)):
            sheet.paste(Image.fromarray((part.clip(0, 1)*255).astype('uint8')), (i*320, 32))
            draw.text((i*320+8, 10), f'{fid:02} | {label}', fill='white')
        sheet.save(out/f'frame_{fid:02}.jpg')
        Image.fromarray((prediction['rgb'].clip(0,1)*255).astype('uint8')).save(out/f'render_{fid:02}.png')
        print(row, flush=True)
    metrics = ['psnr_db', 'ssim', 'depth_mae_m', 'depth_median_m', 'depth_p90_m', 'depth_within_5cm', 'opacity_coverage']
    summary = {k: {'mean': float(np.mean([r[k] for r in rows])), 'median': float(np.median([r[k] for r in rows])),
                   'min': float(min(r[k] for r in rows)), 'max': float(max(r[k] for r in rows))} for k in metrics}
    atomic_json(report_path, {'identity': identity, 'step': state['step'], 'split': args.split, 'frames': rows,
                             'summary': summary, 'seconds': time.perf_counter()-start,
                             'boundary': 'Predictions use learned weights and camera rays only. RGB/depth are loaded afterward for metrics. No prediction-dependent depth mask.'})
    print(summary, flush=True)


if __name__ == '__main__':
    main()
