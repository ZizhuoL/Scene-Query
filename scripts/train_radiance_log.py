"""Versioned training-loss revision; original study source remains immutable."""
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
from scenequery.common import ROOT, DATA, ART, read_json, atomic_json, sha256
from scenequery.radiance import camera_rays, render_rays, render_image
from scenequery.radiance_log import LogDensityField as RadianceField


from scripts.train_radiance import atomic_save, load_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/neural/rgbd_log_s42.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=0)
    args = parser.parse_args()
    config_path = ROOT/args.config
    config = read_json(config_path)
    name = config['name']
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid experiment name')
    run = ART/'neural'/name
    run.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT/'reports/neural'/name
    report_dir.mkdir(parents=True, exist_ok=True)
    device = config['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable to this process; use a native macOS terminal with Metal access. No silent CPU fallback.')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    generator = torch.Generator().manual_seed(config['seed']+100)
    start = time.perf_counter()
    data, bounds, train_ids = load_training(config['width'], config['height'], device)
    model = RadianceField(bounds, config['resolutions'], config['channels'], config['hidden']).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.planes.parameters(), 'lr': config['plane_lr'], 'base_lr': config['plane_lr']},
        {'params': list(model.trunk.parameters())+list(model.density.parameters())+list(model.color.parameters()),
         'lr': config['decoder_lr'], 'base_lr': config['decoder_lr']}])
    identity = {'config': config, 'frames_sha256': sha256(ROOT/'manifests/frames.json'),
                'code': {p: sha256(ROOT/p) for p in ('scenequery/radiance.py', 'scenequery/radiance_log.py', 'scripts/train_radiance.py', 'scripts/train_radiance_log.py')},
                'train_ids': train_ids, 'torch': str(torch.__version__)}
    checkpoint = run/'checkpoint.pt'
    first = 0
    history = []
    elapsed = 0.
    initial_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Resume identity changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['rng'])
        first, history, elapsed = state['step'], state['history'], state['elapsed_seconds']
    elif checkpoint.exists():
        raise FileExistsError('Experiment exists; resume it or use a new name')
    else:
        atomic_save(run/'initial.pt', {'model': initial_model, 'identity': identity, 'step': 0})
    atomic_json(report_dir/'identity.json', identity)
    peak_mps = 0
    def save(step):
        state = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'optimizer': optimizer.state_dict(), 'step': step, 'rng': generator.get_state(),
                 'history': history, 'identity': identity,
                 'elapsed_seconds': elapsed + time.perf_counter()-start}
        atomic_save(checkpoint, state)
        if step in config.get('retain_steps', []):
            atomic_save(run/f'step_{step:06}.pt', state)
    for step in range(first, config['steps']):
        indices = torch.randint(len(data[0]), (config['rays'],), generator=generator).to(device)
        origins, directions, target, depth = [x[indices] for x in data]
        jitter = torch.rand((config['rays'], config['samples']), generator=generator).to(device)
        prediction = render_rays(model, origins, directions, config['samples'], jitter)
        color_loss = F.mse_loss(prediction['rgb'], target)
        valid = (depth > .4) & (depth < 3.5)
        depth_loss = F.smooth_l1_loss(prediction['depth'][valid], depth[valid], beta=.05)
        # Match a noisy measured termination distribution, without constraining invalid depth.
        distribution = torch.exp(-.5*((prediction['z'][valid]-depth[valid, None])/config['termination_sigma_m']).square())
        distribution = distribution / distribution.sum(-1, keepdim=True).clamp_min(1e-8)
        termination = -(distribution*torch.log(prediction['weights'][valid]+1e-8)).sum(-1).mean()
        tv = model.tv_loss()
        loss = color_loss + config['depth_weight']*depth_loss + config['termination_weight']*termination + config['tv_weight']*tv
        decay = .1**(step/max(config['steps']-1, 1))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*decay
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == first:
            gradients = {k: float(p.grad.norm()) for k, p in model.named_parameters() if p.grad is not None}
            atomic_json(report_dir/f'gradients_{first}.json', gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step == 0 or (step+1)%config['log_every']==0 or step+1==config['steps']:
            if device == 'mps':
                torch.mps.synchronize()
                peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())
            row = {'step': step+1, 'loss': float(loss.detach()), 'rgb_mse': float(color_loss.detach()),
                   'batch_psnr': float(-10*torch.log10(color_loss.detach())),
                   'depth_loss': float(depth_loss.detach()), 'termination_loss': float(termination.detach()),
                   'seconds': elapsed+time.perf_counter()-start}
            if not np.isfinite(row['loss']):
                raise RuntimeError('Nonfinite loss')
            history.append(row)
            print(row, flush=True)
            atomic_json(report_dir/'progress.json', {'history': history, 'peak_mps_driver_bytes': peak_mps})
        if (step+1)%config['save_every']==0 or step+1==config['steps'] or (args.stop_after and step+1==args.stop_after):
            save(step+1)
            if args.stop_after and step+1>=args.stop_after:
                print('Stopped after atomic checkpoint.', flush=True)
                return
    changes = {k: float((v.detach().cpu()-initial_model[k]).abs().mean()) for k, v in model.named_parameters()}
    report = {'identity': identity, 'steps': config['steps'], 'parameter_count': sum(p.numel() for p in model.parameters()),
              'seconds': elapsed+time.perf_counter()-start, 'peak_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,
              'sampled_peak_mps_driver_gb': peak_mps/1e9, 'parameter_mean_absolute_changes': changes,
              'checkpoint_sha256': sha256(checkpoint), 'history': history,
              'initialization': 'Random plane features and random MLP weights. Training depth determines only shared scene bounds before fitting.',
              'inference': 'Camera rays and trained parameters only; no target RGB or depth.'}
    atomic_json(report_dir/'training.json', report)
    if config.get('pilot_render_frame') is not None:
        frame = read_json(ROOT/'manifests/frames.json')['frames'][config['pilot_render_frame']]
        render_start = time.perf_counter()
        result = render_image(model, frame['c2w'], 160, 120, config['samples'], 512)
        Image.fromarray((result['rgb'].clip(0, 1)*255).astype('uint8')).save(run/'pilot_render.png')
        atomic_json(report_dir/'render.json', {'seconds': time.perf_counter()-render_start,
                                             'finite': bool(np.isfinite(result['depth']).all()),
                                             'frame': frame['id'], 'width': 160, 'height': 120})
    print('Saved', run, flush=True)


if __name__ == '__main__':
    main()
