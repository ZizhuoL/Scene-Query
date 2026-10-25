"""Bounded local Brush training with pinned inputs and recorded process costs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
import resource
import subprocess
import time
from scenequery.common import ROOT, ART, read_json, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read_json(ROOT/args.config)
    run = ART/'brush'/config['name']
    if not config['name'].replace('_', '').isalnum():
        raise ValueError('Invalid run name')
    run.mkdir(parents=True, exist_ok=True)
    if any(run.glob('export_*.ply')):
        raise FileExistsError('Preserve existing exports; use a new run identity')
    binary = ROOT/'models/brush/brush-app-aarch64-apple-darwin/brush_app'
    archive = ROOT/'models/brush/brush-v0.3.0.tar.xz'
    preparation = read_json(ROOT/'reports/brush/preparation.json')
    if sha256(archive) != preparation['upstream']['archive_sha256']:
        raise ValueError('Brush archive identity changed')
    for path, digest in preparation['files'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Prepared input changed: '+path)
    command = [str(binary), str(ART/'brush/dataset'), '--export-path', str(run)]
    for key, value in config['arguments'].items():
        command.append('--'+key.replace('_', '-'))
        if value is not True:
            command.append(str(value))
    identity = {'config': config, 'preparation_sha256': sha256(ROOT/'reports/brush/preparation.json'),
                'binary_sha256': sha256(binary), 'command': command,
                'runner_sha256': sha256(Path(__file__))}
    report_dir = ROOT/'reports/brush'/config['name']
    atomic_json(report_dir/'identity.json', identity)
    start = time.monotonic()
    status = 'failed'
    try:
        with (run/'training.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            atomic_json(run/'process.json', {'pid': process.pid, 'wrapper_pid': os.getpid()})
            try:
                code = process.wait(timeout=config['timeout_seconds'])
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                status = 'timeout'
                raise
        status = 'completed' if code == 0 else 'failed'
        if code:
            raise RuntimeError(f'Brush exited {code}; inspect {run}/training.log')
    finally:
        report = {'identity': identity, 'status': status, 'wall_seconds': time.monotonic()-start,
                  'child_peak_rss_gb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1e9,
                  'exports': {str(p.relative_to(ROOT)): sha256(p) for p in run.glob('export_*.ply')},
                  'recovery_boundary': 'PLY exports retain learned parameters, but upstream v0.3.0 does not serialize Adam or sampler state. A restart is a warm start, not exact optimizer continuation.'}
        atomic_json(report_dir/'training.json', report)
    print(status, report['wall_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
