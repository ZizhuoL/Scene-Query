"""Create isolated continuation outputs, preserving the delivered evidence ledger."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import shutil
from scenequery.common import ROOT,ART


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='reproduce');args=parser.parse_args()
    if not args.name.replace('_','').replace('-','').isalnum():raise ValueError('Use a simple directory name')
    required=[ROOT/'.venv/bin/python',ROOT/'manifests/frames.json',ART/'scene/fitted.npz',ROOT/'models/brush/brush-v0.3.0.tar.xz']
    if not all(p.exists() for p in required):raise RuntimeError('Prepare the base project data/environment/surfel assets and download Brush first, as described in README.')
    target=ART/'reproductions'/args.name
    if target.exists():raise FileExistsError('Preserve an existing reproduction; choose a new name')
    target.mkdir(parents=True)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    for directory in ['scenequery','scripts','config','manifests','web','licenses','tests']:
        shutil.copytree(ROOT/directory,target/directory,ignore=ignore)
    for name in ['Makefile','run.sh','requirements.txt','requirements.lock.txt','package.json','package-lock.json']:
        shutil.copy2(ROOT/name,target/name)
    for name in ['.venv','data','node_modules']:(target/name).symlink_to(ROOT/name,target_is_directory=True)
    (target/'models').mkdir()
    for path in (ROOT/'models').iterdir():
        if path.name=='brush':shutil.copytree(path,target/'models/brush')
        else:(target/'models'/path.name).symlink_to(path,target_is_directory=path.is_dir())
    (target/'artifacts').mkdir()
    for name in ['scene','semantics_owlv2']:(target/'artifacts'/name).symlink_to(ART/name,target_is_directory=True)
    frozen=['reports/training.json','reports/neural/experiment_freeze.json','reports/brush/experiment_freeze.json',
            'reports/brush/opaque/experiment_freeze.json','reports/brush/dense/experiment_freeze.json',
            'reports/neural/studies/v1/experiment_freeze.json','reports/neural/studies/v2/experiment_freeze.json',
            'reports/neural/studies/v3/experiment_freeze.json']
    for name in frozen:
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    print('Created isolated source/output directory:',target)
    print('Shared read-only inputs: project environment, raw data, frozen OWLv2 and original surfel reference.')
    print('Run there: make train-neural evaluate-neural train-brush evaluate-brush')
    print('Do not run the base data/model/surfel preparation targets here; they use shared inputs.')


if __name__=='__main__':main()
