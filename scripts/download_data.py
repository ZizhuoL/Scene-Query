"""Download the official public sequence with resume, validation and safe extraction."""
from pathlib import Path
import subprocess
import tarfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenequery.common import ROOT, atomic_json, sha256

URL = 'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz'
SIZE = 344011403


def main():
    dest = ROOT / 'data/rgbd_dataset_freiburg1_desk.tgz'
    dest.parent.mkdir(exist_ok=True)
    partial = dest.with_suffix('.tgz.part')
    if not dest.exists():
        if not partial.exists() or partial.stat().st_size != SIZE:
            subprocess.run(['curl', '-fL', '--retry', '3', '--max-time', '900', '-C', '-', URL, '-o', str(partial)], check=True)
        if partial.stat().st_size != SIZE:
            raise ValueError('Archive size differs from inspected release')
        partial.rename(dest)
    manifest_path = ROOT / 'manifests/dataset.json'
    digest = sha256(dest)
    if manifest_path.exists():
        import json
        if json.loads(manifest_path.read_text())['sha256'] != digest:
            raise ValueError('Archive hash differs from pinned input')
    with tarfile.open(dest) as archive:
        archive.extractall(dest.parent, filter='data')
    atomic_json(manifest_path, {
        'id': 'TUM-RGBD-freiburg1-desk-2011-09-30', 'url': URL,
        'bytes': dest.stat().st_size, 'sha256': digest,
        'license': 'CC-BY-4.0',
        'license_source': 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license',
        'specification': 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats',
        'citation': 'Sturm, Engelhard, Endres, Burgard, Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. IROS 2012.',
        'redistribution': 'Raw data and derived media excluded from Git; preserve attribution in local app and derived results.'
    })
    print(digest, dest)


if __name__ == '__main__':
    main()
