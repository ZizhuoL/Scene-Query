"""Reproduce the stated opaque-surface supervision without changing RGB data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shutil
from PIL import Image
from scenequery.common import ROOT, ART, read_json, atomic_json, sha256


def main():
    source=ART/'brush/dataset';destination=ART/'brush/dataset_opaque'
    shutil.copytree(source,destination,dirs_exist_ok=True)
    for path in (destination/'images').glob('*.png'):
        image=Image.open(path).convert('RGBA');image.putalpha(255);image.save(path)
    report=read_json(ROOT/'reports/brush/preparation.json')
    report['files']={str(p.relative_to(ROOT)):sha256(p) for p in destination.rglob('*') if p.is_file()}
    report['opacity_target']='All training RGB observations get alpha=1. No labels or held-out measurements are used.'
    atomic_json(ROOT/'reports/brush/preparation_opaque.json',report)


if __name__=='__main__':main()
