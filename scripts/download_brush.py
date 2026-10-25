"""Download and verify the pinned official native Apple silicon training binary."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tarfile
import urllib.request
from scenequery.common import ROOT,sha256

URL='https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz'
DIGEST='65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c'


def main():
    directory=ROOT/'models/brush';directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'brush-v0.3.0.tar.xz'
    if not archive.exists():
        temporary=archive.with_suffix('.download')
        with urllib.request.urlopen(URL,timeout=60) as source,temporary.open('wb') as target:
            while block:=source.read(1024*1024):target.write(block)
        if sha256(temporary)!=DIGEST:raise ValueError('Official release digest mismatch')
        temporary.replace(archive)
    if sha256(archive)!=DIGEST:raise ValueError('Cached archive digest mismatch')
    with tarfile.open(archive) as handle:handle.extractall(directory,filter='data')
    print('Verified Brush v0.3.0 for Apple silicon.')


if __name__=='__main__':main()
