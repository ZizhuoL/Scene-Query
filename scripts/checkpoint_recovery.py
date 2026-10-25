"""Reproduce the bounded checkpoint interruption/recovery verification."""
from pathlib import Path
import sys,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scenequery.common import ROOT,ART,atomic_json

base=[sys.executable,str(ROOT/'scripts/train.py'),'--pilot','--steps','60','--max-points','20000']
for name,extra in [('recovery_full',[]),('recovery_resumed',['--stop-after','20']),('recovery_resumed',['--resume'])]:
    subprocess.run(base+['--run-name',name]+extra,cwd=ROOT,check=True)
full=np.load(ART/'recovery_full/fitted.npz');resumed=np.load(ART/'recovery_resumed/fitted.npz')
differences={key:float(abs(full[key]-resumed[key]).max()) for key in ['xyz','rgb']}
passed=all(v==0 for v in differences.values())
atomic_json(ROOT/'reports/recovery_test.json',{**differences,'pass_exact':passed,'procedure':'60 steps uninterrupted versus stop after step-20 atomic checkpoint and resume to step 60; same configuration, optimizer and RNG restored'})
if not passed:raise AssertionError(differences)
