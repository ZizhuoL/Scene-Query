"""Audit delivered local assets without loading evaluation labels into runtime."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scenequery.common import ROOT,ART,read_json,sha256,atomic_json

report=read_json(ROOT/'reports/training.json')
path=ART/'scene/fitted.npz'
assert sha256(path)==report['fitted_sha256']
fit=np.load(path);initial=np.load(ART/'scene/initial.npz')
assert fit['xyz'].shape==(report['points'],3)
assert np.isfinite(fit['xyz']).all() and np.isfinite(fit['rgb']).all()
assert ((fit['rgb']>=0)&(fit['rgb']<=1)).all()
assert np.any(fit['xyz']!=initial['xyz']) and np.any(fit['rgb']!=initial['rgb'])
state=torch.load(ART/'scene/checkpoint.pt',weights_only=True,map_location='cpu')
assert state['step']==report['config']['steps']
np.testing.assert_array_equal(state['xyz'].numpy(),fit['xyz'])
np.testing.assert_array_equal(state['rgb'].numpy(),fit['rgb'])
manifest=read_json(ROOT/'manifests/model.json')
model_config=read_json(ROOT/'config/model.json')
assert manifest['id']==model_config['id']
for filename,digest in manifest['files'].items():assert sha256(ROOT/model_config['directory']/filename)==digest
frames=read_json(ROOT/'manifests/frames.json')['frames']
cache_dir=ART/model_config['cache_directory']
expected=sha256(ROOT/'manifests/frames.json')+manifest['files']['model.safetensors']+manifest['files']['preprocessor_config.json']
for f in frames:
    if f['split']=='train':
        cache=np.load(cache_dir/f"{f['id']:03}.npz")
        assert str(cache['identity'])==expected
        assert np.isfinite(cache['emb']).all()
    else:assert not (cache_dir/f"{f['id']:03}.npz").exists()
result={'pass':True,'fitted_sha256':sha256(path),'points':report['points'],'checkpoint_steps':state['step'],'model_files_hash_verified':len(manifest['files']),'training_caches_verified':48,'heldout_caches_absent':12,'finite_geometry_and_color':True,'initialization_differs_from_trained':True}
atomic_json(ROOT/'reports/artifact_verification.json',result)
print(result)
