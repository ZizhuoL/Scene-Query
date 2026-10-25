"""Fetch only the pinned public model files into the ignored project directory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
from scenequery.common import ROOT,read_json
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import snapshot_download

config=read_json(ROOT/'config/model.json')
snapshot_download(config['id'],revision=config['revision'],local_dir=ROOT/config['directory'],allow_patterns=['*.json','*.txt','*.safetensors','README.md'])
print('Pinned frozen model available:',config['id'],config['revision'])
