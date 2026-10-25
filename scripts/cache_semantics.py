"""Cache pinned frozen detector features/boxes. This is inference, not training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
import time
import resource
import numpy as np
import torch
from PIL import Image
from scenequery.common import ROOT, DATA, ART, atomic_json, read_json, sha256
from scenequery.detector import settings,load_detector


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    torch.set_num_threads(4)
    start=time.perf_counter()
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model_config=settings();path=ROOT/model_config['directory']
    processor,model=load_detector();model.to(device)
    for param in model.parameters():param.requires_grad_(False)
    frames=[f for f in read_json(ROOT/'manifests/frames.json')['frames'] if f['split']=='train']
    if args.limit:frames=frames[:args.limit]
    out=ART/model_config['cache_directory'];out.mkdir(exist_ok=True)
    identities={k:sha256(path/k) for k in ['model.safetensors','config.json','preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt']}
    identity=sha256(ROOT/'manifests/frames.json')+identities['model.safetensors']+identities['preprocessor_config.json']
    times=[]
    with torch.inference_mode():
        for f in frames:
            target=out/f"{f['id']:03}.npz"
            if target.exists():
                with np.load(target) as prev:
                    if str(prev['identity'])==identity:continue
            t=time.perf_counter()
            pixels=processor(images=Image.open(DATA/f['rgb']),return_tensors='pt')['pixel_values'].to(device)
            fm,_=model.image_embedder(pixels)
            feats=fm.reshape(fm.shape[0],-1,fm.shape[-1])
            emb=model.class_head.dense0(feats)
            emb=emb/(torch.linalg.norm(emb,dim=-1,keepdim=True)+1e-6)
            shift=model.class_head.logit_shift(feats)
            scale=model.class_head.elu(model.class_head.logit_scale(feats))+1
            boxes=model.box_predictor(feats,fm)
            with open(str(target)+'.tmp','wb') as handle:
                np.savez_compressed(handle,emb=emb[0].cpu().numpy(),shift=shift[0].cpu().numpy(),scale=scale[0].cpu().numpy(),boxes=boxes[0].cpu().numpy(),identity=identity)
            os.replace(str(target)+'.tmp',target)
            times.append(time.perf_counter()-t)
            print(f"Frame {f['id']:02}: {times[-1]:.2f}s",flush=True)
    atomic_json(ROOT/'manifests/model.json',{**model_config,'files':identities,'frozen':True,'training_in_this_project':False})
    name=f"semantic_{'pilot' if args.limit else 'cache'}_{model_config['family']}.json"
    atomic_json(ROOT/'reports'/name,{'device':device,'frames':len(frames),'seconds':time.perf_counter()-start,'new_frame_seconds':times,'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,'identity':identity,'operation':'frozen image inference; no optimization or evaluation labels'})


if __name__=='__main__':main()
