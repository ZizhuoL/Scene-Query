"""Evaluate the preserved surfel baseline under the neural full-image metric mask."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scenequery.common import ROOT,ART,read_json,atomic_json,sha256
from scripts.inspect_geometry import render
from scripts.evaluate_radiance import compare


def main():
    protocol=read_json(ROOT/'config/neural/protocol.json')
    frames=read_json(ROOT/'manifests/frames.json')['frames']
    data=np.load(ART/'scene/fitted.npz')
    rows=[]
    for fid in protocol['final_ids']:
        rgb,depth=render(data['xyz'],data['rgb'],np.array(frames[fid]['c2w']))
        opacity=np.isfinite(depth).astype('float32')
        depth=np.where(np.isfinite(depth),depth,3.5).astype('float32')
        prediction={'rgb':np.asarray(Image.fromarray(rgb).resize((320,240),Image.Resampling.BOX),dtype='float32')/255,
                    'depth':np.asarray(Image.fromarray(depth).resize((320,240),Image.Resampling.NEAREST)),
                    'opacity':np.asarray(Image.fromarray(opacity).resize((320,240),Image.Resampling.NEAREST))}
        row,_,_=compare(prediction,frames[fid],320,240);rows.append(row)
    atomic_json(ROOT/'reports/neural/surfel_reference.json',{
        'frames':rows,'surfel_sha256':sha256(ART/'scene/fitted.npz'),
        'protocol_sha256':sha256(ROOT/'config/neural/protocol.json'),
        'method':'Preserved original surfel renderer, 5x5 pixel footprint at 640x480, area RGB reduction to 320x240. Missing predicted depth assigned 3.5m; all valid target depths scored.',
        'caveat':'Not matched compute or initialization. Original surfels use depth initialization and direct point colors; neural model starts randomly. This is a product reference, not an algorithm ranking.'})


if __name__=='__main__':main()
