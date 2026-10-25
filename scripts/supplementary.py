"""Report the separately predeclared extra phrasings; do not replace acceptance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scenequery.common import ROOT,DATA,ART,read_json,atomic_json
from scenequery.search import SceneSearch
from scenequery.geometry import visibility
from scripts.evaluate import iou

engine=SceneSearch()
frames=read_json(ROOT/'manifests/frames.json')['frames']
annotations=read_json(ROOT/'config/evaluation_annotations.json')['annotations']
results=[]
for item in read_json(ROOT/'config/supplementary_queries.json')['queries']:
    r=engine.query(item['query']);trials=[]
    for c in r['candidates']:
        good=0;views=[]
        for a in [a for a in annotations if a['object']==item['object']]:
            f=frames[a['frame_id']];depth=np.asarray(Image.open(DATA/f['depth']),dtype='float32')/5000
            uv,seen=visibility(engine.xyz[c['point_ids']],np.array(f['c2w']),depth);uv=uv[seen]
            box=[*uv.min(0),*uv.max(0)] if len(uv) else [0,0,0,0]
            center=np.median(uv,axis=0) if len(uv) else [-1,-1]
            overlap=iou(box,a['bbox']);inside=a['bbox'][0]<=center[0]<=a['bbox'][2] and a['bbox'][1]<=center[1]<=a['bbox'][3]
            passed=bool(len(uv)>=30 and inside and overlap>=.25);good+=passed
            views.append({'frame_id':f['id'],'iou':overlap,'pass':passed})
        trials.append({'candidate':c['id'],'passing_views':good,'views':views})
    record={**item,'status':r['status'],'pass':any(t['passing_views']>=2 for t in trials),'trials':trials,'latency_ms':r['latency_ms']}
    results.append(record);print(record,flush=True)
atomic_json(ROOT/'reports/supplementary_evaluation.json',{'queries':results,'pass_count':sum(r['pass'] for r in results),'total':len(results),'acceptance_replacement':False,'purpose':'Fresh phrasings predeclared before OWLv2 acceptance; same object reference boxes and geometric checks. Never replaces original failed runs or rubric.'})
