"""Run the frozen acceptance queries against held-out, approximate reference boxes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import resource
import argparse
import numpy as np
from PIL import Image,ImageDraw
from scenequery.common import ROOT,DATA,ART,read_json,atomic_json,sha256
from scenequery.geometry import visibility
from scenequery.search import SceneSearch


def iou(a,b):
    a=np.array(a);b=np.array(b)
    intersection=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(intersection/(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersection+1e-9))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='v1');args=parser.parse_args()
    if '/' in args.run_id or args.run_id in ('.','..'):raise ValueError('Use a simple run ID')
    report_dir=ROOT/'reports/evaluations'/args.run_id;report_dir.mkdir(parents=True,exist_ok=True)
    config=read_json(ROOT/'config/evaluation.json');annotations=read_json(ROOT/'config/evaluation_annotations.json')
    frames=read_json(ROOT/'manifests/frames.json')['frames'];rubric=config['rubric']
    frozen={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'config/evaluation.json',ROOT/'config/evaluation_annotations.json',ROOT/'config/search.json',ROOT/'config/model.json',ROOT/'scenequery/search.py',ART/'scene/fitted.npz',ROOT/'manifests/model.json']}
    freeze_path=report_dir/'evaluation_freeze.json'
    if freeze_path.exists() and read_json(freeze_path)!=frozen:raise ValueError('Frozen evaluation inputs changed. Preserve the previous run and disclose any new protocol.')
    atomic_json(freeze_path,frozen)
    start=time.perf_counter();engine=SceneSearch();results=[]
    directory=ART/f'evaluation_{args.run_id}';directory.mkdir(exist_ok=True)
    for item in config['final_positive_queries']:
        query=item['query'];r=engine.query(query)
        atomic_json(directory/(query.replace(' ','_')+'.json'),r)
        truth=[a for a in annotations['annotations'] if a['object']==item['object']]
        scored=[]
        for c in r['candidates']:
            views=[]
            for a in truth:
                f=frames[a['frame_id']];d=np.array(Image.open(DATA/f['depth']),dtype='float32')/5000
                ids=np.array(c['point_ids']);uv,visible=visibility(engine.xyz[ids],np.array(f['c2w']),d)
                uv=uv[visible]
                bbox=[*uv.min(0),*uv.max(0)] if len(uv) else [0,0,0,0]
                center=np.median(uv,axis=0) if len(uv) else [-1,-1]
                inside=bool(a['bbox'][0]<=center[0]<=a['bbox'][2] and a['bbox'][1]<=center[1]<=a['bbox'][3])
                overlap=iou(bbox,a['bbox'])
                passed=bool(len(uv)>=rubric['selected_points_minimum'] and inside and overlap>=rubric['projected_selection_bbox_iou_minimum'])
                views.append({'frame_id':f['id'],'visible_points':len(uv),'selection_bbox':[float(x) for x in bbox],'reference_bbox':a['bbox'],'iou':overlap,'centroid_inside':inside,'pass':passed})
            scored.append({'candidate_id':c['id'],'views':views,'passing_views':sum(v['pass'] for v in views),'support_views':c['support_views']})
        best=max(scored,key=lambda x:(x['passing_views'],sum(v['iou'] for v in x['views'])),default=None)
        passed=bool(best and best['passing_views']>=rubric['views_per_positive_query_minimum'] and best['support_views']>=rubric['source_evidence_views_minimum'])
        record={**item,'status':r['status'],'latency_ms':r['latency_ms'],'pass':passed,'candidates':scored}
        results.append(record);print(query,passed,scored,flush=True)
        if best:
            candidate=r['candidates'][best['candidate_id']];ids=np.array(candidate['point_ids'])
            comparison=Image.new('RGB',(1280,520));draw=ImageDraw.Draw(comparison)
            for j,a in enumerate(truth):
                f=frames[a['frame_id']];comparison.paste(Image.open(DATA/f['rgb']),(j*640,40))
                d=np.array(Image.open(DATA/f['depth']),dtype='float32')/5000
                uv,visible=visibility(engine.xyz[ids],np.array(f['c2w']),d)
                for x,y in uv[visible]:draw.ellipse((x+j*640-1,y+39,x+j*640+1,y+41),fill=(0,255,140))
                b=a['bbox'];draw.rectangle((b[0]+j*640,b[1]+40,b[2]+j*640,b[3]+40),outline=(255,220,100),width=2)
                draw.text((j*640+8,10),f'{query} / held-out frame {f["id"]} / yellow: reference / green: 3D selection',fill='white')
            comparison.save(directory/(query.replace(' ','_')+'.jpg'))
    negatives=[];ambiguous=[]
    for category,queries in [('absent',config['final_absent_queries']),('ambiguous',config['final_ambiguous_queries'])]:
        for query in queries:
            r=engine.query(query);atomic_json(directory/(query.replace(' ','_')+'.json'),r)
            record={'query':query,'status':r['status'],'pass':r['status']==('no_evidence' if category=='absent' else 'ambiguous'),'regions':len(r['candidates']),'latency_ms':r['latency_ms']}
            (negatives if category=='absent' else ambiguous).append(record);print(record,flush=True)
    passing=sum(r['pass'] for r in results);cats=sorted(set(r['object'] for r in results if r['pass']))
    latencies=[r['latency_ms'] for r in results+negatives+ambiguous]
    summary={'positive_pass':passing,'positive_total':len(results),'distinct_categories_pass':cats,'absent_rejections':sum(r['pass'] for r in negatives),'ambiguous_flagged':sum(r['pass'] for r in ambiguous),'uncached_query_p95_seconds':float(np.percentile(latencies,95)/1000),'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,'seconds':time.perf_counter()-start}
    summary['localization_rubric_pass']=bool(passing>=rubric['positive_queries_pass_minimum'] and len(cats)>=rubric['distinct_object_categories_pass_minimum'] and summary['absent_rejections']>=rubric['absent_queries_rejected_minimum'] and summary['ambiguous_flagged']>=rubric['ambiguous_queries_flagged_minimum'])
    atomic_json(report_dir/'localization_evaluation.json',{'summary':summary,'positive':results,'absent':negatives,'ambiguous':ambiguous,'annotation_provenance':annotations['annotator'],'limitations':'Small project-specific acceptance sample; not a benchmark or independent human assessment. No final label entered fit/cache/query code. These are new views of the fitted scene, not new scenes. This is a repeated engineering acceptance check of the preserved query protocol, not an untouched independent final trial.'})
    print(summary,flush=True)


if __name__=='__main__':main()
