"""Frozen text encoding and visibility-aware 3D association. No evaluation inputs."""
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import expit
from .common import ROOT, DATA, ART, read_json, sha256
from .geometry import visibility
from .detector import settings,load_detector


def nms(boxes,scores,threshold=.4,limit=4):
    order=np.argsort(scores)[::-1];keep=[]
    while len(order) and len(keep)<limit:
        i=order[0];keep.append(int(i));order=order[1:]
        if not len(order):break
        lo=np.maximum(boxes[i,:2],boxes[order,:2]);hi=np.minimum(boxes[i,2:],boxes[order,2:])
        inter=np.maximum(hi-lo,0).prod(-1)
        area=(boxes[:,2:]-boxes[:,:2]).clip(0).prod(-1)
        iou=inter/(area[i]+area[order]-inter+1e-8)
        order=order[iou<threshold]
    return keep


class SceneSearch:
    def __init__(self,scene='scene',limit=0):
        torch.set_num_threads(4)
        self.lock=threading.Lock()
        self.config=read_json(ROOT/'config/search.json')
        self.model_config=settings();cache_dir=ART/self.model_config['cache_directory']
        training=[f for f in read_json(ROOT/'manifests/frames.json')['frames'] if f['split']=='train']
        self.frames=[f for f in training if (cache_dir/f"{f['id']:03}.npz").exists()]
        if not limit and len(self.frames)!=len(training):raise ValueError('Semantic cache is incomplete. Run scripts/cache_semantics.py.')
        if limit:self.frames=self.frames[:limit]
        fitted=np.load(ART/scene/'fitted.npz');self.xyz=fitted['xyz'];self.rgb=fitted['rgb']
        self.processor,full=load_detector()
        # Only text components are retained at query time.
        backbone=getattr(full,self.model_config['family'])
        self.text_model=backbone.text_model;self.text_projection=backbone.text_projection
        for p in self.text_model.parameters():p.requires_grad_(False)
        for p in self.text_projection.parameters():p.requires_grad_(False)
        self.emb=[];self.shift=[];self.scale=[];self.boxes=[];self.uv=[];self.visible=[];self.images=[]
        manifest=read_json(ROOT/'manifests/model.json')
        expected_identity=sha256(ROOT/'manifests/frames.json')+manifest['files']['model.safetensors']+manifest['files']['preprocessor_config.json']
        for f in self.frames:
            c=np.load(cache_dir/f"{f['id']:03}.npz")
            if str(c['identity'])!=expected_identity:raise ValueError('Semantic cache identity mismatch. Recompute the cache.')
            self.emb.append(c['emb']);self.shift.append(c['shift'][:,0]);self.scale.append(c['scale'][:,0])
            b=c['boxes'];box_scale=[640]*4 if self.model_config['family']=='owlv2' else [640,480,640,480]
            self.boxes.append(np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*box_scale)
            d=np.asarray(Image.open(DATA/f['depth']),dtype=np.float32)/5000
            uv,visible=visibility(self.xyz,np.array(f['c2w']),d,self.config['depth_tolerance_m'])
            self.uv.append(uv);self.visible.append(visible)
            self.images.append(np.asarray(Image.open(DATA/f['rgb'])))
        self.emb=np.stack(self.emb);self.shift=np.stack(self.shift);self.scale=np.stack(self.scale)
        self.query_cache={}
        self.region_cache=[OrderedDict() for _ in self.frames]

    def frame_regions(self,fi,row):
        """Independent source-image work; cache only geometry, never text scores."""
        eligible=np.flatnonzero(row>=self.config['detection_score_min'])
        records=[]
        if not len(eligible):return records
        ids=eligible[nms(self.boxes[fi][eligible],row[eligible])]
        uv=self.uv[fi];vis=self.visible[fi];cache=self.region_cache[fi]
        for i in ids:
            box=np.clip(self.boxes[fi][i],[0,0,0,0],[639,479,639,479])
            x1,y1,x2,y2=box
            if x2-x1<8 or y2-y1<8:continue
            if int(i) in cache:
                point_ids=cache[int(i)];cache.move_to_end(int(i))
            else:
                inside=vis&(uv[:,0]>=x1)&(uv[:,0]<=x2)&(uv[:,1]>=y1)&(uv[:,1]<=y2)
                point_ids=np.flatnonzero(inside)
                if len(point_ids)<30:continue
                if self.config['grabcut']:
                    mask=np.zeros((240,320),np.uint8)
                    rect=(int(x1/2),int(y1/2),max(2,int((x2-x1)/2)),max(2,int((y2-y1)/2)))
                    try:
                        # OpenCV RNG is thread-local; reset for each region.
                        cv2.setRNGSeed(42)
                        cv2.grabCut(cv2.resize(self.images[fi],(320,240)),mask,rect,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_RECT)
                        pix=np.clip((uv[point_ids]/2).astype(int),[0,0],[319,239])
                        foreground=(mask[pix[:,1],pix[:,0]]&1).astype(bool)
                        if foreground.sum()>=30:point_ids=point_ids[foreground]
                    except cv2.error:pass
                cache[int(i)]=point_ids
                while sum(v.nbytes for v in cache.values())>1024*1024:cache.popitem(last=False)
            records.append({'frame_id':self.frames[fi]['id'],'frame_index':fi,'box':box.round(2).tolist(),'score':float(row[i]),'points':point_ids})
        return records

    def encode(self,text):
        tokens=self.processor.tokenizer([self.model_config['query_template'].format(text)],padding=True,truncation=True,max_length=16,return_tensors='pt')
        with torch.inference_mode():
            pooled=self.text_model(**tokens).pooler_output
            q=self.text_projection(pooled);q=q/(q.norm(dim=-1,keepdim=True)+1e-6)
            q=q[0]
        return q.numpy()

    def query(self,text):
        text=' '.join(text.strip().split())
        if not text or len(text)>120:raise ValueError('Use an object name or short description (1–120 characters).')
        with self.lock:
            return self._query(text)

    def _query(self,text):
        started=time.perf_counter()
        key=text.lower()
        if key in self.query_cache:
            return {**self.query_cache[key],'latency_ms':round((time.perf_counter()-started)*1000,2),'cached':True}
        if key in self.config['underspecified_queries']:
            return {'query':text,'status':'ambiguous','message':'Please name a more specific object. This description does not identify a unique scene region.','candidates':[],'latency_ms':round((time.perf_counter()-started)*1000,2),'cached':False}
        q=self.encode(text)
        scores=expit((np.einsum('fnd,d->fn',self.emb,q)+self.shift)*self.scale)
        votes=np.zeros(len(self.xyz),dtype='int32');weights=np.zeros(len(self.xyz),dtype='float32')
        evidence=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for regions in pool.map(lambda fi:self.frame_regions(fi,scores[fi]),range(len(self.frames))):
                hit=np.zeros(len(self.xyz),dtype=bool);w=np.zeros(len(self.xyz),dtype='float32')
                for e in regions:
                    ids=e['points'];hit[ids]=True;w[ids]=np.maximum(w[ids],e['score'])
                evidence.extend(regions);votes+=hit;weights+=w
        visible_count=np.sum(self.visible,axis=0)
        selected=np.flatnonzero((votes>=self.config['min_support_views']) & (votes/np.maximum(visible_count,1)>=self.config['min_visible_support_fraction']))
        candidates=[]
        if len(selected)>=self.config['min_points']:
            # Spatial connectivity of the actual selected world points; no semantic labels.
            pairs=cKDTree(self.xyz[selected]).query_pairs(self.config['component_radius_m'],output_type='ndarray')
            graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(selected),len(selected)))
            n,labels=connected_components(graph,directed=False)
            for label in range(n):
                point_ids=selected[labels==label]
                if len(point_ids)<self.config['min_points']:continue
                ev=[]
                for e in evidence:
                    overlap=np.intersect1d(point_ids,e['points']).size
                    if overlap>=max(15,len(point_ids)*.08):
                        ev.append({k:v for k,v in e.items() if k not in ('points','frame_index')}|{'overlap_points':overlap})
                ev=sorted(ev,key=lambda e:e['score']*e['overlap_points'],reverse=True)
                unique=[];seen=set()
                for e in ev:
                    if e['frame_id'] not in seen:unique.append(e);seen.add(e['frame_id'])
                if len(unique)<self.config['min_support_views']:continue
                center=np.median(self.xyz[point_ids],axis=0)
                lo,hi=np.quantile(self.xyz[point_ids],[.02,.98],axis=0)
                candidates.append({'point_ids':point_ids.tolist(),'point_count':len(point_ids),'center':center.tolist(),'bounds':[lo.tolist(),hi.tolist()],'support_views':len(unique),'evidence':unique[:6],'rank_value':float(weights[point_ids].mean()*np.sqrt(len(point_ids))),'similarity_score':float(max(e['score'] for e in unique))})
        candidates.sort(key=lambda x:x['rank_value'],reverse=True)
        if candidates:
            candidates=[c for c in candidates if c['rank_value']>=candidates[0]['rank_value']*.25][:4]
        for i,c in enumerate(candidates):c['id']=i
        status='no_evidence' if not candidates else ('ambiguous' if len(candidates)>1 else 'match')
        messages={'no_evidence':'No region has enough consistent visual evidence for this query. This does not prove the object is absent.','ambiguous':'Several scene regions may match. Inspect the candidates and source photos, or use a more specific description.','match':'A scene region is supported by multiple source views. Inspect the photos to verify the match.'}
        result={'query':text,'status':status,'message':messages[status],'candidates':candidates,'latency_ms':round((time.perf_counter()-started)*1000,2),'cached':False,'score_note':'Model scores are uncalibrated similarity evidence, not probabilities.','max_detector_score':float(scores.max())}
        self.query_cache[key]=result
        if len(self.query_cache)>64:self.query_cache.pop(next(iter(self.query_cache)))
        return result
