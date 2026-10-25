"""One development query through actual text model, geometry and source projection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import resource
import numpy as np
from PIL import Image,ImageDraw
from scenequery.search import SceneSearch
from scenequery.common import ROOT,DATA,ART,atomic_json

t=time.perf_counter();engine=SceneSearch(scene='pilot',limit=3);r=engine.query('monitor')
report={**r,'model_id':engine.model_config['id'],'load_and_query_seconds':time.perf_counter()-t,'peak_rss_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9}
atomic_json(ROOT/'reports/query_pilot.json',report)
out=Image.new('RGB',(1280,480));draw=ImageDraw.Draw(out)
for j in range(2):
    f=engine.frames[j];out.paste(Image.open(DATA/f['rgb']),(j*640,0))
    for c in r['candidates']:
        ix=np.array(c['point_ids']);ix=ix[engine.visible[j][ix]]
        for x,y in engine.uv[j][ix]:draw.ellipse((x+j*640-1,y-1,x+j*640+1,y+1),fill=(20,255,180))
out.save(ART/'pilot_monitor_projection.jpg')
print({k:v for k,v in report.items() if k!='candidates'})
print([(c['point_count'],c['support_views']) for c in r['candidates']])
