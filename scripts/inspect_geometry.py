"""Z-buffer held-out projections with error statistics and original/render comparisons."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from PIL import Image,ImageDraw
from scenequery.common import ROOT, DATA, ART, read_json,atomic_json
from scenequery.geometry import project


def render(xyz,rgb,c2w,radius=2):
    uv,z=project(xyz,c2w);pix=np.rint(uv).astype(int)
    good=(z>.2)&(pix[:,0]>=0)&(pix[:,0]<640)&(pix[:,1]>=0)&(pix[:,1]<480)
    ids=np.flatnonzero(good);pix=pix[good]
    index=np.full(640*480,-1,dtype=int);buffer=np.full(640*480,np.inf,dtype='float32')
    # Pixel depth wins over draw order, including splat footprint.
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            xy=pix+[dx,dy];valid=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<480)
            ii=ids[valid];linear=xy[valid,1]*640+xy[valid,0]
            order=np.argsort(z[ii])[::-1];linear=linear[order];ii=ii[order]
            closer=z[ii]<buffer[linear];buffer[linear[closer]]=z[ii[closer]];index[linear[closer]]=ii[closer]
    out=np.full((640*480,3),22,dtype='uint8');seen=index>=0
    out[seen]=(rgb[index[seen]].clip(0,1)*255).astype('uint8')
    return out.reshape(480,640,3),buffer.reshape(480,640)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',default='scene');args=p.parse_args()
    frames=[f for f in read_json(ROOT/'manifests/frames.json')['frames'] if f['split']=='heldout']
    fitted=np.load(ART/args.scene/'fitted.npz');initial=np.load(ART/args.scene/'initial.npz')
    out=ART/args.scene/'inspection';out.mkdir(exist_ok=True)
    results=[]
    for f in frames:
        source=np.array(Image.open(DATA/f['rgb']))
        depth=np.array(Image.open(DATA/f['depth']),dtype='float32')/5000
        row={'frame_id':f['id']}
        for name,points in [('initial',initial),('fitted',fitted)]:
            image,z=render(points['xyz'],points['rgb'],np.array(f['c2w']))
            valid=(depth>.4)&(depth<3.5)&np.isfinite(z)
            errors=np.abs(z[valid]-depth[valid])
            color=np.abs(image[valid].astype(float)/255-source[valid].astype(float)/255)
            row[name]={'overlap_fraction':float(valid.mean()),'depth_median_m':float(np.median(errors)),'depth_p90_m':float(np.quantile(errors,.9)),'rgb_mae':float(color.mean())}
            if name=='fitted':
                comparison=Image.new('RGB',(1280,510));comparison.paste(Image.fromarray(source),(0,30));comparison.paste(Image.fromarray(image),(640,30))
                d=ImageDraw.Draw(comparison);d.text((12,10),f"Held-out source frame {f['id']}",fill='white');d.text((652,10),'Optimized surfels / depth-buffered projection',fill='white')
                comparison.save(out/f"heldout_{f['id']:02}.jpg")
        results.append(row)
    atomic_json(ROOT/f'reports/geometry_{args.scene}.json',{'frames':results,'median_of_heldout_depth_medians_m':float(np.median([r['fitted']['depth_median_m'] for r in results])),'method':'nearest-depth 5x5 point footprint; all overlapping valid depth pixels, no residual gate in evaluation','caveat':'Coverage and holes must be interpreted with errors; RGB-D poses supplied by dataset; no unseen-scene claim.'})
    print(results)


if __name__=='__main__':main()
