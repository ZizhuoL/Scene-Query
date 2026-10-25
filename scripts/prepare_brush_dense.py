"""Dense production fit with explicit temporal exclusion around all evaluation views."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation,Slerp
from scenequery.common import ROOT,DATA,ART,read_json,atomic_json,sha256
from scripts.train import initialize


def main():
    original=read_json(ROOT/'manifests/frames.json')['frames']
    held=[f for f in original if f['split']=='heldout']
    excluded_times=np.array([f['timestamp'] for f in held])
    excluded_rgb={f['rgb_sha256'] for f in held}
    rgb=np.loadtxt(DATA/'rgb.txt',dtype=str);depth=np.loadtxt(DATA/'depth.txt',dtype=str);gt=np.loadtxt(DATA/'groundtruth.txt')
    rgb_times=rgb[:,0].astype(float);depth_times=depth[:,0].astype(float)
    rotations=Slerp(gt[:,0],Rotation.from_quat(gt[:,4:8]))
    frames=[]
    for i,t in enumerate(rgb_times):
        j=int(np.argmin(abs(depth_times-t)))
        if abs(depth_times[j]-t)>.02 or not(gt[0,0]<=t<=gt[-1,0]) or np.min(abs(excluded_times-t))<=.10:continue
        digest=sha256(DATA/rgb[i,1])
        if digest in excluded_rgb:raise ValueError('Evaluation image entered training')
        pose=np.eye(4);pose[:3,:3]=rotations(t).as_matrix();pose[:3,3]=[np.interp(t,gt[:,0],gt[:,axis]) for axis in (1,2,3)]
        frames.append({'id':1000+i,'rgb':str(rgb[i,1]),'depth':str(depth[j,1]),'timestamp':float(t),
                       'depth_timestamp':float(depth_times[j]),'split':'train','c2w':pose.tolist(),
                       'rgb_sha256':digest,'depth_sha256':sha256(DATA/depth[j,1])})
    output=ART/'brush/dataset_dense';(output/'images').mkdir(parents=True,exist_ok=True)
    basis=np.diag([1.,-1.,-1.,1.])
    for split,items in [('train',frames),('val',[original[i] for i in [7,27,47]])]:
        rows=[]
        for f in items:
            name=f'frame_{f["id"]:02}.png';path=output/'images'/name
            image=Image.open(DATA/f['rgb']).resize((320,240),Image.Resampling.BOX).convert('RGBA');image.putalpha(255);image.save(path)
            rows.append({'file_path':'images/'+name,'transform_matrix':(np.array(f['c2w'])@basis).tolist()})
        scene={'w':320,'h':240,'fl_x':262.5,'fl_y':262.5,'cx':160.,'cy':120.,'frames':rows}
        if split=='train':scene['ply_file_path']='initial.ply'
        atomic_json(output/f'transforms_{split}.json',scene)
    xyz,color=initialize(frames,.01,80000)
    values=np.concatenate([xyz,(color-.5)/.28209479177387814],axis=1).astype('<f4')
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n'
    header+=''.join('property float '+name+'\n' for name in ['x','y','z','f_dc_0','f_dc_1','f_dc_2'])+'end_header\n'
    with (output/'initial.ply').open('wb') as handle:handle.write(header.encode());handle.write(values.tobytes())
    base=read_json(ROOT/'reports/brush/preparation.json')
    base.update(train_ids=[f['id'] for f in frames],train_views=len(frames),temporal_exclusion_seconds=.10,
                original_frame_manifest_sha256=sha256(ROOT/'manifests/frames.json'),source_sha256=sha256(Path(__file__)),
                files={str(p.relative_to(ROOT)):sha256(p) for p in output.rglob('*') if p.is_file()},
                minimum_train_to_any_heldout_seconds=float(np.min(abs(np.array([f['timestamp'] for f in frames])[:,None]-excluded_times[None,:]))),
                training_frames=frames,opacity_target='All training surfaces alpha=1; no held-out RGB or depth in initialization or fitting.')
    atomic_json(ROOT/'reports/brush/preparation_dense.json',base)
    print('Dense training views:',len(frames),'minimum held-out time gap:',base['minimum_train_to_any_heldout_seconds'],flush=True)


if __name__=='__main__':main()
