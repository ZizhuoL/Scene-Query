"""Associate and pin frames; interpolate mocap poses; build a deterministic split."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
from scenequery.common import ROOT, DATA, ART, sha256, atomic_json
from scenequery.geometry import K, unproject, project


def main():
    rgb = np.loadtxt(DATA / 'rgb.txt', dtype=str)
    dep = np.loadtxt(DATA / 'depth.txt', dtype=str)
    gt = np.loadtxt(DATA / 'groundtruth.txt')
    tr, td = rgb[:, 0].astype(float), dep[:, 0].astype(float)
    rotations = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8]))
    pairs = []
    for i, t in enumerate(tr):
        j = int(np.argmin(abs(td-t)))
        if abs(td[j]-t) <= .02 and gt[0, 0] <= t <= gt[-1, 0]:
            pairs.append((i, j))
    selected = np.linspace(0, len(pairs)-1, 60).round().astype(int)
    frames = []
    for idx, pos in enumerate(selected):
        i, j = pairs[pos]; t=tr[i]
        c2w = np.eye(4)
        c2w[:3,:3] = rotations(t).as_matrix()
        c2w[:3,3] = [np.interp(t,gt[:,0],gt[:,a]) for a in (1,2,3)]
        frames.append({'id':idx, 'rgb':str(rgb[i,1]), 'depth':str(dep[j,1]),
            'timestamp':float(t),'depth_timestamp':float(td[j]),'rgb_index':int(i),
            'split':'heldout' if idx%5==2 else 'train', 'c2w':c2w.tolist(),
            'rgb_sha256':sha256(DATA/rgb[i,1]),'depth_sha256':sha256(DATA/dep[j,1])})
    meta={'dataset_manifest_sha256':sha256(ROOT/'manifests/dataset.json'),
        'intrinsics':K.tolist(),'image_size':[640,480], 'depth_divisor':5000,
        'pose_convention':'camera-to-world; x right, y down, z forward; world in meters',
        'distortion':'ROS default pinhole without additional undistortion, per TUM recommendation',
        'pose_interpolation':'linear translation and quaternion SLERP at RGB timestamp',
        'frames':frames}
    atomic_json(ROOT/'manifests/frames.json',meta)
    uv=np.array([[20,30],[320,240],[600,450]],dtype=float)
    xyz=unproject(uv,np.array([1.,2.,3.]),np.array(frames[0]['c2w']))
    back,z=project(xyz,np.array(frames[0]['c2w']))
    atomic_json(ROOT/'reports/data_validation.json',{
        'total_rgb':len(rgb),'total_depth':len(dep),'associated':len(pairs),
        'selected':len(frames),'train':48,'heldout':12,
        'max_rgb_depth_offset_ms':max(abs(f['timestamp']-f['depth_timestamp'])*1000 for f in frames),
        'projection_roundtrip_max_px':float(abs(uv-back).max()),
        'max_rotation_orthogonality_error':max(float(abs(np.array(f['c2w'])[:3,:3].T@np.array(f['c2w'])[:3,:3]-np.eye(3)).max()) for f in frames),
        'semantic_labels':'No object annotations supplied by this TUM release. Project-specific approximate evaluation boxes are not independent human ground truth and are excluded from runtime.'})
    ART.mkdir(exist_ok=True)
    for split in ('train','heldout'):
        fs=[f for f in frames if f['split']==split]
        sheet=Image.new('RGB',(1280,260*((len(fs)+3)//4)))
        draw=ImageDraw.Draw(sheet)
        for j,f in enumerate(fs):
            im=Image.open(DATA/f['rgb']);im.thumbnail((320,240))
            x=(j%4)*320;y=(j//4)*260;sheet.paste(im,(x,y))
            draw.text((x+5,y+241),f"Frame {f['id']:02} / {f['rgb']}",fill='white')
        sheet.save(ART/f'{split}_contact.jpg')
    print('Prepared 48 training and 12 held-out frames.')


if __name__=='__main__':
    main()
