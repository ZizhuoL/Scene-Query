"""Export only training observations and fixed development cameras for Brush."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scenequery.common import ROOT, DATA, ART, read_json, atomic_json, sha256
from scripts.train import initialize


def main():
    protocol = read_json(ROOT/'config/neural/protocol.json')
    frames = read_json(ROOT/'manifests/frames.json')['frames']
    output = ART/'brush/dataset'
    output.mkdir(parents=True, exist_ok=True)
    (output/'images').mkdir(exist_ok=True)
    # Nerfstudio uses OpenGL camera axes. Brush flips Y/Z back on import.
    basis = np.diag([1., -1., -1., 1.])
    exported = {}
    for split, ids in [('train', protocol['train_ids']), ('val', protocol['development_ids'])]:
        rows = []
        for index in ids:
            frame = frames[index]
            path = output/'images'/f'frame_{index:02}.png'
            Image.open(DATA/frame['rgb']).resize((320, 240), Image.Resampling.BOX).save(path)
            rows.append({'file_path': f'images/{path.name}',
                         'transform_matrix': (np.array(frame['c2w']) @ basis).tolist()})
            exported[str(path.relative_to(ROOT))] = sha256(path)
        # Brush samples centers at pixel index + .5; original K uses integer centers.
        scene = {'w': 320, 'h': 240, 'fl_x': 262.5, 'fl_y': 262.5,
                 'cx': 160., 'cy': 120., 'frames': rows}
        if split == 'train':
            scene['ply_file_path'] = 'initial.ply'
        atomic_json(output/f'transforms_{split}.json', scene)
    xyz, rgb = initialize([frames[i] for i in protocol['train_ids']], .01, 80000)
    # SH DC coefficients encode measured training colors; no pretrained scene weights.
    values = np.concatenate([xyz, (rgb-.5)/.28209479177387814], axis=1).astype('<f4')
    names = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\n'
    header += ''.join('property float '+name+'\n' for name in names)+'end_header\n'
    with (output/'initial.ply').open('wb') as handle:
        handle.write(header.encode('ascii')); handle.write(values.tobytes())
    for path in [output/'initial.ply', output/'transforms_train.json', output/'transforms_val.json']:
        exported[str(path.relative_to(ROOT))] = sha256(path)
    report = {'upstream': {'repository': 'https://github.com/ArthurBrussee/brush',
                          'release': 'v0.3.0', 'commit': '3edecbb2fe79d3e2c87eeab85b15e0b1dd10d486',
                          'archive_sha256': '65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c'},
              'train_ids': protocol['train_ids'], 'development_ids': protocol['development_ids'],
              'initial_points': len(xyz), 'voxel_m': .01, 'seed': 42,
              'initialization': 'Unoptimized measured training RGB-D points, not the prior fitted surfels.',
              'camera_conversion': 'c2w OpenCV @ diag(1,-1,-1,1); unchanged metric world. Brush pixel-center convention adds .5 before scaling principal point.',
              'source_sha256': sha256(Path(__file__)), 'frames_sha256': sha256(ROOT/'manifests/frames.json'),
              'files': exported, 'final_views_exported': False}
    atomic_json(ROOT/'reports/brush/preparation.json', report)
    print('Prepared', len(xyz), 'training-only initialization points and 48/3 train/development views.')


if __name__ == '__main__':
    main()
