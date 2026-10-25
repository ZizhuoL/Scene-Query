"""Reference Gaussian projection/depth inference, validated against native Brush.

Uses the standard covariance projection, SH basis and front-to-back compositing.
Brush v0.3.0 (Apache-2.0) supplies the native rendering contract; no pretrained
weights or observations enter this renderer. This module is not used for fitting.
"""
import ctypes
import subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation
from scenequery.common import ROOT, ART, sha256


def read_ply(path):
    with Path(path).open('rb') as handle:
        names = []; count = None
        if handle.readline().strip() != b'ply': raise ValueError('Expected PLY')
        if handle.readline().strip() != b'format binary_little_endian 1.0':
            raise ValueError('Expected binary little-endian PLY')
        while True:
            line = handle.readline().decode('ascii').strip()
            if not line: raise ValueError('Truncated header')
            if line == 'end_header': break
            words = line.split()
            if words[:2] == ['element', 'vertex']: count = int(words[2])
            elif words[0] == 'element': raise ValueError('Unexpected element')
            elif words[0] == 'property':
                if words[1] != 'float': raise ValueError('Expected float property')
                names.append(words[2])
        values = np.frombuffer(handle.read(), dtype='<f4')
        if count is None or len(values) != count*len(names): raise ValueError('PLY length mismatch')
        values = values.reshape(count, len(names))
        if not np.isfinite(values).all(): raise ValueError('Nonfinite Gaussian parameter')
    return {name: values[:, i] for i, name in enumerate(names)}


def components(data):
    columns = lambda prefix, count: np.stack([data[f'{prefix}{i}'] for i in range(count)], -1)
    xyz = np.stack([data[k] for k in ('x', 'y', 'z')], -1)
    quaternion = columns('rot_', 4)
    rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix().astype('float32')
    scaled = rotation * np.exp(columns('scale_', 3))[:, None, :]
    covariance = scaled @ scaled.transpose(0, 2, 1)
    dc = columns('f_dc_', 3)[:, None, :]
    rest_count = len([k for k in data if k.startswith('f_rest_')])
    rest = columns('f_rest_', rest_count).reshape(len(xyz), 3, -1).transpose(0, 2, 1) if rest_count else np.empty((len(xyz), 0, 3))
    return xyz, covariance, expit(data['opacity']), np.concatenate([dc, rest], 1)


def project_gaussians(data, c2w, width=320, height=240):
    xyz, covariance, opacity, coefficients = components(data)
    pose = np.asarray(c2w, dtype='float32')
    camera = (xyz-pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    visible = (z > .01) & (opacity >= 1/255)
    xyz, covariance, opacity, coefficients, camera, z = [v[visible] for v in (xyz, covariance, opacity, coefficients, camera, z)]
    focal = np.array([525*width/640, 525*height/480], dtype='float32')
    center = np.array([width/2, height/2], dtype='float32')
    uv = camera[:, :2]/z[:, None]
    clipped = np.clip(uv, (-.15*np.array([width, height])-center)/focal,
                      (1.15*np.array([width, height])-center)/focal)
    jacobian = np.zeros((len(z), 2, 3), dtype='float32')
    jacobian[:, 0, 0] = focal[0]/z; jacobian[:, 1, 1] = focal[1]/z
    jacobian[:, :, 2] = -focal[None, :]/z[:, None]*clipped
    view_rotation = pose[:3, :3].T
    covariance_camera = view_rotation @ covariance @ view_rotation.T
    covariance_image = jacobian @ covariance_camera @ jacobian.transpose(0, 2, 1)
    covariance_image[:, 0, 0] += .3; covariance_image[:, 1, 1] += .3
    inverse = np.linalg.inv(covariance_image)
    mean = uv*focal+center
    extent = np.sqrt(2*np.log(255*opacity[:, None])*np.diagonal(covariance_image, axis1=1, axis2=2))
    direction = xyz-pose[:3, 3]; direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    x, y, d = direction.T
    basis = [np.full_like(x, .28209479177387814)]
    if coefficients.shape[1] >= 4:
        basis += [-.4886025119029199*y, .4886025119029199*d, -.4886025119029199*x]
    if coefficients.shape[1] >= 9:
        basis += [1.0925484305920792*x*y, -1.0925484305920792*y*d,
                  .31539156525252005*(3*d*d-1), -1.0925484305920792*x*d,
                  .5462742152960396*(x*x-y*y)]
    if coefficients.shape[1] not in (1, 4, 9): raise ValueError('Only SH degree 0–2 supported')
    colors = .5 + (np.stack(basis, -1)[:, :, None]*coefficients).sum(1)
    rows = np.column_stack([mean, inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1], colors, opacity, z, extent])
    inside = (mean[:, 0]+extent[:, 0] > 0) & (mean[:, 0]-extent[:, 0] < width) & (mean[:, 1]+extent[:, 1] > 0) & (mean[:, 1]-extent[:, 1] < height)
    rows = rows[inside]
    return np.ascontiguousarray(rows[np.argsort(rows[:, 9], kind='stable')], dtype='float32')


def render_gaussians(data, c2w, width=320, height=240):
    source = ROOT/'scenequery/gaussian_blend.cpp'
    directory = ART/'tools'; directory.mkdir(parents=True, exist_ok=True)
    library = directory/f'gaussian_{sha256(source)[:16]}.dylib'
    if not library.exists():
        temporary = library.with_suffix('.tmp.dylib')
        subprocess.run(['/usr/bin/clang++', '-std=c++17', '-O3', '-dynamiclib', str(source), '-o', str(temporary)], check=True, timeout=60)
        temporary.replace(library)
    function = ctypes.CDLL(str(library)).blend_gaussians
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, pointer]
    function.restype = None
    projected = project_gaussians(data, c2w, width, height)
    output = np.zeros((height, width, 5), dtype='float32')
    function(projected.ctypes.data_as(pointer), len(projected), width, height, output.ctypes.data_as(pointer))
    return {'rgb': output[:, :, :3].clip(0, 1), 'depth': output[:, :, 3], 'opacity': output[:, :, 4]}
