"""TUM camera-to-world geometry in meters, with OpenCV optical axes."""
import numpy as np

K = np.array([[525., 0., 319.5], [0., 525., 239.5], [0., 0., 1.]], dtype=np.float32)


def unproject(uv, depth, c2w, intrinsics=K):
    xy = (np.asarray(uv) - intrinsics[:2, 2]) / np.diag(intrinsics)[:2]
    camera = np.c_[xy, np.ones(len(xy))] * np.asarray(depth)[:, None]
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def project(xyz, c2w, intrinsics=K):
    camera = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
    z = camera[:, 2]
    uv = camera[:, :2] / np.maximum(z[:, None], 1e-8) * np.diag(intrinsics)[:2] + intrinsics[:2, 2]
    return uv, z


def visibility(xyz, c2w, depth, tolerance=0.05):
    uv, z = project(xyz, c2w)
    pix = np.rint(np.clip(uv, -1e7, 1e7)).astype(np.int32)
    inside = (z > 0.2) & (pix[:, 0] >= 0) & (pix[:, 0] < depth.shape[1]) & (pix[:, 1] >= 0) & (pix[:, 1] < depth.shape[0])
    valid = np.flatnonzero(inside)
    observed = depth[pix[valid, 1], pix[valid, 0]]
    inside[valid] &= (observed > 0) & (np.abs(z[valid] - observed) < tolerance)
    return uv, inside
