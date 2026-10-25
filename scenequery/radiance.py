"""Compact scene-specific radiance field; no pretrained reconstruction weights."""
import torch
from torch import nn
import torch.nn.functional as F


def sample_planes(plane, grid):
    """Bilinear plane lookup using gathers supported by pinned PyTorch MPS.

    Equivalent to grid_sample(..., align_corners=True, padding_mode='border').
    PyTorch 2.6 MPS does not implement grid_sampler_2d_backward.
    """
    resolution = plane.shape[-1]
    xy = (grid.clamp(-1, 1)+1)*((resolution-1)/2)
    lo = xy.floor().long()
    hi = (lo+1).clamp(max=resolution-1)
    fraction = xy-lo
    offset = torch.arange(3, device=plane.device)[:, None]*resolution*resolution
    flat = plane.permute(0, 2, 3, 1).reshape(-1, plane.shape[1])
    def get(x, y):
        return flat[(offset+y*resolution+x).reshape(-1)].reshape(3, -1, plane.shape[1])
    a = get(lo[..., 0], lo[..., 1])
    b = get(hi[..., 0], lo[..., 1])
    c = get(lo[..., 0], hi[..., 1])
    d = get(hi[..., 0], hi[..., 1])
    fx, fy = fraction[..., :1], fraction[..., 1:]
    return ((a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy).permute(1, 0, 2).reshape(grid.shape[1], -1)


class RadianceField(nn.Module):
    def __init__(self, bounds, resolutions=(32, 64, 128), channels=8, hidden=64):
        super().__init__()
        self.register_buffer('bounds', torch.as_tensor(bounds, dtype=torch.float32))
        self.planes = nn.ParameterList([
            nn.Parameter(torch.randn(3, channels, r, r) * .05) for r in resolutions])
        features = 3 * channels * len(resolutions) + 3
        self.trunk = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.density = nn.Linear(hidden, 1)
        self.color = nn.Sequential(nn.Linear(hidden + 9, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, xyz, direction):
        q = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) * 2 - 1
        grid = torch.stack([q[:, [0, 1]], q[:, [0, 2]], q[:, [1, 2]]])
        values = [q]
        for plane in self.planes:
            values.append(sample_planes(plane, grid))
        hidden = self.trunk(torch.cat(values, -1))
        sigma = F.softplus(self.density(hidden)[:, 0] - 1)
        sigma = sigma * (q.abs().amax(-1) <= 1)
        d = F.normalize(direction, dim=-1)
        rgb = self.color(torch.cat([hidden, d, d.square(), d.roll(1, -1)*d], -1))
        return sigma, rgb

    def tv_loss(self):
        return sum((p[:, :, 1:] - p[:, :, :-1]).square().mean() +
                   (p[:, :, :, 1:] - p[:, :, :, :-1]).square().mean()
                   for p in self.planes)


def ray_bounds(origins, directions, bounds, near=.25, far=3.5):
    safe = torch.where(directions.abs() < 1e-7, torch.full_like(directions, 1e-7), directions)
    a = (bounds[0] - origins) / safe
    b = (bounds[1] - origins) / safe
    lo = torch.minimum(a, b).amax(-1).clamp(min=near, max=far-1e-3)
    hi = torch.maximum(a, b).amin(-1).clamp(min=near+1e-3, max=far)
    hi = torch.maximum(hi, lo + 1e-3)
    return lo, hi


def composite(sigma, colors, z, directions, far):
    delta = torch.cat([z[:, 1:]-z[:, :-1], far[:, None]-z[:, -1:]], -1).clamp_min(1e-6)
    optical = sigma * delta * directions.norm(dim=-1, keepdim=True)
    transmittance = torch.exp(-torch.cat([torch.zeros_like(optical[:, :1]),
                                         optical[:, :-1].cumsum(-1)], -1))
    weights = -torch.expm1(-optical) * transmittance
    opacity = weights.sum(-1)
    rgb = (weights[..., None] * colors).sum(-2)
    depth = (weights*z).sum(-1) + (1-opacity)*far
    return {'rgb': rgb, 'depth': depth, 'opacity': opacity, 'weights': weights, 'z': z}


def render_rays(model, origins, directions, samples=96, jitter=None):
    """Camera-z samples; inference needs only rays and learned parameters."""
    lo, hi = ray_bounds(origins, directions, model.bounds)
    fraction = (torch.arange(samples, device=origins.device, dtype=origins.dtype)[None] +
                (.5 if jitter is None else jitter)) / samples
    z = lo[:, None] + (hi-lo)[:, None]*fraction
    xyz = origins[:, None] + directions[:, None]*z[..., None]
    dirs = directions[:, None].expand_as(xyz)
    sigma, color = model(xyz.reshape(-1, 3), dirs.reshape(-1, 3))
    return composite(sigma.reshape(-1, samples), color.reshape(-1, samples, 3),
                     z, directions, hi)


def camera_rays(c2w, width=320, height=240, device='cpu'):
    pose = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    # Preserve pixel centers when scaling the original 640 x 480 pinhole image.
    u = (x.float()+.5)*640/width-.5
    v = (y.float()+.5)*480/height-.5
    camera = torch.stack([(u-319.5)/525, (v-239.5)/525, torch.ones_like(u)], -1)
    direction = camera.reshape(-1, 3) @ pose[:3, :3].T
    return pose[:3, 3].expand_as(direction), direction


@torch.no_grad()
def render_image(model, c2w, width=320, height=240, samples=128, chunk=1024):
    origins, directions = camera_rays(c2w, width, height, model.bounds.device)
    values = {k: [] for k in ('rgb', 'depth', 'opacity')}
    for start in range(0, len(origins), chunk):
        result = render_rays(model, origins[start:start+chunk], directions[start:start+chunk], samples)
        for k in values:
            values[k].append(result[k].cpu())
    return {k: torch.cat(v).reshape(height, width, -1).squeeze(-1).numpy()
            for k, v in values.items()}
