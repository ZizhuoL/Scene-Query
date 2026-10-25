"""Versioned log-density conditioning for thin scene surfaces."""
import torch
import torch.nn.functional as F
from .radiance import RadianceField, sample_planes


class LogDensityField(RadianceField):
    def forward(self, xyz, direction):
        q = (xyz-self.bounds[0])/(self.bounds[1]-self.bounds[0])*2-1
        grid = torch.stack([q[:, [0, 1]], q[:, [0, 2]], q[:, [1, 2]]])
        values = [q]+[sample_planes(p, grid) for p in self.planes]
        hidden = self.trunk(torch.cat(values, -1))
        # Density is inverse meters. Log-space reaches opaque centimeter-scale
        # surfaces without requiring a decoder output in the hundreds.
        sigma = torch.exp((self.density(hidden)[:, 0]-1).clamp(-10, 10))
        sigma = sigma*(q.abs().amax(-1) <= 1)
        d = F.normalize(direction, dim=-1)
        rgb = self.color(torch.cat([hidden, d, d.square(), d.roll(1, -1)*d], -1))
        return sigma, rgb


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = state['identity']['config']
    activation = config.get('density_activation', 'softplus')
    if activation not in ('exp', 'softplus'):
        raise ValueError('Unknown checkpoint density activation')
    cls = LogDensityField if activation == 'exp' else RadianceField
    model = cls(state['model']['bounds'], config['resolutions'], config['channels'], config['hidden'])
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state
