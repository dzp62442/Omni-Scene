"""Bounded copy of frozen evaluate.py PLY export; original utility stays unchanged."""

import numpy as np
import torch

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def save_ply(gaussians, path, crop_range=[-50.0, -50.0, -3.0, 50.0, 50.0, 12.0], compatible=True):
    # gaussians: [B, N, 14]
    # compatible: save pre-activated gaussians as in the original paper
    gaussians = torch.cat([gaussians[:, 0:3],
                           gaussians[:, 6:7],
                           gaussians[:, 11:14],
                           gaussians[:, 7:11],
                           gaussians[:, 3:6]], dim=-1)

    from plyfile import PlyData, PlyElement
    
    means3D = gaussians[:, 0:3].contiguous().float()
    opacity = gaussians[:, 3:4].contiguous().float()
    scales = gaussians[:, 4:7].contiguous().float()
    rotations = gaussians[:, 7:11].contiguous().float()
    shs = gaussians[:, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

    if crop_range is not None:
        x_start, y_start, z_start, x_end, y_end, z_end = crop_range
        mask = (means3D[:, 0] > x_start) & (means3D[:, 0] < x_end) & \
               (means3D[:, 1] > y_start) & (means3D[:, 1] < y_end) & \
               (means3D[:, 2] > z_start) & (means3D[:, 2] < z_end)
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

    # prune by opacity
    mask = opacity.squeeze(-1) >= 0.005
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(path)
