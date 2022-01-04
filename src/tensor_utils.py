import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def vector_to_pitchyaw(a):
    if a.shape[1] == 2:
        return a
    elif a.shape[1] == 3:
        a = a.view(-1, 3)
        norm_a = torch.div(a, torch.norm(a, dim=1).view(-1, 1) + 1e-7)
        return torch.stack([
            torch.asin(norm_a[:, 1]),
            torch.atan2(norm_a[:, 0], norm_a[:, 2]),
        ], dim=1)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def pitchyaw_to_rotation(a):
    if a.shape[1] == 3:
        a = vector_to_pitchyaw(a)

    cos = torch.cos(a)
    sin = torch.sin(a)
    ones = torch.ones_like(cos[:, 0])
    zeros = torch.zeros_like(cos[:, 0])
    matrices_1 = torch.stack([ones, zeros, zeros,
                              zeros, cos[:, 0], sin[:, 0],
                              zeros, -sin[:, 0], cos[:, 0]
                              ], dim=1)
    matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                              zeros, ones, zeros,
                              -sin[:, 1], zeros, cos[:, 1]
                              ], dim=1)
    matrices_1 = matrices_1.view(-1, 3, 3)
    matrices_2 = matrices_2.view(-1, 3, 3)
    matrices = torch.matmul(matrices_2, matrices_1)
    return matrices


def rotation_to_vector(a):
    assert(a.ndim == 3)
    assert(a.shape[1] == a.shape[2] == 3)
    frontal_vector = torch.cat([
        torch.zeros_like(a[:, :2, 0]).reshape(-1, 2, 1),
        torch.ones_like(a[:, 2, 0]).reshape(-1, 1, 1),
    ], axis=1)
    return torch.matmul(a, frontal_vector)


def apply_transformation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    return torch.matmul(T, h_vec)[:, :3, 0]


def apply_rotation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    R = T[:, :3, :3]
    return torch.matmul(R, vec).reshape(-1, 3)

nn_plane_normal = None
nn_plane_other = None


def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    global nn_plane_normal, nn_plane_other
    if nn_plane_normal is None:
        nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=g.device).view(1, 3, 1)
        nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=g.device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))[:, :2, 0]

def to_screen_coordinates(origin, direction, rotation, monitor):
    direction = pitchyaw_to_vector(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 1, 2)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    # Transform values
    inv_camera_transformation = torch.Tensor(monitor.inv_camera_transformation).to(origin.device)
    inv_camera_transformation = inv_camera_transformation.repeat(origin.shape[0],1,1)
    direction = apply_rotation(inv_camera_transformation, direction)
    origin = apply_transformation(inv_camera_transformation, origin)

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero(origin, direction)

    PoG_mm = recovered_target_2D

    # Convert back from mm to pixels
    ppm_w = monitor.pixels_per_millimeter[0]
    ppm_h = monitor.pixels_per_millimeter[1]
    PoG_px = torch.stack([
        torch.clamp(recovered_target_2D[:, 0] * ppm_w,
                    0.0, 1920),
        torch.clamp(recovered_target_2D[:, 1] * ppm_h,
                    0.0, 1080)
    ], axis=-1)

    return PoG_mm, PoG_px