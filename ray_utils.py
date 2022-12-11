import numpy as np
import torch
from einops import rearrange,repeat
from kornia import create_meshgrid


@torch.no_grad()
def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W),torch.linspace(0, H-1, H), indexing='xy')

    i = i + 0.5
    j = j + 0.5

    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

@torch.no_grad()
def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = torch.sum(directions[..., None, :] * c2w[:3,:3], -1)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

@torch.no_grad()
def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf
    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18
    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate
    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d


@torch.no_grad()
def get_view_rays(height, width, pose_c2w, focal, pixel_mode='center',intrinsic=None):
    """
    height: scalar
    width: scalar
    intrinsic: (3,3) or (4, 4)
    pose_c2w: (4, 4)
    pixel_mode: string
    """
    device = pose_c2w.device
    # (u, v) from screen coordinate system
    us, vs = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=device),
        torch.linspace(0, height - 1, height, device=device))

    us = us.t().float()
    vs = vs.t().float()

    if pixel_mode == 'lefttop':
        pass
    elif pixel_mode == 'center':
        us = us + 0.5
        vs = vs + 0.5
    elif pixel_mode == 'random':
        us = us + torch.rand_like(us)
        vs = vs + torch.rand_like(vs)
    else:
        raise NotImplementedError(f'Not support pixel mode: {pixel_mode}')

    # Intrinsic params: focal length and center shift
    if intrinsic is None:
        fx = focal
        fy = focal
        cx = width/2
        cy = height/2
    else:
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

    # (H, W, 3)
    # Points on the image plane in camera frame
    dirs_c = torch.stack(
        [(us - cx) / fx,
         -(vs - cy) / fy, -torch.ones_like(us)], dim=-1)

    # Convert to world frame (rotation)
    rays_d = torch.sum(dirs_c[..., None, :] * pose_c2w[:3, :3], dim=-1)

    # Compute ray origin and ray direction
    rays_o = pose_c2w[:3, 3].expand(rays_d.shape)

    return rays_o.contiguous(), rays_d.contiguous()

@torch.no_grad()
def sample_pts_along_rays(rays_o, rays_d, near, far, samples_per_ray=200):
    """
    rays_o: (num_rays, 3)
    rays_d: (num_rays, 3)
    near: scalar
    far: scalar
    samples_per_ray: scalar
    """
    z_vals = near + (far - near) * torch.linspace(
        0., 1., steps=samples_per_ray, device=rays_o.device)
    z_vals = z_vals.unsqueeze(0).expand(rays_o.shape[0], -1)
    rays_pts = rays_o[...,
                      None, :] + z_vals[..., :, None] * rays_d[..., None, :]
    return rays_pts, z_vals


@torch.no_grad()
def sample_pts_by_voxel_size(rays_o, rays_d, near, far, xyz_max, xyz_min, step_size, samples_per_ray):
    """
    rays_o: (num_rays, 3)
    rays_d: (num_rays, 3)
    near: scalar
    far: scalar
    samples_per_ray: scalar
    """
    step_size = step_size.to(rays_o.device)
    ray_d_norm = rays_d.norm(dim=-1, keepdim=True)
    viewdirs = rays_d/ray_d_norm
    t_1 = ((xyz_max.to(rays_o.device) - rays_o) / (rays_d + 1e-6)) #(num_rays,3)
    t_2 = ((xyz_min.to(rays_o.device) - rays_o) / (rays_d + 1e-6)) #(num_rays,3)
    t_min = torch.minimum(t_1, t_2).amax(-1).clamp(min=near, max=far)
    t_max = torch.maximum(t_1, t_2).amin(-1).clamp(min=near, max=far)
    #mask = (t_max <= t_min)
    z_vals = (torch.arange(samples_per_ray)).to(rays_o.device)*step_size

    #z_vals = repeat(z_vals, 'n -> r n', r = rays_o.shape[0])
    delta = torch.einsum('ij,ik->ikj', [viewdirs, t_min[...,None]*ray_d_norm  + z_vals[None,:]])
    rays_pts = rays_o[...,None, :] + delta

    '''
    ray_start = rays_o + t_min[...,None] * rays_d
    rays_pts = ray_start[...,
                      None, :] + z_vals[..., :, None] * viewdirs[..., None, :]
    mask = mask[...,None] | ((xyz_min>rays_pts) | (rays_pts>xyz_max)).any(dim=-1)
    '''
    return rays_pts, z_vals


@torch.no_grad()
def compute_rays_pts_mask(rays_pts, xyz_max, xyz_min):
    """
    rays_pts: (num_rays, num_pts, 3)
    xyz_limit: (3,)
    """
    xyz_max = xyz_max.reshape(1, 1, -1).expand(rays_pts.shape[0],
                                                   rays_pts.shape[1], 3)

    xyz_min = xyz_min.reshape(1, 1, -1).expand(rays_pts.shape[0],
                                                   rays_pts.shape[1], 3)
    xyz_max = xyz_max.to(rays_pts.device)
    xyz_min = xyz_min.to(rays_pts.device)

    mask_high = torch.all(rays_pts < xyz_max, dim=-1)
    mask_low = torch.all(rays_pts > xyz_min, dim=-1)
    return torch.logical_and(mask_low, mask_high).to(rays_pts.device)


def compute_alpha(density, distances):
    return 1. - torch.exp((-density * distances))


def compute_weights(density, z_vals, rays_d):
    """
    density: (num_rays, num_pts)
    z_vals: (num_rays, num_pts)
    rays_d: (num_rays, 3)
    """
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat(
        [distances, 1e10 * torch.ones_like(distances[..., :1])], dim=-1)
    distances = distances * torch.norm(rays_d[..., None, :], dim=-1)
    alpha = compute_alpha(density, distances)
    weights = alpha * torch.cumprod(
        torch.cat([
            torch.ones(
                (alpha.shape[0], 1), device=density.device), 1. - alpha + 1e-10
        ], -1), -1)[:, :-1]
    return weights


def compute_map(rgb, density, z_vals, rays_d, mask, white_background=False):
    """
    rgb: (num_rays, num_pts, 3)
    density: (num_rays, num_pts, 1)
    z_vals: (num_rays, num_pts)
    rays_d: (num_rays, 3)
    mask: (num_rays, num_pts)
    """
    # (num_rays, num_pts)
    weights = compute_weights(density[..., 0] * mask.float(), z_vals, rays_d)
    # (num_rays, 3)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    # (num_rays, 1)
    depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)
    # (num_rays, 1)
    acc_map = torch.sum(weights, dim=-1, keepdim=True)
    if white_background:
        rgb_map = rgb_map + (1. - acc_map)
    return rgb_map, depth_map

def dvgo_compute_alpha(density, voxel_size_ratio):
    distances = 0.5 * voxel_size_ratio
    return 1. - torch.exp((-density * distances))

def dvgo_compute_weights(density, voxel_size_ratio):
    alpha = dvgo_compute_alpha(density, voxel_size_ratio).squeeze()
    weights = alpha * torch.cumprod(
        torch.cat([
            torch.ones(
                (alpha.shape[0], 1), device = density.device), 1. - alpha + 1e-10
        ], -1), -1)[:, :-1]
    
    return weights

def dvgo_compute_map(rgb, density, z_vals, voxel_size_ratio, mask, white_background=False):
    # (num_rays, num_pts)
    weights = dvgo_compute_weights(density[..., 0] * mask.float(), voxel_size_ratio)
    # (num_rays, 3)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    # (num_rays, 1)
    depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)
    # (num_rays, 1)
    acc_map = torch.sum(weights, dim=-1, keepdim=True)
    if white_background:
        rgb_map = rgb_map + (1. - acc_map)
    return rgb_map, depth_map

if __name__ == "__main__":
    intrinsic = torch.FloatTensor(
        [[1169.621094, 0.000000, 646.295044, 0.000000],
         [0.000000, 1167.105103, 489.927032, 0.000000],
         [0.000000, 0.000000, 1.000000, 0.000000],
         [0.000000, 0.000000, 0.000000, 1.000000]])

    pose_c2w = torch.FloatTensor([[-0.181626, -0.477725, 0.859530, 4.717887],
                                  [-0.982468, 0.125533, -0.137833, 4.241773],
                                  [-0.042053, -0.869494, -0.492149, 1.347169],
                                  [0.000000, 0.000000, 0.000000, 1.000000]])

    height = 800
    width = 800
    focal = 1111.1110311937682
    pixel_mode = 'center'
    directions = get_ray_directions(height, width, focal)
    rays_o, rays_d = get_rays(directions, pose_c2w)

    near = 2
    far = 5
    step_size = torch.tensor([0.5*0.0559])
    xyz_max = torch.tensor([3.0054, 3.0157, 2.3378])
    xyz_min = torch.tensor([-3.0165, -3.0083, -2.5941])
    samples_per_ray = 315
    rays_o = torch.tensor([[0.,1.,2.],[3.,3.,2.]])
    rays_d = torch.tensor([[0.,1.,2.],[3.,3.,2.]])
    rays_pts, z_vals = sample_pts_by_voxel_size(rays_o, rays_d, near, far, xyz_max, xyz_min, step_size, samples_per_ray)

    ind_norm = ((rays_pts - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1
    #print(ind_norm)

    xyz_min = xyz_min.reshape(1, 1, -1)
    xyz_max = xyz_max.reshape(1, 1, -1)
    query = ((rays_pts - xyz_min)/(xyz_max - xyz_min)).flip((-1,))*2 - 1


    #print(query)

    #rays_o, rays_d= get_view_rays(height, width, intrinsic, pose_c2w,pixel_mode)

    rgb = torch.ones([2,10,3])
    density = torch.ones([2,10,1])
    mask = torch.ones([2,10])
    z_vals = z_vals = (torch.arange(10))*step_size
    rgb_map, depth_map = dvgo_compute_map(rgb, density, z_vals, 1, mask, white_background=False)
    print(rgb_map)


    alpha = torch.tensor([[0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935,
         0.3935],
        [0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935, 0.3935,
         0.3935]])
    
    weights, alphainv_cum = get_ray_marching_ray(alpha)
    rgb_marched = (weights[...,None] * rgb).sum(-2)
    print(rgb_marched)