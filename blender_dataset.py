import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import time

from ray_utils import *
from einops import rearrange,repeat
import torch.nn.functional as F

class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.transforms = T.ToTensor()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        self.w, self.h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800
        print(self.focal)
        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0

        self.directions = get_ray_directions(self.h, self.w, self.focal) # (h, w, 3)
            
        if self.split == 'train':
            self.image_paths = []
            self.poses = []
            self.all_ray_o = []
            self.all_ray_d = []
            self.all_rgbs = []
            self.viewdirs = []
            print('get_training_rays: start')
            eps_time = time.time()
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transforms(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

                #rays_o, rays_d = get_view_rays(800, 800, pose_c2w = c2w, focal = self.focal, pixel_mode='center')
                #rays_o = rays_o.reshape(-1,3)
                #rays_d = rays_d.reshape(-1,3)

                viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
                
                self.all_ray_o += [rays_o]
                self.all_ray_d += [rays_d]
                self.viewdirs += [viewdirs]

            self.all_ray_o = torch.cat(self.all_ray_o, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_ray_d = torch.cat(self.all_ray_d, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.viewdirs = torch.cat(self.viewdirs, 0)  # (len(self.meta['frames])*h*w, 3)
            print('get_training_rays: finish (eps time:', eps_time, 'sec)')
  
    def get_bound(self):
        return self.near, self.far
        
    def define_bbox(self):
        print('Compute_bbox: start')
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        pts_nf = torch.stack([self.all_ray_o+self.viewdirs*self.near, self.all_ray_o+self.viewdirs*self.far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1)))
        print('Compute_bbox: xyz_min', xyz_min)
        print('Compute_bbox: xyz_max', xyz_max)
        print('Compute_bbox: finish')
        return xyz_max, xyz_min

    def voxel_count_views(self, channel, Nx, Ny, Nz, xyz_max, xyz_min, voxel_size):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        #return  torch.ones([1, channel, Nx, Ny, Nz])
        count = torch.zeros([1, channel, Nx, Ny, Nz]).to('cuda')
        samples_per_ray = int(np.linalg.norm(np.array([Nx, Ny, Nz])+1) / 0.5) + 1
        z_vals = torch.arange(samples_per_ray).to('cuda')*voxel_size*0.5
        chunk_size = 10000
 
        rays_o = self.all_ray_o.to('cuda').split(chunk_size)
        rays_d = self.all_ray_d.to('cuda').split(chunk_size)
        i = 0
        ones = torch.ones([1, channel, Nx, Ny, Nz]).to('cuda').requires_grad_()
        for ray_os, ray_ds in zip(rays_o, rays_d):

            print(i)
            ray_d_norm = ray_ds.norm(dim=-1, keepdim=True)
            #viewdirs = ray_ds/ray_d_norm

            t_1 = ((xyz_max.to(ray_os.device) - ray_os) / (ray_ds + 1e-6)) #(num_rays,3)
            t_2 = ((xyz_min.to(ray_os.device) - ray_os) / (ray_ds + 1e-6)) #(num_rays,3)

            t_min = torch.minimum(t_1, t_2).amax(-1).clamp(min=self.near, max=self.far)
            t_max = torch.maximum(t_1, t_2).amin(-1).clamp(min=self.near, max=self.far)
                     
            delta = torch.einsum('ij,ik->ikj', [ray_ds/ray_d_norm, t_min[...,None]*ray_d_norm  + z_vals[None,:]])
            rays_pts = ray_os[...,None, :] + delta
            '''
            ray_start = ray_os + t_min[...,None] * ray_ds
            rays_pts = ray_start[...,
                      None, :] + z_vals[..., :, None] * viewdirs[..., None, :]
            '''
            xyz_min_ = xyz_min.to(rays_pts.device).reshape(1, 1, -1)
            xyz_max_ = xyz_max.to(rays_pts.device).reshape(1, 1, -1)
            rays_pts = ((rays_pts - xyz_min_)/(xyz_max_ - xyz_min_))*2 - 1
            rays_pts = rays_pts.reshape(1, 1, 1, -1, 3)
            output = F.grid_sample(ones, rays_pts, mode='bilinear', align_corners=True)
            output = output.sum().backward()
            i = i + 1

            with torch.no_grad():
                count += (ones.grad > 1)

        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

        
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rgbs)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': 
            sample = {  'ray_o': self.all_ray_o[idx],
                    'ray_d': self.all_ray_d[idx],
                    'viewdirs' : self.viewdirs[idx],
                    'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transforms(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            '''
            rays_o, rays_d = get_view_rays(800, 800, pose_c2w = c2w, focal = self.focal, pixel_mode='center')
            rays_o = rays_o.reshape(-1,3)
            rays_d = rays_d.reshape(-1,3)
            '''

            viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)

            
            sample = {'ray_o': rays_o,
                      'ray_d': rays_d,
                      'viewdirs' : viewdirs,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample

if __name__ == "__main__":
    dataset = BlenderDataset(
            root_dir = 'data/nerf_synthetic/lego', split='val', img_wh=(800, 800)
        )
    
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            shuffle=False,
            num_workers=1,
            batch_size = 1,
            pin_memory=True
        )
    
    for data in dataloader:
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        #print(rays_o.shape)
        #print(rays_d.shape)
        break


