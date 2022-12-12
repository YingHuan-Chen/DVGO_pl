import torch
from torch import nn
from embedding import Embedding
from dvgo_field_coarse import DVGOCoarseField
from dvgo_field_fine import DVGOFineField
from soft_plus import SoftPlus
from ray_utils import*

class DVGO(nn.Module):
    def __init__(
        self,
        xyz_max,
        xyz_min,
        near,
        far,
        num_voxels_coarse = 1024000,
        num_voxels_fine = 160*160*160,
        num_sample = 64,
    ):
        super(DVGO, self).__init__()

        self.near = near
        self.far = far
        self.coarse_xyz_max = xyz_max
        self.coarse_xyz_min = xyz_min

        self.xyz_max = xyz_max
        self.xyz_min = xyz_min

        self.num_sample = num_sample

        #self.position_encoding = Embedding(3, 10)
        #self.direction_encoding = Embedding(3, 4)

        self.num_voxels_coarse = num_voxels_coarse
        self.num_voxels = self.num_voxels_coarse
        self.num_voxels_fine = num_voxels_fine

        self.L_coarse = self.coarse_xyz_max - self.coarse_xyz_min
        self.Lx_coarse, self.Ly_coarse, self.Lz_coarse = self.L_coarse[0], self.L_coarse[1], self.L_coarse[2]
        self.voxel_size_coarse = torch.pow(self.Lx_coarse*self.Ly_coarse*self.Lz_coarse/self.num_voxels_coarse,1/3)

        self.voxel_size = self.voxel_size_coarse
        self.voxel_size_ratio = self.voxel_size/self.voxel_size_coarse
        print('voxel_size: ', self.voxel_size)
        print('voxel_size_ratio: ', self.voxel_size_ratio)

        self.coarse_Nx = int(self.Lx_coarse/self.voxel_size_coarse) #torch.floor
        self.coarse_Ny = int(self.Ly_coarse/self.voxel_size_coarse) #torch.floor
        self.coarse_Nz = int(self.Lz_coarse/self.voxel_size_coarse) #torch.floor

        self.Nx = self.coarse_Nx
        self.Ny = self.coarse_Ny
        self.Nz = self.coarse_Nz

        print('coarse_Nx: ', self.coarse_Nx)
        print('coarse_Ny: ', self.coarse_Ny)
        print('coarse_Nz: ', self.coarse_Nz)
   
        self.coarse_field = DVGOCoarseField(
            coarse_Nx = self.coarse_Nx, 
            coarse_Ny = self.coarse_Ny, 
            coarse_Nz = self.coarse_Nz,
            xyz_max=self.coarse_xyz_max,
            xyz_min=self.coarse_xyz_min,
            density_grid_channel = 1,
            feature_grid_channel = 3,
            coarse_soft_plus = SoftPlus(voxel_size= self.voxel_size_coarse, alpha_init=1e-6),
        )

    def start_fine(self, fine_xyz_min , fine_xyz_max):
        
        self.xyz_max = fine_xyz_max
        self.xyz_min = fine_xyz_min

        self.L_fine = fine_xyz_max - fine_xyz_min
        self.Lx_fine, self.Ly_fine, self.Lz_fine = self.L_fine[0], self.L_fine[1], self.L_fine[2]
        self.voxel_size_fine = torch.pow(self.Lx_fine*self.Ly_fine*self.Lz_fine/self.num_voxels_fine,1/3)

        self.num_voxels = self.num_voxels_fine/8
        self.voxel_size = torch.pow(self.Lx_fine*self.Ly_fine*self.Lz_fine/self.num_voxels,1/3)
        self.voxel_size_ratio = self.voxel_size/self.voxel_size_fine
        print('voxel_size: ', self.voxel_size)
        print('voxel_size_base: ', self.voxel_size_fine)
        print('voxel_size_ratio: ', self.voxel_size_ratio)

        self.fine_Nx = int(self.Lx_fine/self.voxel_size)
        self.fine_Ny = int(self.Ly_fine/self.voxel_size)
        self.fine_Nz = int(self.Lz_fine/self.voxel_size)

        self.Nx = self.fine_Nx
        self.Ny = self.fine_Nx
        self.Nz = self.fine_Nx

        print('fine_Nx: ', self.fine_Nx)
        print('fine_Ny: ', self.fine_Ny)
        print('fine_Nz: ', self.fine_Nz)

        self.fine_soft_plus = SoftPlus(voxel_size=self.voxel_size_fine, alpha_init=1e-2)
        self.position_encoding = Embedding(3, 5)
        self.direction_encoding = Embedding(3, 4)

        self.fine_field = DVGOFineField(
            fine_Nx = self.fine_Nx, 
            fine_Ny = self.fine_Ny, 
            fine_Nz = self.fine_Nz,
            xyz_max = fine_xyz_max,
            xyz_min = fine_xyz_min,
            density_grid_channel = 1,
            feature_grid_channel = 12,
            fine_color_mlp_width = 128,
            fine_soft_plus = self.fine_soft_plus,
            position_encoding = self.position_encoding,
            direction_encoding = self.direction_encoding,
        )
        self.fine_field = self.fine_field.to('cuda')
    
    def voxel_count_view(self, dataset):
        count = dataset.voxel_count_views(1, self.Nx, self.Ny, self.Nz, 
                                                self.xyz_max, self.xyz_min, self.voxel_size)
        return count
    
    def ProgressiveScaling(self):
        print('Start Scaling')
        self.num_voxels = self.num_voxels*2
        self.voxel_size = torch.pow(self.Lx_fine*self.Ly_fine*self.Lz_fine/self.num_voxels,1/3)

        self.fine_Nx = int(torch.floor(self.Lx_fine/self.voxel_size))
        self.fine_Ny = int(torch.floor(self.Ly_fine/self.voxel_size))
        self.fine_Nz = int(torch.floor(self.Lz_fine/self.voxel_size))

        self.Nx = self.fine_Nx
        self.Ny = self.fine_Nx
        self.Nz = self.fine_Nx

        self.voxel_size_ratio = self.voxel_size/self.voxel_size_fine
        print('voxel_size: ', self.voxel_size)
        print('voxel_size_ratio: ', self.voxel_size_ratio)

        self.fine_field.scale_volume_grid(self.fine_Nx, self.fine_Ny, self.fine_Nz)
        print('new Nx: ', self.fine_Nx)
        print('new Ny: ', self.fine_Ny)
        print('new Nz: ', self.fine_Nz)
        print('Finish Scaling ')
    
    @torch.no_grad()
    def define_fine_bbox(self):
        print('Compute_fine_bbox: start')
        interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, int(self.coarse_Nx)),
        torch.linspace(0, 1, int(self.coarse_Ny)),
        torch.linspace(0, 1, int(self.coarse_Nz)),
        ), -1) 
        dense_xyz = self.coarse_xyz_min * (1-interp) + self.coarse_xyz_max * interp
        dense_xyz = dense_xyz.to('cuda')
        density = self.coarse_field.get_coarse_density(dense_xyz)
        alpha = dvgo_compute_alpha(density, 2.).squeeze()
        mask = (alpha > 1e-3)
        active_xyz = dense_xyz[mask]
        #self.fine_xyz_min = active_xyz.amin(0)
        #self.fine_xyz_max = active_xyz.amax(0)
        self.fine_xyz_min = torch.tensor([-0.6873, -1.1898, -0.5533])
        self.fine_xyz_max = torch.tensor([0.7330, 1.1971, 1.0907])
        print('Compute_fine_bbox: xyz_min', self.fine_xyz_min)
        print('Compute_fine_bbox: xyz_max', self.fine_xyz_max)
        return self.fine_xyz_min , self.fine_xyz_max

    def get_coarse_output(self, rays_o, rays_d):
        samples_per_ray = int(np.linalg.norm(np.array([self.coarse_Nx, self.coarse_Ny, self.coarse_Nz])+1) / 0.5) + 1
        rays_pts, z_vals = sample_pts_by_voxel_size(rays_o, rays_d, self.near, self.far, self.coarse_xyz_max,
                                                            self.coarse_xyz_min , 0.5*self.voxel_size, samples_per_ray)
        mask = compute_rays_pts_mask(rays_pts, self.coarse_xyz_max, self.coarse_xyz_min)
        density = self.coarse_field.get_coarse_density(rays_pts)
        rgb = self.coarse_field.get_coarse_color(rays_pts)
        rgb_map, density_map = dvgo_compute_map(rgb, density, z_vals, 0.5*self.voxel_size_ratio, mask)
        return rgb_map, density_map
    
    def get_fine_output(self, rays_o, rays_d, viewdirs):
        samples_per_ray = int(np.linalg.norm(np.array([self.fine_Nx, self.fine_Ny, self.fine_Nz])+1) / 0.5) + 1
        rays_pts, z_vals = sample_pts_by_voxel_size(rays_o, rays_d, self.near, self.far, self.fine_xyz_max,
                                                            self.fine_xyz_min, 0.5*self.voxel_size, samples_per_ray)
        mask = compute_rays_pts_mask(rays_pts, self.fine_xyz_max, self.fine_xyz_min)
        density = self.fine_field.get_fine_density(rays_pts)
        rgb = self.fine_field.get_fine_color(rays_pts,viewdirs)
        rgb_map, density_map = dvgo_compute_map(rgb, density, z_vals, 0.5*self.voxel_size_ratio, mask)
        return rgb_map, density_map

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

    height = 968
    width = 1296
    pixel_mode = 'center'
    rays_o, rays_d = get_view_rays(height, width, intrinsic, pose_c2w,
                                   pixel_mode)

    num_rays = 16

    xyz_max = torch.tensor([5.2, 5.2, 3.28])
    xyz_min = torch.tensor([2.2, 2.2, 1.28])

    rays_o_sm = rays_o.reshape(-1, 3)
    rays_d_sm = rays_d.reshape(-1, 3)

    dvgo = DVGO(
        xyz_max = xyz_max,
        xyz_min = xyz_min,
        near = 2,
        far = 6,
        num_voxels_coarse = 100*100*100,
        num_voxels_fine = 160*160*160,
        num_sample = 64,
    )

    rgb_map, density_map = dvgo.get_coarse_output(rays_o_sm, rays_d_sm)
    print(rgb_map.shape)
    print(density_map.shape)
