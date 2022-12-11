import torch
from torch import nn
from grid import DenseGrid
from soft_plus import SoftPlus
from embedding import Embedding

class DVGOCoarseField(nn.Module):
    """DVGO Field
    Args:
    """

    def __init__(
        self,
        coarse_Nx, coarse_Ny, coarse_Nz,
        xyz_max, xyz_min,
        coarse_soft_plus,
        density_grid_channel,
        feature_grid_channel 
    ):
        super().__init__()

        self.coarse_Nx = coarse_Nx
        self.coarse_Ny = coarse_Ny
        self.coarse_Nz = coarse_Nz

        self.xyz_max = xyz_max
        self.xyz_min = xyz_min 
        
        self.density_grid_channel = density_grid_channel
        self.feature_grid_channel = feature_grid_channel

        self.coarse_soft_plus = coarse_soft_plus

        
        self.coarse_density_grid = DenseGrid(
            Nx = self.coarse_Nx, 
            Ny = self.coarse_Ny, 
            Nz = self.coarse_Nz, 
            channel = self.density_grid_channel,
            xyz_max = self.xyz_max,
            xyz_min = self.xyz_min
        )

        self.coarse_feature_grid = DenseGrid(
            Nx = self.coarse_Nx, 
            Ny = self.coarse_Ny, 
            Nz = self.coarse_Nz, 
            channel = self.feature_grid_channel,
            xyz_max = self.xyz_max,
            xyz_min = self.xyz_min
        )

    def get_coarse_density(self, query):
        output = self.coarse_density_grid(query)
        output = self.coarse_soft_plus(output)
        return output

    def get_coarse_color(self, query):
        output = self.coarse_feature_grid(query)
        output = torch.sigmoid(output)
        return output
