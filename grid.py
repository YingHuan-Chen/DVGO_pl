import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat

class DenseGrid(nn.Module):
    def __init__(self, Nx, Ny, Nz, channel, xyz_max, xyz_min):
        super(DenseGrid, self).__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.xyz_max = xyz_max
        self.xyz_min = xyz_min
        self.channel = channel
        self.grid = nn.Parameter(torch.zeros([1, self.channel, Nx, Ny, Nz]))
    
    def forward(self, query):
        shape = query.shape[:-1]
        xyz_min = self.xyz_min.to(query.device)
        xyz_max = self.xyz_max.to(query.device)
        query = ((query - xyz_min)/(xyz_max - xyz_min)).flip((-1,)) * 2 - 1
        query = query.reshape(1, 1, 1, -1, 3)
        output = F.grid_sample(self.grid, query, mode='bilinear', align_corners=True)
        #output = output.reshape(r, n, -1)
        output = output.reshape(self.channel,-1).T.reshape(*shape,self.channel)
        return output
    
    def scale_volume_grid(self, new_Nx, new_Ny, new_Nz):
        self.grid = nn.Parameter(
            F.interpolate(
                self.grid.data,
                size=(new_Nx, new_Ny, new_Nz),
                mode="trilinear",
                align_corners=True
            )
        )

if __name__ == "__main__":
    grid = DenseGrid(Nx=100, Ny=100, Nz=100, channel=5)
    x = torch.rand(1000,64,3)
    output = grid(x)
    print(output.shape)

