import torch
from torch import nn
from grid import DenseGrid
from soft_plus import SoftPlus
from embedding import Embedding
from einops import repeat,rearrange

class DVGOFineField(nn.Module):
    """DVGO Field
    Args:
    """

    def __init__(
        self,
        fine_Nx, fine_Ny, fine_Nz,
        xyz_max, xyz_min,
        fine_soft_plus,
        density_grid_channel,
        feature_grid_channel,
        fine_color_mlp_width,
        position_encoding,
        direction_encoding 
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.fine_Nx = fine_Nx
        self.fine_Ny = fine_Ny
        self.fine_Nz = fine_Nz

        self.xyz_max = xyz_max
        self.xyz_min = xyz_min

        self.density_grid_channel = density_grid_channel
        self.feature_grid_channel = feature_grid_channel

        self.fine_soft_plus = fine_soft_plus


        self.fine_feature_grid = DenseGrid(
            Nx = self.fine_Nx, 
            Ny = self.fine_Ny, 
            Nz = self.fine_Nz, 
            channel = self.feature_grid_channel,
            xyz_max = self.xyz_max,
            xyz_min = self.xyz_min
        )

        self.fine_density_grid = DenseGrid(
            Nx = self.fine_Nx, 
            Ny = self.fine_Ny, 
            Nz = self.fine_Nz, 
            channel = self.density_grid_channel,
            xyz_max = self.xyz_max,
            xyz_min = self.xyz_min
        )

        self.mlp_input_dimension = self.position_encoding.get_out_dim() + \
                                        self.direction_encoding.get_out_dim() \
                                        + self.feature_grid_channel

        self.fine_color_mlp_width = fine_color_mlp_width

        self.fine_color_head = nn.Sequential(
            nn.Linear(self.mlp_input_dimension,self.fine_color_mlp_width),
            nn.ReLU(),
            nn.Linear(self.fine_color_mlp_width,self.fine_color_mlp_width),
            nn.ReLU(),
            nn.Linear(self.fine_color_mlp_width, 3),
            nn.Sigmoid()
        )

    def scale_volume_grid(self,new_Nx, new_Ny, new_Nz):
        self.fine_feature_grid.scale_volume_grid(new_Nx, new_Ny, new_Nz)
        self.fine_density_grid.scale_volume_grid(new_Nx, new_Ny, new_Nz)

    def get_fine_density(self, query):
        output = self.fine_density_grid(query)
        output = self.fine_soft_plus(output)
        return output

    def get_fine_color(self, query, viewdir):
        output = self.fine_feature_grid(query)
        position = self.position_encoding(query)
        view = self.direction_encoding(viewdir)
        view = repeat(view,'r c -> r n c', n=output.shape[1])
        mlp_in = torch.cat([output.to(position.device),position,view],dim=-1)
        #output = torch.tensor([]).cuda()

        output = self.fine_color_head(mlp_in)
        '''
        r = mlp_in.shape[0]
        mlp_in = rearrange(mlp_in,'r n c -> (r n) c')
        print()
        for i in range(0, int(mlp_in.shape[0]), 32*1024):
            buffer = self.fine_color_head(mlp_in[i:i+32*1024])
            output = torch.cat((output,buffer), dim=0)
        output = rearrange(output,'(r n) c -> r n c',r=r)
        '''
        return output 