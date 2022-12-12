import torch
from torch import nn
import torch.nn.functional as F

class SoftPlus(nn.Module):
    def __init__(
        self, 
        voxel_size, 
        alpha_init = 1e-6
    ):
        super().__init__()
        self.alpha_init = torch.FloatTensor([alpha_init])
        self.voxel_size = voxel_size
        power = -(1/self.voxel_size)
        #self.b = torch.log(torch.pow(1-(self.alpha_init).to(power.device), power) - 1)
        self.b = torch.log(1/(1-self.alpha_init) - 1)
        print('b:', self.b)
    
    def forward(self, pre_density):
        self.b = self.b.to(pre_density.device)
        return F.softplus(pre_density + self.b)

if __name__ == "__main__":
    soft = SoftPlus(1.,1e-6)
    density = torch.FloatTensor([5])
    output = soft(density)
    print(output)