import os, sys
import torch

from torch import nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.data import DataLoader
from blender_dataset import BlenderDataset

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# models
from dvgo import DVGO
from optimizer import Adam

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from einops import rearrange

from visualization import visualize_depth



class DVGOTrainer(LightningModule):
    def __init__(self):
        super(DVGOTrainer, self).__init__()
        self.rgb_loss = nn.MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.lr = 0.1
        self.decay_steps = 20000
        self.decay_factor = 0.1 ** (self.global_step/self.decay_steps)
        #self.ssim = structural_similarity_index_measure
        #self.lpips = LearnedPerceptualImagePatchSimilarity()
    
    def setup(self, stage=None):
        """
        setup dataset for each machine
        """
        self.train_dataset = BlenderDataset(
            root_dir = 'data/nerf_synthetic/lego', split='train', img_wh=(800, 800)
        )
        
        self.val_dataset = BlenderDataset(
            root_dir = 'data/nerf_synthetic/lego', split='val', img_wh=(800, 800)
        )

        self.xyz_max_coarse , self.xyz_min_coarse = self.train_dataset.define_bbox()
        self.near, self.far =  self.train_dataset.get_bound()

        self.model = DVGO(
            xyz_max = self.xyz_max_coarse,
            xyz_min = self.xyz_min_coarse,
            near = self.near,
            far = self.far,
            num_voxels_coarse = 1024000,
            num_voxels_fine = 160*160*160
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size = 8192,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=True,
            num_workers=4,
            batch_size = 1,
            pin_memory=True
        )

    def configure_optimizers(self):
        '''
        print('============================================================================')
        for name, param in self.model.state_dict().items():
            print(name, param.size())
        print('============================================================================')
        '''
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        if  self.global_step < 10000:
            count = self.model.voxel_count_view(self.train_dataset)
            self.optimizer.set_pervoxel_lr(count)
        return self.optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=8192,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step in [11000, 12000, 13000]:#[11000, 12000, 13000, 14000]:[20, 30, 40, 50]
            self.model.ProgressiveScaling()
            self.configure_optimizers()
    
    def training_step(self, batch, batch_nb):
        ray_o, ray_d, rgbs, viewdirs = batch['ray_o'].to('cuda'), batch['ray_d'].to('cuda'), batch['rgbs'].to('cuda'), batch['viewdirs'].to('cuda')
        if self.global_step < 10000:
            rgb_map, density_map = self.model.get_coarse_output(rays_o=ray_o, rays_d=ray_d)
        else:
            rgb_map, density_map = self.model.get_fine_output(rays_o=ray_o, rays_d=ray_d, viewdirs=viewdirs)
        rgb_map = rgb_map.clamp(0, 1)
        loss = self.rgb_loss(rgb_map, rgbs)
        self.log('train/loss', loss)
        
        with torch.no_grad():
            psnr_ = self.psnr(rgb_map, rgbs)
            self.log('train/psnr',  psnr_)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * self.decay_factor

        if self.global_step == 10000:
            fine_xyz_min , fine_xyz_max = self.model.define_fine_bbox()
            self.model.start_fine(fine_xyz_min , fine_xyz_max)
            self.configure_optimizers()
            print("Starts fine training!")


    def validation_step(self, batch, batch_nb):
        ray_o, ray_d, rgbs, viewdirs = batch['ray_o'].to('cuda'), batch['ray_d'].to('cuda'), batch['rgbs'].to('cuda'), batch['viewdirs'].to('cuda')
        ray_o = ray_o.squeeze()
        ray_d = ray_d.squeeze()
        rgbs = rgbs.squeeze()
        viewdirs = viewdirs.squeeze()
        #rgb_map, density_map = self.model.get_coarse_output(rays_o=ray_o, rays_d=ray_d) 

        rgb_map = torch.tensor([]).cuda()
        density_map = torch.tensor([]).cuda()

        for i in range(0, int(ray_o.shape[0]), 8000):
            if self.global_step < 10000:
                rgb_render, density_render = self.model.get_coarse_output(rays_o=ray_o[i:i+8000], 
                                                                    rays_d=ray_d[i:i+8000]) 
            else:
                rgb_render, density_render = self.model.get_fine_output(rays_o=ray_o[i:i+8000], 
                                                                    rays_d=ray_d[i:i+8000],
                                                                    viewdirs = viewdirs[i:i+8000])
            rgb_render = rgb_render.clamp(0, 1)
            rgb_map = torch.cat((rgb_map,rgb_render), dim=0)
            density_map = torch.cat((density_map,density_render), dim=0)

        if batch_nb == 4:
            W, H = 800, 800
            img = rgb_map.view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(density_map.view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
        '''
        rgb_map = rearrange(rgb_map, '(h w) c -> c h w',h = 800)
        rgbs = rearrange(rgbs, '(h w) c -> c h w',h = 800)
        '''

        log = {'val_loss': self.rgb_loss(rgb_map, rgbs.to(rgb_map.device)),
                'val_psnr':  self.psnr(rgb_map, rgbs.to(rgb_map.device))} 
        
        return log
        

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)


if __name__ == '__main__':
    system = DVGOTrainer()
    checkpoint_callback = ModelCheckpoint(filename=os.path.join(f'ckpts/exp',
                                                                '{epoch:d}'),
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=5,)

    callbacks = [checkpoint_callback]

    logger = TensorBoardLogger(save_dir="logs",
                               name='exp',
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=4,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      gpus=[2],
                      num_sanity_val_steps=1,
                      benchmark=True)

    trainer.fit(system)