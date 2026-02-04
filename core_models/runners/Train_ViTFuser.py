
import os, sys
# Add the path to the first folder
folder1_path = '../'
sys.path.append(folder1_path)
import logging
import numpy as np
import torch
import sigpy as sp
import sigpy.mri as mr
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
# import custom libraries
from utils import transforms as T
from utils import complex_utils as cplx
# import custom classes
from utils.datasets import SliceData
from utils.subsample_fastmri import MaskFunc
import matplotlib.pyplot as plt
import argparse
from models.ViTFuserWrap import ViTFuserWrap
from models.FSloss_wrap import VGGPerceptualLoss
from models.vision_transformer import VisionTransformer
from fastmri.data import transforms, subsample
from fastmri.losses import SSIMLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

## Training params
# Create the argument parser
parser = argparse.ArgumentParser(description="Generating results")

# Define the arguments
parser.add_argument('--mask', type=str, default='1D', help="Mask used - 1D / Poisson")
parser.add_argument('--loss', type=str, default='MSE', help="Loss function used - MSE / SSIM / Hybrid")
parser.add_argument('--factor', type=float, default=3, help="Acceleration factor")
parser.add_argument('--snr', type=int, required=True, help="SNR input")
parser.add_argument('--seed', type=int, default=0, help="Seed used for randomizing")
parser.add_argument('--epochs', type=int, default=100, help="Number of Epochs")
parser.add_argument('--loss_lambda', type=int, default=1, help="Hybrid loss lambda")
parser.add_argument('--outfolder', type=str, default='assets', help="output folder")

# Parse the arguments
args = parser.parse_args()

# %%
SNR = args.snr # [dB]
acc_factor = args.factor
mask_type = args.mask # 1D / Poisson
exp_dir = args.outfolder
loss_type = args.loss # MSE / SSIM / Hybrid
loss_lambda = args.loss_lambda
model_name = 'ViTFuser'
if loss_type == 'MSE' or loss_type == 'SSIM':
    model_name = 'ViTFuser' + loss_type
model_name_endings = ''
SEED = args.seed


# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# Data process for simulation 
# select parameters manualy

class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()
        self.snr = args.snr 
        self.mask_type = args.mask_type
        self.factor = args.factor

    def get_mask_func(self, factor):
        center_fractions = 0.08 * 4/factor# RandomMaskFunc EquiSpacedMaskFunc
        mask_func = subsample.EquiSpacedMaskFunc(
        center_fractions=[center_fractions],
        accelerations=[factor], 
        )
        return mask_func
    
    def __call__(self, kspace, target, reference_kspace, reference,slice):
        ## Normalize the data
        im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(256,24)),(256,160))))
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        kspace = kspace/scale
        target = target/scale
        
        ## Convert everything from numpy arrays to tensors and crop for low-resolution
        kspace_torch = cplx.to_tensor(kspace).float()   
        target_torch = cplx.to_tensor(target).float()  
        target_torch = T.ifft2(T.kspace_cut(T.fft2(target_torch),0.67,0.67)) 

        ## Select acceleration mask
        if self.mask_type == 'Poisson':
            # Use poisson mask instead
            mask2 = mr.poisson((172,108),self.factor, calib=(18,14), dtype=float, crop_corner=False, return_density=True, seed=0, max_attempts=6, tol=0.01)
            mask2[86-10:86+9,54-8:54+7] = 1
            mask_torch_pois = torch.stack([torch.tensor(mask2).float(),torch.tensor(mask2).float()],dim=2)
            kspace_torch = T.awgn_torch(kspace_torch,self.snr,L=1) 
            kspace_torch = T.kspace_cut(kspace_torch,0.67,0.67)
            kspace_torch = kspace_torch*mask_torch_pois
        else:
            ## Masking 1D
            kspace_torch = T.awgn_torch(kspace_torch,self.snr,L=1)
            mask_func = self.get_mask_func(self.factor) 
            kspace_torch = T.kspace_cut(kspace_torch,0.67,0.67)
            kspace_torch = transforms.apply_mask(kspace_torch, mask_func)[0]


        mask = np.abs(cplx.to_numpy(kspace_torch))!=0
        mask_torch = torch.stack([torch.tensor(mask).float(),torch.tensor(mask).float()],dim=2)
        
        ### Reference addition ###
        im_lowres_ref = abs(sp.ifft(sp.resize(sp.resize(reference_kspace,(256,24)),(256,160))))
        magnitude_vals_ref = im_lowres_ref.reshape(-1)
        k_ref = int(round(0.05 * magnitude_vals_ref.shape[0]))
        scale_ref = magnitude_vals_ref[magnitude_vals_ref.argsort()[::-1][k_ref]]
        reference = reference / scale_ref
        reference_torch = cplx.to_tensor(reference).float()
        reference_torch_kspace = T.fft2(reference_torch)
        reference_torch_kspace = T.kspace_cut(reference_torch_kspace,0.67,0.67)
        reference_torch = T.ifft2(reference_torch_kspace)
        

        return kspace_torch,target_torch,mask_torch, reference_torch 


# Helper functions for dataset
def create_datasets(args):
    # Generate k-t undersampling masks
    train_mask = MaskFunc([0.08],[4])
    train_data = SliceData(
        root=str(args.data_path),
        transform=DataTransform(train_mask, args),
        sample_rate=1
    )
    return train_data
def create_data_loaders(args):
    train_data = create_datasets(args)
#     print(train_data[0])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader
def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


# Set params
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# %%
#Hyper parameters
params = Namespace()
params.data_path = "../registered_data/"
params.batch_size = 24 
params.num_grad_steps = 1 
params.num_cg_steps = 8 
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.0001 
params.weight_decay = 0
params.lr_step_size = 6
params.lr_gamma = 0.3
params.epoch = 101
params.reference_mode = 1
params.reference_lambda = 0.1
params.snr = SNR
params.factor = acc_factor
params.mask_type = mask_type


train_loader = create_data_loaders(params)


# Models creation
VGGloss = VGGPerceptualLoss().to(device)

net = VisionTransformer(
  avrg_img_size=320,
  patch_size = (10,10),
  in_chans=1,
  embed_dim=64,
  depth=10,
  num_heads=16

)

model = ViTFuserWrap(params).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer, 
    max_lr=0.0002,
    steps_per_epoch=len(train_loader),
    epochs=100,
    pct_start=0.01,
    anneal_strategy='linear',
    cycle_momentum=False,
    base_momentum=0., 
    max_momentum=0.,
    div_factor = 25.,
    final_div_factor=1.,
)



# Training loop
criterion = SSIMLoss().to(device)
criterionMSE = nn.MSELoss()

epochs_plot = []
losses_plot = []

for epoch in range(params.epoch):
    model.train()
    avg_loss = 0.
    running_loss = 0.0
    for iter, data in enumerate(train_loader):
        input,target,mask,reference = data
        input = input.to(device).float()
        target = target.to(device).float()
        mask = mask.to(device)
        reference = reference.to(device).float()
        image = T.ifft2(input)
        image = image.permute(0,3,1,2)


        target_image = target.permute(0,3,1,2) 
        real_part_tar = target_image[:,0,:,:].unsqueeze(1)
        imag_part_tar = target_image[:,1,:,:].unsqueeze(1)
        mag_tar = torch.sqrt(real_part_tar**2 + imag_part_tar**2).to(device)

        reference_image = reference.permute(0,3,1,2)
        real_part_ref = reference_image[:,0,:,:].unsqueeze(1)
        imag_part_ref = reference_image[:,1,:,:].unsqueeze(1)
        mag_ref = torch.sqrt(real_part_ref**2 + imag_part_ref**2).to(device)



        im_out = model(input,reference)

        maxval = torch.max(torch.cat((im_out,mag_tar.permute(0,2,3,1)),dim=1))
        maxval = torch.max(torch.cat((im_out,mag_ref.permute(0,2,3,1)),dim=1))
        im_out = im_out.permute(0,3,1,2)
        data_range = torch.tensor([maxval], device=device).view(1, 1, 1, 1).expand(im_out.size(0), im_out.size(1), im_out.size(2)-6, im_out.size(3)-6)
        
        ## Losses
        if loss_type == 'SSIM':
            ## SSIM loss
            loss = criterion(im_out, mag_tar.to(device), data_range.to(device)) 
        else:
            if loss_type == 'MSE':
                ## MSE loss
                loss = criterionMSE(im_out,mag_tar.to(device)) #MSE
            else:
                ## Hybrid loss
                loss = loss_lambda*VGGloss(im_out,mag_tar.to(device))  +  criterion(im_out, mag_tar.to(device), data_range.to(device)) 

     
        
        running_loss = running_loss + loss.item()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        if iter % 400 == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{params.epoch:3d}] '
                f'Iter = [{iter:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g}'
            )
    #Saving the model
    if epoch % 100 == 0:
        torch.save(
            {
                'epoch': epoch,
                'params': params,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model_'+ model_name+ model_name_endings+'_%d.pt'%(epoch))
    )
    running_loss = running_loss / len(train_loader)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}, Learning rate: {current_lr}')

    # Append epoch and average loss to plot lists
    epochs_plot.append(epoch)
    losses_plot.append(running_loss)



