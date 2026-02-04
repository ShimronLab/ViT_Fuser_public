"""
All rights reserved.
Tal Oved, 2026.
"""

import os, sys
# Add the path to the first folder
folder1_path = '../'
sys.path.append(folder1_path)
import logging
import numpy as np
import torch
import sigpy as sp
# import custom libraries
from utils import transforms as T
from utils import complex_utils as cplx
# import custom classes
from models.MoDL_single import UnrolledModel
from models.ViTFuserWrap import ViTFuserWrap
from models.ViTWrap import ViTWrap
import matplotlib.pyplot as plt
import nibabel as nib
from fastmri.data import transforms, subsample
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device_num = "1"

parser = argparse.ArgumentParser(description="Generating results")

# Define the arguments
parser.add_argument('--subject', type=int, default=2, help="Subject number")
parser.add_argument('--slice', type=int, default=25, help="Slice number")
parser.add_argument('--mask', type=str, default='1D', help="Mask used")
parser.add_argument('--factor', type=int, default=3, help="Acceleration factor")
parser.add_argument('--snr', type=int, required=True, help="SNR input")
parser.add_argument('--seed', type=int, default=0, help="Seed used for randomizing")
parser.add_argument('--outfolder', type=str, default='assets', help="output folder")


# Parse the arguments
args = parser.parse_args()

# Access the arguments
patient = args.subject
slice_number = args.slice
SNR = args.snr
mask_type = args.mask
factor = args.factor
SEED = args.seed
output_folder = args.outfolder

# Masks
# Define acceleration factors
acceleration_factors = [2, 4, 6.1, 8, 10.1, 12.1]
# Define mask shape and calibration region
shape = (172, 108)
calib = (18, 14)
# Define patch to be fully sampled
patch_center = (86, 54)  # Center of the patch
patch_size = (19, 15)  # (rows, cols)


# Select params
DPS_ES = 1
DPS_concat = 0
minus_factor = 0
SEED = 0

# Paper experiments:
# Subject 3 slice 24
# Subject 2 slice 24
# Subject 1 slice 20

# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# # Load data

# %%
# Path to your NIfTI file
if patient == 1:
    # Patient 1
    nii_file_target = '../test_data/patient29b/T1_week152reg.nii'
    nii_file_ref = '../test_data/patient29b/T1_week165regT1_week152.nii'
else:
    if patient == 2:
        # Patient 2
        nii_file_target = '../test_data/patient28/T1_week25reg.nii'
        nii_file_ref = '../test_data/patient28/T1_week35regT1_week25.nii'
    else:
        if patient == 3:
            #patient 3
            nii_file_target = '../test_data/patient72b/T1_week29reg.nii'
            nii_file_ref = '../test_data/patient72b/T1_week37regT1_week29.nii'   
        else: 
            if patient ==4:
                #patient 4
                nii_file_target = '../test_data/subject34_test/T1_week28reg.nii'
                nii_file_ref = '../test_data/subject34_test/T1_week40regT1_week28.nii'   

img_target = nib.load(nii_file_target)
img_ref = nib.load(nii_file_ref)
target = img_target.get_fdata()[...,slice_number]
reference = img_ref.get_fdata()[...,slice_number]



random_phase = torch.angle(T.random_map((1,256,160), 'cpu',kspace_radius_range=(0.001, 0.001))) 
target = target * (torch.exp(1j * random_phase)).numpy() 
reference = reference * (torch.exp(1j * random_phase)).numpy()
target = target.squeeze(0)
target_torch = cplx.to_tensor(target).float() 
reference_torch = cplx.to_tensor(reference).float() 
reference_kspace_torch = T.fft2(reference_torch)
reference_kspace = cplx.to_numpy(reference_kspace_torch)
kspace_torch = T.fft2(target_torch)
kspace_prior_torch = T.fft2(reference_torch)
target = cplx.to_numpy(target_torch)
kspace = cplx.to_numpy(kspace_torch)
kspace_prior = cplx.to_numpy(kspace_prior_torch)
mask2 = sp.mri.poisson((172,108),factor, calib=(18,14), dtype=float, crop_corner=False, return_density=True, seed=0, max_attempts=6, tol=0.01)
mask2[86-10:86+9,54-8:54+7] = 1
mask_torch_pois = torch.stack([torch.tensor(mask2).float(),torch.tensor(mask2).float()],dim=2)

## Print 1/acceleration  
s = (172)*(108)
print((torch.sum(mask_torch_pois))/s/2)

factor = np.int8(factor - minus_factor)

# %% [markdown]
# # Simulation

# %%
im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(256,24)),(256,160))))
magnitude_vals = im_lowres.reshape(-1)
k = int(round(0.05 * magnitude_vals.shape[0]))
scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
kspace = kspace/scale
target = target/scale
#kspace_prior = kspace_prior/scale


# Apply kspace crop on target
target_torch = cplx.to_tensor(target)
### prior ###
#kspace_prior_torch = cplx.to_tensor(kspace_prior).float()
#kspace_prior_torch = T.kspace_cut(kspace_prior_torch,0.67,0.67)
##############################
target_torch = T.ifft2( T.kspace_cut(T.fft2(target_torch),0.67,0.67))
target = cplx.to_numpy(target_torch)
# Convert everything from numpy arrays to tensors
kspace_torch = cplx.to_tensor(kspace).float()

kspace_torch = T.awgn_torch(kspace_torch,SNR,L=1)
kspace_noised = kspace_torch.clone()
kspace_noised = T.kspace_cut(kspace_noised,0.67,0.67)
kspace_torch = T.kspace_cut(kspace_torch,0.67,0.67)
input_torch = T.ifft2( kspace_torch)
target_torch = cplx.to_tensor(target).float()




### Reference addition ###
im_lowres_ref = abs(sp.ifft(sp.resize(sp.resize(reference_kspace,(256,24)),(256,160))))
magnitude_vals_ref = im_lowres_ref.reshape(-1)
k_ref = int(round(0.05 * magnitude_vals_ref.shape[0]))
scale_ref = magnitude_vals_ref[magnitude_vals_ref.argsort()[::-1][k_ref]]
reference_DPS_PI = reference / scale
reference_DPS_PI_torch = cplx.to_tensor(reference_DPS_PI).float()
reference_DPS_PI_torch_kspace = T.fft2(reference_torch.squeeze(0))
reference_DPS_PI_torch_kspace = T.kspace_cut(reference_DPS_PI_torch_kspace,0.67,0.67)

reference = reference / scale_ref
reference_torch = cplx.to_tensor(reference).float()
reference_torch_kspace = T.fft2(reference_torch.squeeze(0))
reference_torch_kspace = T.kspace_cut(reference_torch_kspace,0.67,0.67)
reference_torch = T.ifft2(reference_torch_kspace)

def get_mask_func(factor, seed=1):
    torch.manual_seed(seed)  # Set PyTorch seed
    np.random.seed(seed)  # Set NumPy seed   
    center_fractions = 0.08 * 4 / factor
    mask_func = subsample.EquiSpacedMaskFunc(
        center_fractions=[center_fractions],
        accelerations=[factor],
        seed=seed  
    )
    return mask_func

if mask_type == '1D':
    ## 1d
    mask_func = get_mask_func(factor)
    kspace_torch, mask_torch_1d, _ = transforms.apply_mask(kspace_torch, mask_func)    
    # Ensure mask is the right shape for export later (H, W, 1) or similar
    # fastMRI usually returns mask as (1, 1, W, 1) or similar.
    # We need to broadcast it to match the image shape (H, W)
    # Assuming kspace_torch is (H, W, 2)
    H, W = kspace_torch.shape[0], kspace_torch.shape[1]
    
    # Broadcast 1D mask to 2D
    # mask_torch_1d might be on different device or shape, let's standarize
    # Usually shape is (1, 1, W, 1).
    current_mask = mask_torch_1d.squeeze().unsqueeze(0).repeat(H, 1) # Shape (H, W)
    
    # Add extra dims for export if needed (1, 1, H, W)
    mask_export = current_mask.unsqueeze(0).unsqueeze(0).cpu()
else:
    ## 2d poisson
    kspace_torch = kspace_torch*mask_torch_pois
    

mask_np = np.abs(cplx.to_numpy(kspace_torch))!=0
print(f'Mask torch size: {mask_np.shape}')
s = (172)*(108)
print(f'Acceleration factor R: {np.sum(mask_np)/s}')



# %%
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# %%
#Hyper parameters
params = Namespace()
params.data_path = "../registered_data/"
params.batch_size = 2
params.num_grad_steps = 1 
params.num_cg_steps = 8 
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.00001
params.weight_decay = 0
params.lr_step_size = 7
params.lr_gamma = 0.1
params.epoch = 61
params.reference_mode = 0
params.reference_lambda = 0.1

# %% [markdown]
# # Load checkpoints and models 

# %%
## load checkpoints and models  
checkpoint_file_MoDL = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_MoDLTests_100.pt" 

checkpoint_file_vit1 = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTMSE_100.pt"
checkpoint_file_vit2 = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViT_100.pt" 
checkpoint_file_vit3 = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViT_100.pt" 

checkpoint_file_vitfuser1 = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTFuser_100.pt"
checkpoint_file_vitfuser2 = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTFuserMSE_100.pt" 
checkpoint_file_vitfuser3 = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTFuserMSE_100.pt" 


## Test different ViT-Fuser models
checkpoint_file_vitfuser2 = "../checkpoints_forbenius_" + f"{SNR}"+ "dB/model_ViTFuser_100.pt" 

# Poisson checkpoints
if mask_type == '2D':
    checkpoint_file_MoDL = "../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{factor}" "_paper/model_MoDLTests_100.pt" 
    checkpoint_file_vit1 = "../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{factor}" "_paper/model_ViTMSE_100.pt"
    checkpoint_file_vitfuser1 = "../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{factor}" "_paper/model_ViTFuser_100.pt"
    checkpoint_file_vitfuser3 = "../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{factor}" "_paper/model_ViTFuserMSE_100.pt" 


checkpoint_MoDL = torch.load(checkpoint_file_MoDL,map_location=device)
checkpoint_vit1 = torch.load(checkpoint_file_vit1,map_location=device)
checkpoint_vit2 = torch.load(checkpoint_file_vit2,map_location=device)
checkpoint_vit3 = torch.load(checkpoint_file_vit3,map_location=device)

checkpoint_vitfuser1 = torch.load(checkpoint_file_vitfuser1,map_location=device)

## For MSE ViT-Fuser models test
#checkpoint_vitfuser2 = torch.load(checkpoint_file_vitfuser2,map_location=device)
#checkpoint_vitfuser3 = torch.load(checkpoint_file_vitfuser3,map_location=device)

checkpoint_vitfuser2 = torch.load(checkpoint_file_vitfuser1,map_location=device)
checkpoint_vitfuser3 = torch.load(checkpoint_file_vitfuser1,map_location=device)
##

params_MoDL = checkpoint_MoDL["params"] #Different params the ViT, ViT-Fuser



model_vit1 = ViTWrap(params).to(device)
model_vit2 = ViTWrap(params).to(device)
model_vit3 = ViTWrap(params).to(device)
model_vitfuser1 = ViTFuserWrap(params).to(device)
model_vitfuser2 = ViTFuserWrap(params).to(device)
model_vitfuser3 = ViTFuserWrap(params).to(device)
model_MoDL = UnrolledModel(params_MoDL).to(device)

# load checkpoint
model_vit1.load_state_dict(checkpoint_vit1['model'])
model_vit2.load_state_dict(checkpoint_vit2['model'])
model_vit3.load_state_dict(checkpoint_vit3['model'])
model_vitfuser1.load_state_dict(checkpoint_vitfuser1['model'])
model_vitfuser2.load_state_dict(checkpoint_vitfuser2['model'])
model_vitfuser3.load_state_dict(checkpoint_vitfuser3['model'])
model_MoDL.load_state_dict(checkpoint_MoDL['model'])


# %% [markdown]
# # Run models

# %%


img = cplx.to_tensor(np.abs(cplx.to_numpy(T.ifft2(kspace_torch)))).permute(2,0,1).unsqueeze(0).to(device)
img_chan = img[:,0,:,:].unsqueeze(0)
ref = cplx.to_tensor(np.abs(cplx.to_numpy(reference_torch))).permute(2,0,1).unsqueeze(0).to(device)
ref_chan = ref[:,0,:,:].unsqueeze(0)
ref_np = ref_chan.cpu().numpy()[0,0,:,:]
img_padded_np = img_chan.cpu().numpy()[0,0,:,:]


## MoDL
im_out_MoDL = model_MoDL(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_MoDL = np.abs(cplx.to_numpy(im_out_MoDL.cpu().detach()))


## Vit1
im_out_vit1 = model_vit1(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vit1 = np.abs(cplx.to_numpy(im_out_vit1.cpu().detach()))
print(im_out_vit1.shape)

## Vit2
im_out_vit2 = model_vit2(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vit2 = np.abs(cplx.to_numpy(im_out_vit2.cpu().detach()))

## Vit3
im_out_vit3 = model_vit3(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vit3 = np.abs(cplx.to_numpy(im_out_vit3.cpu().detach()))


## Vit fuser 1
im_out_vitfuser1 = model_vitfuser1(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vitfuser1 = T.ifft2(T.fft2(im_out_vitfuser1))
target_torch = T.ifft2(T.fft2(cplx.to_tensor(target)))
target = cplx.to_numpy(target_torch.cpu().detach())
im_out_vitfuser1 = np.abs(cplx.to_numpy(im_out_vitfuser1.cpu().detach()))


## Vit fuser 2
im_out_vitfuser2 = model_vitfuser2(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vitfuser2 = T.ifft2(T.fft2(im_out_vitfuser2))
im_out_vitfuser2 = np.abs(cplx.to_numpy(im_out_vitfuser2.cpu().detach()))


## Vit fuser 3
im_out_vitfuser3 = model_vitfuser3(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vitfuser3 = T.ifft2(T.fft2(im_out_vitfuser3))
im_out_vitfuser3 = np.abs(cplx.to_numpy(im_out_vitfuser3.cpu().detach()))



# %% [markdown]
# # DPS

# %%
# %% [markdown]
# # Run DPS via External Script

import subprocess
# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------
# Path to the root of the EDM repo
repo_path = "../../diffusion_mri/EDM"
script_path = os.path.join(repo_path, "dps.py")

# Temporary folder to exchange data between this notebook and the script
temp_root = "../temp_dps_compare_io"
meas_path = os.path.join(temp_root, "measurements")
ksp_path = os.path.join(temp_root, "kspace")
out_dir = os.path.join(temp_root, "results")

os.makedirs(meas_path, exist_ok=True)
os.makedirs(ksp_path, exist_ok=True)

# Model Settings
model_path = "../../diffusion_mri/models/edm/brain/32dB/00019-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"
target_dims = (172, 108) # Network resolution
inference_R = int(factor) # e.g. 6

# -------------------------------------------------------------------------
# 2. EXPORT DATA 
# -------------------------------------------------------------------------
print(f"Exporting data for DPS script (R={inference_R})...")

# A. Prepare K-Space
if kspace_torch.ndim == 3 and kspace_torch.shape[-1] == 2:
    ksp_cplx = torch.view_as_complex(kspace_torch.cpu())
    ksp_reference_cplx = torch.view_as_complex(reference_torch_kspace.cpu())
else:
    ksp_cplx = kspace_torch.cpu()
    ksp_reference_cplx = reference_torch_kspace.cpu()


ksp_padded = ksp_cplx
ksp_reference_padded = ksp_reference_cplx

# Save with shape [1, 1, 384, 320] (Coils, H, W)
ksp_export = ksp_padded.unsqueeze(0)
ksp_reference_export = ksp_reference_padded.unsqueeze(0)

# B. Prepare Maps (Ones)
s_map_export = torch.ones_like(ksp_export)

# C. Prepare Mask

if mask_type == '1D':
    mask_real = mask_torch_1d.cpu().repeat(172,1,1).squeeze(-1) 
else: 
    mask_real = mask_torch_pois[:,:,0]

mask_padded = torch.zeros(target_dims, dtype=torch.float32)
mask_padded = mask_real.float()

# Save with shape [1, 1, 384, 320]
mask_export = mask_padded.unsqueeze(0).unsqueeze(0)

# D. Prepare Ground Truth
gt_cplx = torch.view_as_complex(target_torch.cpu())
gt_padded = gt_cplx

# Calculate normalization constant from zero-filled like in the training process
zf_ref = sp.ifft(ksp_cplx.numpy() * mask_real.numpy())
norm_val = np.percentile(np.abs(zf_ref), 99)
if norm_val == 0: norm_val = 1.0
ksp_export = ksp_padded / norm_val
ksp_reference_export = ksp_reference_padded / norm_val
ksp_export_saved = ksp_export.clone()

# Save to disk
torch.save({'ksp': ksp_export, 's_map': s_map_export, 'prior_ksp': ksp_reference_export}, os.path.join(ksp_path, "sample_0.pt"))
torch.save({'gt': gt_padded, f'mask_{inference_R}': mask_export}, os.path.join(meas_path, "sample_0.pt"))

# -------------------------------------------------------------------------
# 3. EXECUTE EXTERNAL SCRIPT
# -------------------------------------------------------------------------
print("Running dps.py subprocess...")
if DPS_ES == 0:
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", script_path,
        "--gpu", device_num,
        "--seed", str(SEED),
        "--sample_start", "0",
        "--sample_end", "1",
        "--inference_R", str(inference_R),
        "--inference_snr", "32dB", # Just for folder naming
        "--num_steps", "500",
        "--S_churn", "0",
        "--measurements_path", meas_path,
        "--ksp_path", ksp_path,
        "--network", model_path,
        "--outdir", out_dir
    ]
else:
    # Construct the command
    # Note: We set sample_start=0, sample_end=1 to run just our one exported file
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", script_path,
        "--gpu", device_num,
        "--seed", str(SEED),
        "--sample_start", "0",
        "--sample_end", "1",
        "--inference_R", str(inference_R),
        "--inference_snr", "32dB", # Just for folder naming
        "--num_steps", "500",
        "--S_churn", "0",
        "--measurements_path", meas_path,
        "--ksp_path", ksp_path,
        "--network", model_path,
        "--outdir", out_dir,
        "--snr", str(SNR)
    ]

# Run it
process = subprocess.run(cmd, capture_output=True, text=True)

if process.returncode != 0:
    print("❌ DPS Script Failed!")
    print(process.stderr)
else:
    print("✅ DPS Script Finished Successfully.")
    # print(process.stdout) # Uncomment to see full log

# -------------------------------------------------------------------------
# 4. LOAD & POST-PROCESS RESULTS
# -------------------------------------------------------------------------
# The result will be in outdir/R=.../snr.../seed.../sample_0.pt
# We need to find it dynamically because 'snr' folder name depends on arguments
result_file = None
for root, dirs, files in os.walk(out_dir):
    for file in files:
        if file == "sample_0.pt":
            result_file = os.path.join(root, file)
            break

if result_file and os.path.exists(result_file):
    print(f"Loading result from: {result_file}")
    dps_data = torch.load(result_file, map_location=device)
    
    # Extract Recon (It is padded 384x320)
    recon_padded = dps_data['recon']
    stop_dc_iter = dps_data['stop_dc_iter']
    if isinstance(recon_padded, torch.Tensor):
        recon_padded = recon_padded.detach().cpu().numpy()
    
    # Squeeze dimensions
    recon_padded = recon_padded.squeeze() # (384, 320) or (320, 384)
    
    # --- Crop back to Original (172x108) ---
    # Using the pad indices calculated in Step 2
    recon_crop = recon_padded
    
    # --- Intensity Normalization ---
    # Match intensity to target for fair comparison
    im_out_dps = recon_crop * norm_val
    
    print(f"Loaded DPS Image. Shape: {im_out_dps.shape}")
    
    # Assign for plotting block
    cplx_image_out_dps = im_out_dps 

else:
    print("❌ Could not find output file. Check process.stderr above.")
    cplx_image_out_dps = np.zeros_like(target)
print(f"Inference Complete. Output shape: {im_out_dps.shape}") 
print(f"Stopped DC Iteration at: {stop_dc_iter}")


# %%
# --- DEBUG PLOT: Mask and Intensity Comparison ---
import matplotlib.pyplot as plt
import sigpy as sp
import numpy as np

# 1. Prepare Mask
if isinstance(mask_real, torch.Tensor):
    mask_disp = mask_real.squeeze().cpu().numpy()
else:
    mask_disp = mask_real

# 2. Compute Zero-Filled Input (Reference for intensity)
if isinstance(ksp_cplx, torch.Tensor):
    ksp_input = ksp_cplx.squeeze().cpu().numpy()
else:
    ksp_input = ksp_cplx

# Perform IFFT on masked k-space
zf_input = sp.ifft(ksp_input * mask_disp)
input_mag = np.abs(zf_input)

# 3. Get DPS Result
dps_mag = np.abs(cplx_image_out_dps)

# 4. Concatenate for Intensity Check
# Stack them horizontally: [Zero-Filled | DPS Output]
comparison_img = np.concatenate((input_mag, dps_mag), axis=1)

# Calculate global max for shared scaling
global_vmax = np.percentile(comparison_img, 99)
if global_vmax == 0: global_vmax = 1.0


# %% [markdown]
# # DPS prior informed

# %%
# -------------------------------------------------------------------------
# 2. EXPORT DATA 
# -------------------------------------------------------------------------
ACS_size = 24
def forward_fs(img, mps):
    coil_imgs = img[None, ...] * mps
    ksp = sp.fft(coil_imgs, axes=(-2, -1))
    return ksp

def adjoint_fs(ksp, mps):
    img_space = sp.ifft(ksp, axes=(-2, -1))
    img = np.sum(img_space * np.conj(mps), axis=0)
    return img

def normalization_const(ksp, mps, gt_shape):                   
    C, H, W = ksp.shape
    mask = np.zeros((H, W), dtype=np.float32)
    center_h, center_w = H // 2, W // 2
    half_acs = ACS_size // 2
    mask[center_h - half_acs : center_h + half_acs, 
         center_w - half_acs : center_w + half_acs] = 1.0
    ksp_acs = ksp * mask[None, ...]
    acs_img = adjoint_fs(ksp_acs, mps)
    norm_const_99 = np.percentile(np.abs(acs_img), 99)
    return norm_const_99

inference_R = int(factor)
target_dims = (172, 108)
print(f"Exporting data for DPS_PI script (R={inference_R})...")
# Temporary folder to exchange data between this notebook and the script
temp_root = "../temp_dps_PI_compare_io"
meas_path = os.path.join(temp_root, "measurements")
ksp_path = os.path.join(temp_root, "kspace")
out_dir = os.path.join(temp_root, "results")

os.makedirs(meas_path, exist_ok=True)
os.makedirs(ksp_path, exist_ok=True)
#ksp_prior = cplx.to_tensor(sp.fft(cplx.to_numpy(reference_torch), axes=(-2, -1)))  
#kspace_torch = cplx.to_tensor(sp.fft(cplx.to_numpy(input_torch), axes=(-2, -1))) 
# kspace_torch_noised_full 

# A. Prepare K-Space
if kspace_torch.ndim == 3 and kspace_torch.shape[-1] == 2:
    ksp_cplx = torch.view_as_complex(kspace_torch.cpu()) 
    ksp_reference_cplx = torch.view_as_complex(reference_DPS_PI_torch_kspace.cpu())
else:
    ksp_cplx = kspace_torch.cpu()
    ksp_reference_cplx = reference_DPS_PI_torch_kspace.cpu()

prior_img = T.ifft2(reference_DPS_PI_torch_kspace).cpu()
prior_img = prior_img.unsqueeze(0)
prior_img = prior_img.permute(0,3,1,2)
ksp_padded = ksp_cplx
ksp_reference_padded = ksp_reference_cplx

# Save with shape [1, 1, 384, 320] (Coils, H, W)
ksp_export = ksp_padded.unsqueeze(0)
ksp_reference_export = ksp_reference_padded.unsqueeze(0)

# B. Prepare Maps (Ones)
s_map_export = torch.ones_like(ksp_export)

# C. Prepare Mask

if mask_type == '1D':
    mask_real = mask_torch_1d.cpu().repeat(172,1,1).squeeze(-1) 
else:
    mask_real = mask_torch_pois[:,:,0]

mask_padded = torch.zeros(target_dims, dtype=torch.float32)
mask_padded = mask_real.float()

# Save with shape [1, 1, 384, 320]
mask_export = mask_padded.unsqueeze(0)

# D. Prepare Ground Truth
gt_cplx = torch.view_as_complex(target_torch.cpu())
gt_padded = gt_cplx

# Calculate normalization constant from zero-filled like in the training process
#zf_ref = sp.ifft(ksp_cplx.numpy() * mask_real.numpy())
#norm_val = np.percentile(np.abs(zf_ref), 99)
#if norm_val == 0: norm_val = 1.0
#norm_val = normalization_const(ksp_export.numpy(), s_map_export.numpy(), target_dims)
ksp_export = ksp_padded / norm_val
ksp_export = ksp_export.unsqueeze(0)
ksp_reference_export = ksp_reference_padded / norm_val
ksp_reference_export = ksp_reference_export.unsqueeze(0)
prior_img = prior_img / norm_val

print(f'shape of prior img: {prior_img.shape}')
# ---------------------------------------------------------------------
# D. NOTEBOOK VISUALIZATION [UPDATED WITH ZF INPUT]
# ---------------------------------------------------------------------
print("Generating comparison plot in notebook...")

# 1. Prepare Target
gt_cplx = torch.view_as_complex(target_torch.cpu())
img_target_vis = np.abs(gt_cplx.numpy()) / norm_val
maps_in = s_map_export.numpy()
# 2. Prepare Prior (Full Res)
# Ensure we handle dimensions correctly for adjoint_fs (expecting numpy)
if ksp_reference_export.ndim == 4: # [1, 1, H, W]
    ksp_prior_in = ksp_reference_export[0].numpy()
else: # [H, W] or [1, H, W]
    ksp_prior_in = ksp_reference_export.numpy()

img_prior_vis_complex = adjoint_fs(ksp_prior_in, maps_in)
img_prior_vis = np.abs(img_prior_vis_complex)

# 3. [NEW] Prepare Zero-Filled Input (The Actual Input)
# Apply mask to the k-space
# Note: ksp_export is [H, W], mask_export is [1, 1, H, W]
# We need to ensure dimensions match for multiplication
ksp_data_np = ksp_export.numpy()
mask_np = mask_export.squeeze().numpy() # [H, W]

# Apply Mask
ksp_masked_np = ksp_data_np * mask_np

# Reconstruct
img_zf_vis_complex = adjoint_fs(ksp_masked_np, maps_in)
img_zf_vis = np.abs(img_zf_vis_complex)


# 4. Plot (Now 3 Columns)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Target Plot
im1 = axes[0].imshow(img_target_vis, cmap='gray')
axes[0].set_title(f"Target GT\nMax: {img_target_vis.max():.4f}")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].axis('off')

# Zero-Filled Plot (New)
im2 = axes[1].imshow(img_zf_vis, cmap='gray')
axes[1].set_title(f"Zero-Filled Input (Masked)\nMax: {img_zf_vis.max():.4f}")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
axes[1].axis('off')

# Prior Plot
im3 = axes[2].imshow(img_prior_vis, cmap='gray')
axes[2].set_title(f"Prior Recon\nMax: {img_prior_vis.max():.4f}")
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"Exported k-space shape: {ksp_export.shape}")
print(f"Exported mask shape: {mask_export.shape}")
print(f"ksp_reference_export shape: {ksp_reference_export.shape}")
print(f"maps_export shape: {s_map_export.shape}")
print(f"gt_padded shape: {gt_padded.shape}")

# Save to disk
torch.save({'ksp': ksp_export, 's_map': s_map_export,'prior_img' : prior_img , 'prior_ksp': ksp_reference_export}, os.path.join(ksp_path, "sample_0.pt"))
torch.save({'gt': gt_padded, f'mask_{inference_R}': mask_export}, os.path.join(meas_path, "sample_0.pt"))


## Run Prior-DPS via External Script
import subprocess
# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------
# Path to the root of the EDM repo
repo_path = "../../diffusion_mri/EDM"
if DPS_concat == 1:
    script_path = os.path.join(repo_path, "dps_PI_concat.py")
    model_path = "../../diffusion_mri/models/edm/brain/32dB_PI/00020-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"

else:
    script_path = os.path.join(repo_path, "PI_dps.py")
    model_path = "../../diffusion_mri/models/edm/brain/32dB/00019-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"


target_dims = (172, 108) # Network resolution
inference_R = int(factor) # e.g. 6

# -------------------------------------------------------------------------
# 2. EXECUTE EXTERNAL SCRIPT
# -------------------------------------------------------------------------
print("Running PI_dps.py subprocess...")

if DPS_ES == 0:
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", script_path,
        "--gpu", device_num,
        "--seed", str(SEED),
        "--sample_start", "0",
        "--sample_end", "1",
        "--inference_R", str(inference_R),
        "--inference_snr", "32dB", # Just for folder naming
        "--num_steps", "500",
        "--S_churn", "0",
        "--measurements_path", meas_path,
        "--ksp_path", ksp_path,
        "--network", model_path,
        "--outdir", out_dir,
        "--snr", str(SNR+100)
    ]
else:
    # Construct the command
    # Note: We set sample_start=0, sample_end=1 to run just our one exported file
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1", script_path,
        "--gpu", device_num,
        "--seed", str(SEED),
        "--latent_seeds", str(SEED),
        "--sample_start", "0",
        "--sample_end", "1",
        "--inference_R", str(inference_R),
        "--inference_snr", "32dB", # Just for folder naming
        "--num_steps", "500",
        "--S_churn", "0",
        "--measurements_path", meas_path,
        "--ksp_path", ksp_path,
        "--network", model_path,
        "--outdir", out_dir,
        "--snr", str(SNR)
    ]

# Run it
process = subprocess.run(cmd, capture_output=True, text=True)

if process.returncode != 0:
    print("❌ DPS PI Script Failed!")
    print(process.stderr)
    print("\n--- DEBUG PRINTS (STDOUT) ---")
    print(process.stdout)
else:
    print("✅ DPS PI Script Finished Successfully.")
    # print(process.stdout) # Uncomment to see full log

# -------------------------------------------------------------------------
# 3. LOAD & POST-PROCESS RESULTS
# -------------------------------------------------------------------------
# The result will be in outdir/R=.../snr.../seed.../sample_0.pt
# We need to find it dynamically because 'snr' folder name depends on arguments
result_file = None
for root, dirs, files in os.walk(out_dir):
    for file in files:
        if file == "sample_0.pt":
            result_file = os.path.join(root, file)
            break

if result_file and os.path.exists(result_file):
    print(f"Loading result from: {result_file}")
    dps_data = torch.load(result_file, map_location=device)
    
    # Extract Recon (It is padded 384x320)
    recon_padded = dps_data['recon']
    stop_dc_iter = dps_data['stop_dc_iter']
    if isinstance(recon_padded, torch.Tensor):
        recon_padded = recon_padded.detach().cpu().numpy()
    
    # Squeeze dimensions
    recon_padded = recon_padded.squeeze() # (384, 320) or (320, 384)

    # --- Crop back to Original (172x108) ---
    # Using the pad indices calculated in Step 2
    recon_crop = recon_padded
    
    # --- Intensity Normalization ---
    # Match intensity to target for fair comparison
    im_out_dps_pi = recon_crop * norm_val
    
    print(f"Loaded DPS Image. Shape: {im_out_dps_pi.shape}")
    
    # Assign for plotting block
    print(f'im_out_dps_pi shape before channel split: {im_out_dps_pi.shape}')
    cplx_image_out_dps_pi = im_out_dps_pi[:,:] 

else:
    print("❌ Could not find output file. Check process.stderr above.")
    cplx_image_out_dps_pi = np.zeros_like(target)

print(f"Stopped DC Iteration at: {stop_dc_iter}")



# 1. Prepare Mask
if isinstance(mask_real, torch.Tensor):
    mask_disp = mask_real.squeeze().cpu().numpy()
else:
    mask_disp = mask_real

# 2. Compute Zero-Filled Input (Reference for intensity)
if isinstance(ksp_cplx, torch.Tensor):
    ksp_input = ksp_cplx.squeeze().cpu().numpy()
else:
    ksp_input = ksp_cplx

# Perform IFFT on masked k-space
zf_input = sp.ifft(ksp_input * mask_disp)
input_mag = np.abs(zf_input)

# 3. Get DPS Result
dps_PI_mag = np.abs(cplx_image_out_dps_pi)

# 4. Concatenate for Intensity Check
# Stack them horizontally: [Zero-Filled | DPS Output]
comparison_img = np.concatenate((input_mag, dps_PI_mag), axis=1)

# Calculate global max for shared scaling
global_vmax = np.percentile(comparison_img, 99)
if global_vmax == 0: global_vmax = 1.0


"""
# --- 1. Save DPS PI Image ---
# Use 'cmap' to specify it is grayscale
plt.imsave(
    fname=os.path.join("./assets", "sample_0_DPS_PI.png"), 
    arr=dps_PI_mag, 
    cmap='gray'
)

# --- 2. Save ViT Fuser Image ---
# Ensure you are saving the magnitude (np.abs) if the data is complex
plt.imsave(
    fname=os.path.join("./assets", "sample_0_ViT_Fuser.png"), 
    arr=np.abs(im_out_vitfuser1), 
    cmap='gray'
)
"""

# Plot results
def compute_psnr(img1, img2, maxval):
    """Computes PSNR in dB"""
    mse = np.mean((img1 - img2) ** 2)

    return (10 * np.log10(maxval / mse)).item()
cplx_image_target = target
cplx_image_in = img_padded_np
cplx_image_out_vit1 = im_out_vit1
cplx_image_out_vit2 = im_out_vit2
cplx_image_out_vit3 = im_out_vit3
cplx_image_out_vitfuser1 = im_out_vitfuser1
cplx_image_out_vitfuser2 = im_out_vitfuser2
cplx_image_out_vitfuser3 = im_out_vitfuser3
cplx_image_out_MoDL = im_out_MoDL

cplx_image_reference = ref_np
maxval = np.max(np.abs(np.concatenate((cplx_image_target, cplx_image_in, cplx_image_out_vit1, cplx_image_out_vitfuser1, cplx_image_out_MoDL, cplx_image_out_dps), axis=0)))
minval = np.min(np.abs(np.concatenate((cplx_image_target,cplx_image_in,cplx_image_out_vit1,cplx_image_out_vit2,cplx_image_out_vitfuser1,cplx_image_out_vitfuser2,cplx_image_out_MoDL),axis=0)))


plt_concat = np.concatenate((np.abs(cplx_image_reference),np.abs(cplx_image_in),np.abs(cplx_image_out_MoDL),np.abs(cplx_image_out_dps), np.abs(cplx_image_out_dps_pi),np.abs(cplx_image_out_vit1),np.abs(cplx_image_out_vit2),np.abs(cplx_image_out_vitfuser1),np.abs(cplx_image_target)),axis=1)
fig, axs = plt.subplots(1, 1, figsize=(30, 5))  
im = axs.imshow(plt_concat, cmap='gray')
axs.set_title(f'          Reference                                     Input-ZF                                     MoDL                               EDM (DPS)                        EDM PI                      ViT MSE                            ViT Hybrid-loss                      ViT-Fuser Hybrid-loss                            Target')
fig.colorbar(im, ax=axs)
plt.axis('off')
plt.show()



# Plot data(SNR)
SNRs = [0,3,5,10,15,20]
# For plot
target_LR = cplx.to_numpy(T.ifft2( T.kspace_crop(T.fft2( cplx.to_tensor(target)),0.67)))
target_LR = cplx.to_numpy(T.ifft2( T.fft2( cplx.to_tensor(target))))
# Convert everything from numpy arrays to tensors
kspace_torch = cplx.to_tensor(kspace).float()

# Initialize list to store images
out = []

for snr in SNRs:
    # Add Gaussian noise to k-space
    kspace_noised = T.awgn_torch(kspace_torch.clone(), snr, L=1)
    # Apply k-space cropping
    kspace_noised = T.kspace_cut(kspace_noised, 0.67, 0.67)
    # Perform inverse FFT to get the scan
    scan_noised = T.ifft2(kspace_noised)
    # Convert to numpy
    scan_noised_np = cplx.to_numpy(scan_noised)
    # Append to output list
    out.append(scan_noised_np)

# Concatenate images side-by-side
out_concat = np.concatenate(np.abs(out), axis=1)

# Plot the result with SNR labels
fig, ax = plt.subplots(figsize=(30, 5))
ax.imshow(np.sqrt(out_concat[40:-10,:]**1.5), cmap="gray")
ax.axis("off")

# Add SNR labels above each image
image_width = out[0].shape[1]  # Width of each image
for idx, snr in enumerate(SNRs):
    x_position = idx * image_width + image_width / 2  # Center of each image
    ax.text(
        x_position, -10, f"SNR: {snr}[dB]", fontsize=12, color="black", ha="center", va="bottom"
    )

plt.show()



