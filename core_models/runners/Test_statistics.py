"""
All rights reserved.
Tal Oved (2026).
"""

import os, sys
# Add the path to the first folder
folder1_path = '../'
sys.path.append(folder1_path)
import logging
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
import sigpy.mri as mr
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_fill_holes
from torch.utils.data import DataLoader
# import custom libraries
from utils import transforms as T
from utils import subsample as ss
import subprocess
from utils import complex_utils as cplx
# import custom classes
from utils.datasets import SliceData
from utils.subsample_fastmri import MaskFunc
import matplotlib.pyplot as plt

import scipy.ndimage
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from models.ViTFuserWrap import ViTFuserWrap
from models.ViTWrap import ViTWrap
from models.MoDL_single import UnrolledModel
from fastmri.data import transforms, subsample
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


### Usage example:
# python Test_statistics_revision_python.py --r 3 --snr 10 --model modl --device_num 0 --mask 1D
# python3 Test_statistics_revision_python.py --r 3 --snr 5 --model DPS_PI --device_num 0 --mask 1D

import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="MRI Reconstruction Script")

# -- Define Arguments --
parser.add_argument('--r', type=int, default=3, help='Acceleration factor (e.g., 3)')
parser.add_argument('--snr', type=int, default=10, help='Signal-to-Noise Ratio in dB (e.g., 10)')
parser.add_argument('--model', type=str, default='modl', 
                    choices=['modl', 'ViT', 'ViTFuser', 'DPS', 'DPS_PI'],
                    help='Model type')
parser.add_argument('--device_num', type=int, default=0, help='GPU device index (e.g., 3)')
parser.add_argument('--mask', type=str, default='1D', 
                    choices=['1D', 'Poisson'],
                    help='Mask type')

# Parse arguments
args = parser.parse_args()

# -- Assign to your existing variables --
acc_factor = args.r
SNR = args.snr
model_type = args.model
mask_type = args.mask
DPS_ES = 1
dps_concat = 0
name_factor = acc_factor
if acc_factor == 6 or acc_factor == 10 or acc_factor == 12:
    acc_factor = acc_factor + 0.1
# Set Device
device_str = f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

# tmp folder for DPS I/O
run_id = f"{model_type}_SNR{SNR}_R{name_factor}_{mask_type}"
if model_type == 'DPS' or model_type == 'DPS_PI':
    # Create a temporary directory for DPS I/O
    temp_root = f"../temp_dps_io/temp_dps_io_{run_id}"
    os.makedirs(temp_root, exist_ok=True)

### CMMD folders
output_dir = f'../cmmd-pytorch/generated_images_{run_id}'
target_dir = f'../cmmd-pytorch/reference_images_{run_id}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

print(f"Configuration: Model={model_type}, R={name_factor}, SNR={SNR}dB, Mask={mask_type}, Device={device}")
# # Select model and checkpoints

##### Select Checkpoints
#checkpoint_file = "../checkpoints_forbenius_20dB/model_ViTFuser_100.pt" 
checkpoint_file = "../checkpoints_10dB_2dvar_r2_paper/model_MoDLTestsSSIM_100.pt" 

# simulation selctions

SEED = 0
### Select acceleration mask down  ##

# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


### Set params
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Hyperparameters
params = Namespace()
params.data_path = "../test_data/patient29b"
params.batch_size = 1
if model_type == 'modl':
    params.num_grad_steps = 4
else:
    params.num_grad_steps = 1
if model_type == 'modl' and acc_factor == 2:
    params.num_cg_steps = 8
else:
    params.num_cg_steps = 8
    
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.00001
params.weight_decay = 0
params.lr_step_size = 10
params.lr_gamma = 0.5
params.epoch = 21
params.reference_mode = 0
params.reference_lambda = 0.1
params.snr = SNR
params.factor = acc_factor
params.mask_type = mask_type

# Test cp
if model_type != 'DPS' and model_type != 'DPS_PI':
    checkpoint = torch.load(checkpoint_file,map_location=device)


# Model selection
# %%
## In case of special params take from cp
#params = checkpoint["params"]

if model_type == 'modl':
    model = UnrolledModel(params).to(device)
    model.load_state_dict(checkpoint['model'])
elif model_type == 'ViT':
    model = ViTWrap(params).to(device) #Use this for regular ViT
    model.load_state_dict(checkpoint['model'])
elif model_type == 'ViTFuser':
    model = ViTFuserWrap(params).to(device) #Use this for ViT-Fuser
    model.load_state_dict(checkpoint['model'])


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
        center_fractions = 0.08 * 4/factor  # RandomMaskFunc EquiSpacedMaskFunc
        mask_func = subsample.EquiSpacedMaskFunc(
        center_fractions=[center_fractions],
        accelerations=[factor], 
        )
        return mask_func
    def __call__(self, kspace, target, reference_kspace, reference,slice):
        im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(256,24)),(256,160))))
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        kspace = kspace/scale
        target = target/scale
        # Convert everything from numpy arrays to tensors
        kspace_torch = cplx.to_tensor(kspace).float()   
        target_torch = cplx.to_tensor(target).float() 
        target_torch = T.ifft2(T.kspace_cut(T.fft2(target_torch),0.67,0.67))   
        if self.mask_type == 'Poisson':
            # Use poisson mask instead
            mask2 = mr.poisson((172,108),self.factor, calib=(18,14), dtype=float, crop_corner=False, return_density=True, seed=0, max_attempts=6, tol=0.01)
            mask2[86-10:86+9,54-8:54+7] = 1
            mask_torch_pois = torch.stack([torch.tensor(mask2).float(),torch.tensor(mask2).float()],dim=2)
            kspace_torch = T.awgn_torch(kspace_torch,self.snr,L=1) 
            kspace_torch = T.kspace_cut(kspace_torch,0.67,0.67)
            kspace_torch = kspace_torch*mask_torch_pois
        else:
            ## Masking 1d
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
        if model_type != 'DPS_PI':
            reference = reference / scale_ref
        else:
            reference = reference / scale
        reference_torch = cplx.to_tensor(reference).float()
        reference_torch_kspace = T.fft2(reference_torch)
        reference_torch_kspace = T.kspace_cut(reference_torch_kspace,0.67,0.67)
        reference_torch = T.ifft2(reference_torch_kspace)

        return kspace_torch,target_torch,mask_torch, reference_torch


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


# Loop over test data

def expand_to_rgb(image):
    """Convert a 2D grayscale image to 3D RGB with zeros in the other two channels."""
    return np.stack([image, np.zeros_like(image), np.zeros_like(image)], axis=-1)


# Load test data
test_loader = create_data_loaders(params)
slice = 1

# Initialize lists to store metrics
mse_in_list, mse_out_list = [], []
psnr_in_list, psnr_out_list = [], []
ssim_in_list, ssim_out_list = [], []
lpips_out_list = []
recon_sum, inp_sum, tar_sum = torch.zeros((1,3,172,108)), torch.zeros((1,3,172,108)), torch.zeros((1,3,172,108))
lpips_model = lpips.LPIPS(net='alex').to(device)

fid_in = FrechetInceptionDistance(feature=64,normalize=True,input_img_size=(3, 172, 108))
fid_out = FrechetInceptionDistance(feature=64,normalize=True,input_img_size=(3, 172, 108))
if model_type != 'DPS' and model_type != 'DPS_PI':
    model.eval()  # Set model to evaluation mode

for i in range(3):
    with torch.no_grad():  # Disable gradient computation for evaluation
        for data in test_loader:
            input, target, mask, reference = data
            input = input.to(device)
            reference = reference.to(device)
            mask = mask.to(device)
            # Forward pass through the model
            if model_type != 'DPS' and model_type != 'DPS_PI':
                output = model(input.float(),reference_image=reference, mask=mask)
            else: # Running DPS reconstruction
                # -------------------------------------------------------------------------
                # CONFIGURATION
                # -------------------------------------------------------------------------
                # Path to the root of the EDM repo
                repo_path = "../../diffusion_mri/EDM"
                if model_type == 'DPS':
                    script_path = os.path.join(repo_path, "dps.py")
                else: # DPS-PI
                    if dps_concat:
                        script_path = os.path.join(repo_path, "dps_PI_concat.py")
                    else:
                        script_path = os.path.join(repo_path, "PI_dps.py")

                # Temporary folder to exchange data between this notebook and the script
                meas_path = os.path.join(temp_root, "measurements")
                ksp_path = os.path.join(temp_root, "kspace")
                out_dir = os.path.join(temp_root, "results")

                os.makedirs(meas_path, exist_ok=True)
                os.makedirs(ksp_path, exist_ok=True)

                # Model Settings
                if model_type == 'DPS' or model_type == 'DPS_PI' and dps_concat == 0:
                    model_path = "../../diffusion_mri/models/edm/brain/32dB/00019-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"
                else: # DPS-PI
                    model_path = "../../diffusion_mri/models/edm/brain/32dB_PI/00020-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"
                # -------------------------------------------------------------------------
                # EXPORT DATA 
                # -------------------------------------------------------------------------
                kspace_torch = input.squeeze(0) 
                kspace_reference_torch = T.fft2(reference.squeeze(0))
                target_torch = target.squeeze(0)
                mask_torch = mask[:,:,:,0].squeeze(0)
                target_dims = (172, 108)
                inference_R = name_factor
                # A. Prepare K-Space
                if kspace_torch.ndim == 3 and kspace_torch.shape[-1] == 2:
                    ksp_cplx = torch.view_as_complex(kspace_torch.cpu())
                    ksp_reference_cplx = torch.view_as_complex(kspace_reference_torch.cpu())
                else:
                    ksp_cplx = kspace_torch.cpu()
                    ksp_reference_cplx = kspace_reference_torch.cpu()


                ksp_padded = ksp_cplx
                ksp_reference_padded = ksp_reference_cplx

                # Save with shape [1, 1, 384, 320] (Coils, H, W)
                ksp_export = ksp_padded.unsqueeze(0)
                ksp_reference_export = ksp_reference_padded.unsqueeze(0)

                # B. Prepare Maps (Ones)
                s_map_export = torch.ones_like(ksp_export)

                # C. Prepare Mask
                mask_real = mask_torch.cpu() 

                mask_padded = torch.zeros(target_dims, dtype=torch.float32)
                mask_padded = mask_real.float()

                # Save with shape [1, 1, 384, 320]
                mask_export = mask_padded.unsqueeze(0).unsqueeze(0)

                # D. Prepare Ground Truth
                gt_cplx = torch.view_as_complex(target_torch.cpu())
                gt_padded = torch.zeros(target_dims, dtype=torch.complex64)
                gt_padded = gt_cplx

                # Calculate normalization constant from zero-filled like in the training process
                zf_ref = sp.ifft(ksp_cplx.numpy() * mask_real.numpy())
                norm_val = np.percentile(np.abs(zf_ref), 99)
                if norm_val == 0: norm_val = 1.0
                ksp_export = ksp_padded / norm_val
                ksp_reference_export = ksp_reference_padded / norm_val

                # Save to disk
                torch.save({'ksp': ksp_export, 's_map': s_map_export, 'prior_ksp': ksp_reference_export}, os.path.join(ksp_path, "sample_0.pt"))
                torch.save({'gt': gt_padded, f'mask_{inference_R}': mask_export}, os.path.join(meas_path, "sample_0.pt"))                                       
                        

                # -------------------------------------------------------------------------
                #  EXECUTE EXTERNAL SCRIPT
                # -------------------------------------------------------------------------
                #print("Running dps.py subprocess...")

                # Construct the command
                # Note: We set sample_start=0, sample_end=1 to run just our one exported file
                if not DPS_ES and model_type == 'DPS_PI':
                    SNR = SNR + 100
                cmd = [
                    "torchrun", "--standalone", "--nproc_per_node=1", script_path,
                    "--gpu", "0",
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
                    k=5
                    #print("✅ DPS Script Finished Successfully.")
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
                    #print(f"Loading result from: {result_file}")
                    dps_data = torch.load(result_file, map_location=device)
                    
                    # Extract Recon (It is padded 384x320)
                    recon_padded = dps_data['recon']
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
                    
                    #print(f"Loaded DPS Image. Shape: {im_out_dps.shape}")
                    
                    # Assign for plotting block
                    cplx_image_out_dps = im_out_dps 
                    output = cplx.to_tensor(cplx_image_out_dps).unsqueeze(0)
                    
                else:
                    print("❌ Could not find output file. Check process.stderr above.")
                    cplx_image_out_dps = np.zeros_like(target)
                    output = cplx.to_tensor(cplx_image_out_dps)


            # Handle output
            kspace_out = T.fft2(output.cpu().squeeze(0))
            im_out = T.ifft2(kspace_out)
        
            kspace_target = T.fft2(target.cpu().squeeze(0))
            target = T.ifft2(kspace_target)

        
            #print(cplx.to_numpy(im_out.cpu()).shape)
            cplx_image_target = cplx.to_numpy(target.cpu())
            cplx_image_in = cplx.to_numpy(T.ifft2(input.cpu())).squeeze(0)
            cplx_image_out = cplx.to_numpy(im_out.cpu().squeeze(0))
            cplx_image_reference = cplx.to_numpy(reference.cpu()).squeeze(0)
    

            target_numpy = cplx.to_numpy(target.cpu())
            input_numpy = cplx.to_numpy(T.ifft2(input.cpu())).squeeze(0)
            out_numpy = cplx.to_numpy(output.cpu()).squeeze(0)
            max_val = np.max(np.abs(np.concatenate((target_numpy,input_numpy,out_numpy),axis=0)))
            min_val = np.min(np.abs(np.concatenate((cplx_image_target,cplx_image_in,cplx_image_out),axis=0)))
            target_numpy_norm = np.abs(target_numpy)/max_val.squeeze(0)
            input_numpy_norm = np.abs(input_numpy)/max_val.squeeze(0)
            out_numpy_norm = np.abs(out_numpy)/max_val.squeeze(0)


            # Find comparison area:
            area = target_numpy_norm > 0.30
            kernel = np.ones((10, 10)) / 25.0
            #area = np.convolve(area, kernel, mode='constant', cval=0.0)
            area = scipy.ndimage.convolve(area.astype(float), kernel, mode='constant', cval=0.0)
            area[area>0.009] = 1
            structuring_element = np.ones((4,4))
            area = binary_closing(area, structure=structuring_element)
            area = binary_fill_holes(area)

            target_numpy_norm = target_numpy_norm * area
            input_numpy_norm = input_numpy_norm * area
            out_numpy_norm = out_numpy_norm * area
            
            # Save for CMMD calculation
            target_rgb = expand_to_rgb(target_numpy_norm)
            output_rgb = expand_to_rgb(out_numpy_norm)

            target_image_pil = Image.fromarray((target_rgb * 255).astype(np.uint8))
            output_image_pil = Image.fromarray((output_rgb * 255).astype(np.uint8))
            
            target_image_pil.save(os.path.join(target_dir, f'target_image_{slice}_1.png'))
            output_image_pil.save(os.path.join(output_dir, f'output_image_{slice}_2.png'))
        
            slice = slice + 1

            ## Calculate metrics
            # Calculate SSIM values
            data_range = max_val - min_val

            ssim_in, _ = ssim(target_numpy_norm, input_numpy_norm, data_range=data_range, full=True)
            ssim_out, _ = ssim(target_numpy_norm, out_numpy_norm, data_range=data_range, full=True)

            # Calculate PSNR
            psnr_in = T.PSNR_numpy(target_numpy_norm, input_numpy_norm)
            psnr_out = T.PSNR_numpy(target_numpy_norm, out_numpy_norm)

            # Calculate FID
            zeros_vec = torch.zeros((1,1,172,108))
            tar = torch.cat((cplx.to_tensor(np.abs(target_numpy_norm)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1)
            tar_sum = torch.cat((tar_sum,tar),dim=0)
            recon = torch.cat((cplx.to_tensor(np.abs(out_numpy_norm)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1)
            recon_sum = torch.cat((recon_sum,recon),dim=0)
            inp = torch.cat((cplx.to_tensor(np.abs(input_numpy_norm)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1) 
            inp_sum = torch.cat((inp_sum,inp),dim=0)       

            # Calculate MSE
            mse_in = np.mean(np.abs(input_numpy_norm-target_numpy_norm)**2)
            mse_out = np.mean(np.abs(out_numpy_norm-target_numpy_norm)**2)

            # LpiPS
            # Compute LPIPS
            # Convert to PyTorch tensors and move to GPU
            target_tensor = torch.tensor(target_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1  # Normalize to [-1, 1]
            out_tensor = torch.tensor(output_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1

            lpips_out = lpips_model(target_tensor, out_tensor).item()

            # Store LPIPS values
            lpips_out_list.append(lpips_out)

            # Append metrics to lists
            mse_in_list.append(mse_in)
            mse_out_list.append(mse_out)
            psnr_in_list.append(psnr_in)
            psnr_out_list.append(psnr_out)
            ssim_in_list.append(ssim_in)
            ssim_out_list.append(ssim_out)

            #break
        #break

# Sum results and create statistics
fid_in.update(tar_sum, real=True)
fid_out.update(tar_sum, real=True)
fid_out.update(recon_sum, real=False)
fid_recon = fid_out.compute()
fid_in.update(inp_sum, real=False)
fid_inp = fid_in.compute()

# Print average metrics
print(f'Average MSE input: {np.mean(mse_in_list):.4f}')
print(f'Average MSE output: {np.mean(mse_out_list):.4f} ± {np.std(mse_out_list):.4f}')
print(f'Average PSNR input: {np.mean(psnr_in_list):.4f}')
print(f'Average PSNR output: {np.mean(psnr_out_list):.4f}± {np.std(psnr_out_list):.4f}')
print(f'Average SSIM input: {np.mean(ssim_in_list):.4f}')
print(f'Average SSIM output: {np.mean(ssim_out_list):.4f} ± {np.std(ssim_out_list):.4f}')
print(f'Average FID input: {fid_inp:.4f}')
print(f'Average FID output: {fid_recon:.4f}')
print(f'Average LPIPS output: {np.mean(lpips_out_list):.4f} ± {np.std(lpips_out_list):.4f}')
print(f'Test slices: {len(test_loader)}')



# CMMD calculation
#%%bash
#cd ../cmmd-pytorch
#python3 main.py reference_images_DPS_PI_SNR20_R3_1D generated_images_DPS_PI_SNR20_R3_1D --batch_size=1


