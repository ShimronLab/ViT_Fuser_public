# %%
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
import scipy.io as sio
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
import utils.ut2 as ut
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
# Assuming these models exist in your path
from models.ViTFuserWrap import ViTFuserWrap
from models.ViTWrap import ViTWrap
from models.MoDL_single import UnrolledModel
from fastmri.data import transforms, subsample
import argparse
import shutil

#### running example:
# python3 Bias_Variance_Tradeoff.py --r 3 --snr 5 --model ViTFuser --device_num 0 --mask 1D --slice_idx 26 --num_variations 20 --subject 2
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- NEW ARGUMENTS FOR BIAS-VARIANCE --
parser = argparse.ArgumentParser(description="Bias-Variance Analysis Script")
parser.add_argument('--r', type=int, default=3, help='Acceleration factor')
parser.add_argument('--snr', type=int, default=10, help='SNR dB')
parser.add_argument('--model', type=str, default='DPS_PI', choices=['modl', 'ViT', 'ViTFuser', 'DPS', 'DPS_PI'])
parser.add_argument('--device_num', type=int, default=0)
parser.add_argument('--mask', type=str, default='1D')
# New args
parser.add_argument('--slice_idx', type=int, default=15, help='Index of the slice to analyze')
parser.add_argument('--num_variations', type=int, default=20, help='Number of different latent seeds to sample')
parser.add_argument('--subject', type=int, default=1, help='Subject')

args = parser.parse_args()

# Configuration
acc_factor = args.r
SNR = args.snr
model_type = args.model
mask_type = args.mask
slice_to_analyze = args.slice_idx
num_variations = args.num_variations
subject_num = args.subject

name_factor = acc_factor
if acc_factor in [6, 10, 12]:
    acc_factor = acc_factor + 0.1

acc_factor_name = round(name_factor)

device_str = f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

# Paths for IO
run_id = f"{model_type}_SNR{SNR}_R{name_factor}_{mask_type}"
temp_root = f"../temp_dps_io/temp_dps_io_{run_id}"
os.makedirs(temp_root, exist_ok=True)

# NEW OUTPUT FOLDER FOR VARIANCE ANALYSIS
analysis_dir = f'../bias_variance_analysis/{run_id}_slice{slice_to_analyze}'
os.makedirs(analysis_dir, exist_ok=True)
print(f"Saving {num_variations} variations to: {analysis_dir}")

# %% 
# Set Params & Checkpoints (Same as original)
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Hyperparameters (Subset needed for dataloader)
params = Namespace()
if subject_num == 1:
    params.data_path = "../test_data/patient29b"
if subject_num == 2:
    params.data_path = "../test_data/patient28"
if subject_num == 3:
    params.data_path = "../test_data/patient72b"
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

# Load Model (If not DPS)
if model_type not in ['DPS', 'DPS_PI'] and mask_type != '1D':
    if model_type == 'modl':
        checkpoint_file = "../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{acc_factor_name}" "_paper/model_MoDLTests_100.pt" #"../checkpoints_" + f"{SNR}"+ "dB_paper/model_MoDLTests_100.pt" # Update if needed
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = UnrolledModel(params).to(device)
        model.load_state_dict(checkpoint['model'])
    if model_type == 'ViT':
        checkpoint_file = "../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{acc_factor_name}" "_paper/model_ViTMSE_100.pt"#"../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViT_100.pt"  # Update if needed
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = ViTWrap(params).to(device)
        model.load_state_dict(checkpoint['model'])
    if model_type == 'ViTFuser':
        checkpoint_file ="../checkpoints_" + f"{SNR}"+ "dB_2dvar_r" +f"{acc_factor_name}" "_paper/model_ViTFuser_100.pt"# "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTFuser_100.pt" # Update if needed
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = ViTFuserWrap(params).to(device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
elif model_type not in ['DPS', 'DPS_PI'] and mask_type == '1D':
    if model_type == 'modl':
        checkpoint_file = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_MoDLTests_100.pt" 
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = UnrolledModel(params).to(device)
        model.load_state_dict(checkpoint['model'])
    if model_type == 'ViT':
        checkpoint_file = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTMSE_100.pt" 
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = ViTWrap(params).to(device)
        model.load_state_dict(checkpoint['model'])
    if model_type == 'ViTFuser':
        checkpoint_file = "../checkpoints_" + f"{SNR}"+ "dB_paper/model_ViTFuserMSE_100.pt" 
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = ViTFuserWrap(params).to(device)
        model.load_state_dict(checkpoint['model'])
    model.eval()   

def get_mask_func(factor):
        center_fractions = 0.08 * 4/factor
        mask_func = subsample.EquiSpacedMaskFunc(center_fractions=[center_fractions], accelerations=[factor])
        return mask_func
 
# %% Data Transformer (Same as original)
class DataTransform:
    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.snr = args.snr 
        self.mask_type = args.mask_type
        self.factor = args.factor
    
    def get_mask_func(self, factor):
        center_fractions = 0.08 * 4/factor
        mask_func = subsample.EquiSpacedMaskFunc(center_fractions=[center_fractions], accelerations=[factor])
        return mask_func

    def __call__(self, kspace, target, reference_kspace, reference, slice):
        # Scaling logic
        im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(256,24)),(256,160))))
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        kspace = kspace/scale
        target = target/scale
        
        kspace_torch = cplx.to_tensor(kspace).float()   
        target_torch = cplx.to_tensor(target).float() 
        target_torch = T.ifft2(T.kspace_cut(T.fft2(target_torch),0.67,0.67))   
        
        # Add Noise
        #kspace_torch = T.awgn_torch(kspace_torch, self.snr, L=1)
        
        # Apply Mask
        if self.mask_type == 'Poisson':
            mask2 = mr.poisson((172,108), self.factor, calib=(18,14), dtype=float, crop_corner=False, return_density=True, seed=0, max_attempts=6, tol=0.01)
            mask2[86-10:86+9,54-8:54+7] = 1
            mask_torch = torch.stack([torch.tensor(mask2).float(),torch.tensor(mask2).float()],dim=2)
            kspace_torch = T.kspace_cut(kspace_torch, 0.67, 0.67) * mask_torch
        else:
            kspace_torch = T.awgn_torch(kspace_torch,SNR,L=1)
            mask_func = self.get_mask_func(self.factor) 
            kspace_torch = T.kspace_cut(kspace_torch, 0.67, 0.67)
            kspace_torch = transforms.apply_mask(kspace_torch, mask_func)[0]
        
        mask = np.abs(cplx.to_numpy(kspace_torch))!=0
        mask_torch = torch.stack([torch.tensor(mask).float(),torch.tensor(mask).float()],dim=2)
        
        # Reference
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
        reference_torch_kspace = T.kspace_cut(reference_torch_kspace, 0.67, 0.67)
        reference_torch = T.ifft2(reference_torch_kspace)

        return kspace_torch, target_torch, mask_torch, reference_torch

def create_data_loaders(args):
    train_mask = MaskFunc([0.08],[4])
    train_data = SliceData(root=str(args.data_path), transform=DataTransform(train_mask, args), sample_rate=1)
    # Important: shuffle=False so we can find the specific slice index deterministically
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader

def expand_to_rgb(image):
    return np.stack([image, np.zeros_like(image), np.zeros_like(image)], axis=-1)

# %% [markdown]
# # BIAS-VARIANCE ANALYSIS LOOP
# %%
test_loader = create_data_loaders(params)

# List to collect all reconstructed images for PCA
all_recons_for_pca = []

# 2. RUN RECONSTRUCTION N TIMES
print(f"Starting {num_variations} variations...")
CURRENT_SEED_DPS = 42
for i in range(num_variations):
    # Vary the seed for this iteration
    CURRENT_SEED = i * 100 + 42 # Arbitrary deterministic variation
    print(f"--- Variation {i+1}/{num_variations} (Seed {CURRENT_SEED}) ---")
    # 1. FIND THE TARGET SLICE
    print(f"Extracting slice index {slice_to_analyze}...")
    target_data = None
    for i, data in enumerate(test_loader):
        if i == slice_to_analyze:
            target_data = data
            break

    if target_data is None:
        raise ValueError(f"Slice index {slice_to_analyze} out of range for dataset.")

    input_batch, target_batch, mask_batch, reference_batch = target_data
    input_batch = input_batch.to(device)
    reference_batch = reference_batch.to(device)
    mask_batch = mask_batch.to(device)

    # Save the Ground Truth Once
    target_numpy = cplx.to_numpy(target_batch.squeeze(0).cpu())
    target_norm = np.abs(target_numpy) / np.max(np.abs(target_numpy))
    Image.fromarray((np.clip(target_norm, 0, 1) * 255).astype(np.uint8)).save(os.path.join(analysis_dir, 'ground_truth.png'))

    # =========================================================================
    # 2. SELECT MODEL PATH
    # =========================================================================

    output_img = None
    
    # --- IF DPS / DPS_PI ---
    if model_type in ['DPS', 'DPS_PI']:
        # Prepare Paths
        repo_path = "../../diffusion_mri/EDM"
        script_name = "dps.py" if model_type == 'DPS' else "PI_dps.py"
        script_path = os.path.join(repo_path, script_name)
        
        meas_path = os.path.join(temp_root, "measurements")
        ksp_path = os.path.join(temp_root, "kspace")
        out_dir = os.path.join(temp_root, "results")
        os.makedirs(meas_path, exist_ok=True)
        os.makedirs(ksp_path, exist_ok=True)

        if model_type == 'DPS':
            model_path = "../../diffusion_mri/models/edm/brain/32dB/00019-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"
        else:
            model_path = "../../diffusion_mri/models/edm/brain/32dB/00019-samples-uncond-ddpmpp-edm-gpus4-batch128-fp32-container_test/network-snapshot-010000.pkl"

        # --- EXPORT DATA (Done inside loop, but data is FIXED) ---
        # -------------------------------------------------------------------------
        # EXPORT DATA 
        # -------------------------------------------------------------------------
        kspace_torch = input_batch.squeeze(0) 


        kspace_reference_torch = T.fft2(reference_batch.squeeze(0))
        target_torch = target_batch.squeeze(0)
        mask_torch = mask_batch[:,:,:,0].squeeze(0)
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
        gt_export = gt_padded / norm_val
        """
        # Note: We export every time just to be safe, but input_batch is constant
        kspace_torch = input_batch.squeeze(0) 
        kspace_reference_torch = T.fft2(reference_batch.squeeze(0))
        target_torch_tens = target_batch.squeeze(0)
        mask_torch_tens = mask_batch[:,:,:,0].squeeze(0)
        target_dims = (172, 108)

        # 1. Complex Conversion
        if kspace_torch.ndim == 3 and kspace_torch.shape[-1] == 2:
            ksp_cplx = torch.view_as_complex(kspace_torch.cpu())
            ksp_reference_cplx = torch.view_as_complex(kspace_reference_torch.cpu())
        else:
            ksp_cplx = kspace_torch.cpu()
            ksp_reference_cplx = kspace_reference_torch.cpu()

        # 2. HELPER FUNCTIONS REPLACED WITH TORCH LOGIC
        # We must use norm='ortho' to match the physics of the DPS script!
        
        # 3. Calculate Norm Constant (Robustly)
        # Use the Reference (Prior) for stable scaling, just like the working notebook.
        # ksp_reference_cplx is [172, 108]
        
        # Shift first (to match the 'ortho' check done in the notebook)
        ksp_shifted_temp = torch.fft.ifftshift(ksp_cplx, dim=(-2, -1))
        
        # Reconstruct using ORTHO norm
        temp_img = torch.fft.ifft2(ksp_shifted_temp, dim=(-2, -1), norm='ortho')
        vals = torch.abs(temp_img)
        norm_val = np.percentile(vals.numpy(), 99)

        if norm_val == 0: norm_val = 1.0
        print(f"Variation {i}: Norm Val (Ortho) = {norm_val:.4f}")

        # 4. Apply Shift (Center->Corner) and Norm
        ksp_shifted = torch.fft.ifftshift(ksp_cplx, dim=(-2, -1))
        prior_shifted = torch.fft.ifftshift(ksp_reference_cplx, dim=(-2, -1))
        mask_shifted = torch.fft.ifftshift(mask_torch_tens.cpu(), dim=(-2, -1))
        
        ksp_export = (ksp_shifted / norm_val).unsqueeze(0).unsqueeze(0)
        prior_export = (prior_shifted / norm_val).unsqueeze(0).unsqueeze(0)
        mask_export = mask_shifted.float().unsqueeze(0).unsqueeze(0)
        s_map_export = torch.ones_like(ksp_export)
        
        # GT Export (Just for script compatibility)
        gt_cplx = torch.view_as_complex(target_torch_tens.cpu())
        gt_export = gt_cplx / norm_val
        """
        # Save
        torch.save({'ksp': ksp_export, 's_map': s_map_export, 'prior_ksp': ksp_reference_export}, 
                   os.path.join(ksp_path, "sample_0.pt"))
        torch.save({'gt': gt_export, f'mask_{name_factor}': mask_export}, 
                   os.path.join(meas_path, "sample_0.pt"))

        # --- SUBPROCESS CALL ---
        # CLEANUP OLD RESULTS
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1", script_path,
            "--gpu", "0",
            "--seed", str(CURRENT_SEED_DPS),
            "--latent_seeds", str(CURRENT_SEED_DPS), # CRITICAL: Vary this to see variance!
            "--sample_start", "0",
            "--sample_end", "1",
            "--inference_R", str(name_factor),
            "--inference_snr", str(SNR),
            "--num_steps", "500",
            "--S_churn", "0",
            "--measurements_path", meas_path,
            "--ksp_path", ksp_path,
            "--network", model_path,
            "--outdir", out_dir,
             "--snr", str(SNR) # REMOVE to full diffuion DC on
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"❌ Variation {i} Failed.")
            print(process.stderr)
            continue
        
        # --- LOAD RESULT ---
        result_file = None
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                if file == "sample_0.pt":
                    result_file = os.path.join(root, file)
                    break
        
        if result_file:
            dps_data = torch.load(result_file, map_location=device)
            recon_padded = dps_data['recon']
            if isinstance(recon_padded, torch.Tensor):
                recon_padded = recon_padded.detach().cpu().numpy()
            
            # Post-process
            recon_img = recon_padded.squeeze() * norm_val
            output_img = cplx.to_tensor(recon_img).unsqueeze(0).to(device) # Keep on GPU for metric calc
        
    # --- IF STANDARD MODEL (MoDL, ViT) ---
    else:
        with torch.no_grad():
            output_img = model(input_batch.float(), reference_image=reference_batch, mask=mask_batch)

    # 3. CALCULATE METRICS & SAVE IMAGE
    if output_img is not None:
        # To Numpy
        if output_img.shape[-1] == 2: # Complex tensor
             kspace_out = T.fft2(output_img.cpu().squeeze(0))
             im_out = T.ifft2(kspace_out)
             out_numpy = cplx.to_numpy(im_out)
        else: # Real tensor/numpy
             out_numpy = cplx.to_numpy(output_img.cpu().squeeze(0))

        
        # 1. Get RAW Magnitude for Statistics
        out_mag_raw = np.abs(out_numpy)

        # 2. Create Normalized Version ONLY for PNG Visualization
        max_val = np.max(np.abs(target_numpy))
        if max_val == 0: max_val = 1.0
        out_vis = out_mag_raw / max_val
        
        # 2.5 mask the output to the brain region only
        area = out_vis > 0.30
        kernel = np.ones((10, 10)) / 25.0
        #area = np.convolve(area, kernel, mode='constant', cval=0.0)
        area = scipy.ndimage.convolve(area.astype(float), kernel, mode='constant', cval=0.0)
        area[area>0.009] = 1
        structuring_element = np.ones((4,4))
        area = binary_closing(area, structure=structuring_element)
        area = binary_fill_holes(area)
        out_mag_raw = out_mag_raw * area
        out_vis = out_vis * area

        # Save Image (Use out_vis)
        filename = f'recon_var_{i}_seed{CURRENT_SEED}.png'
        Image.fromarray((np.clip(out_vis, 0, 1) * 255).astype(np.uint8)).save(os.path.join(analysis_dir, filename))        
        
        # --- COLLECT FOR PCA ---
        # CRITICAL: Append the RAW magnitude, NOT the normalized visualization!
        all_recons_for_pca.append(out_mag_raw)
        
        # Calc Metrics (Use raw comparison)
        data_range = np.max(np.abs(target_numpy)) - np.min(np.abs(target_numpy))
        val_ssim = ssim(np.abs(target_numpy), out_mag_raw, data_range=data_range)
        val_psnr = T.PSNR_numpy(np.abs(target_numpy), out_mag_raw)
        
        print(f"   -> Saved. SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}")
    else:
        print("   -> No output generated.")

# -----------------------------------------------------------------------------
# BIAS-VARIANCE STATISTICS & PCA
# -----------------------------------------------------------------------------
print("Computing Bias-Variance Statistics...")

if len(all_recons_for_pca) > 1:
    # 1. Convert Stack to Numpy [N, H, W]
    stack = np.array(all_recons_for_pca)
    
    # 2. Compute Mean Reconstruction (E[x_hat])
    mean_recon = np.mean(stack, axis=0)
    
    # 3. Compute Standard Deviation Map
    std_map = np.std(stack, axis=0)
    
    # 4. Compute Bias Map
    if target_numpy.shape == mean_recon.shape:
        bias_map = np.abs(mean_recon - np.abs(target_numpy))
    else:
        bias_map = np.zeros_like(mean_recon)

    # -------------------------------------------------------------------------
    # COMPUTE PCA (Moved UP so we can save the values)
    # -------------------------------------------------------------------------
    print("Computing PCA...")
    N, H, W = stack.shape
    flat_data = stack.reshape(N, -1)
    
    # Center the data
    centered_data = flat_data - mean_recon.reshape(1, -1)
    
    # Initialize Singular Values (Default to zeros if deterministic)
    singular_values = np.zeros(min(N, flat_data.shape[1]))
    U, S, Vh = None, None, None

    if np.allclose(centered_data, 0):
        print("⚠️ ALERT: Zero Variance. PCA skipped.")
    else:
        # Perform SVD
        # S contains the singular values (related to eigenvalues)
        U, S, Vh = np.linalg.svd(centered_data, full_matrices=False)
        singular_values = S  # Store for saving

    # -------------------------------------------------------------------------
    # SAVE NUMERICAL RESULTS (Now includes 'singular_values')
    # -------------------------------------------------------------------------
    results_dict = {
        'mean_recon': mean_recon,
        'std_map': std_map,       
        'variance_map': std_map**2,
        'bias_map': bias_map,
        'ground_truth': target_numpy,
        'model_type': model_type,
        'snr': SNR,
        'acceleration': name_factor,
        'singular_values': singular_values # <--- ADDED THIS LINE
    }
    
    mat_filename = f'results_{model_type}_R{name_factor}_SNR{SNR}_sub{args.subject}_slice{args.slice_idx}.mat'
    mat_save_path = os.path.join(analysis_dir, mat_filename)
    
    sio.savemat(mat_save_path, results_dict)
    print(f"✅ Numerical data saved to: {mat_save_path}")

    # -------------------------------------------------------------------------
    # SAVE STATISTICAL MAPS (Images)
    # -------------------------------------------------------------------------
    # ... (Keep your existing image saving code here) ...
    Image.fromarray((np.clip(mean_recon / max_val, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(analysis_dir, 'stat_mean_reconstruction.png'))
    # ... etc ...

    # -------------------------------------------------------------------------
    # PLOT PCA IMAGES (Only if SVD was successful)
    # -------------------------------------------------------------------------
    if Vh is not None:
        # ... (Keep your existing PCA plotting code here) ...
        # Just ensure you use 'S' and 'Vh' calculated above
        num_pcs = min(5, N)
        fig, axes = plt.subplots(1, num_pcs, figsize=(4 * num_pcs, 5))
        if num_pcs == 1: axes = [axes]
        
        for i in range(num_pcs):
            pc_img = Vh[i].reshape(H, W)
            # ... rest of your plotting code ...
            singular_val = S[i]
            
            # Use 'bwr' (Blue-White-Red) for diverging positive/negative values
            limit = np.max(np.abs(pc_img))
            im = axes[i].imshow(pc_img, cmap='bwr', vmin=-limit, vmax=limit)
            
            # High Precision Title
            axes[i].set_title(f"PC {i+1}\nVal: {singular_val:.6e}")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
        plt.suptitle(f"Top {num_pcs} Principal Components (Variance Directions)\nModel: {model_type}, Slice: {slice_to_analyze}")
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'pca_analysis.png'), dpi=150)
        plt.close()
        print(f"✅ Analysis Saved to: {analysis_dir}")

else:
    print("❌ Not enough variations to compute statistics.")

print("Bias-Variance Analysis Complete.")

