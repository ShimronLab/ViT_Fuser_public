import os, sys
# Add the path to the first folder
folder1_path = '../'
sys.path.append(folder1_path)

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.ViTFuserWrap import ViTFuserWrap
from models.ViTWrap import ViTWrap
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
from fastmri.data import transforms, subsample
from utils import transforms as T
from utils import complex_utils as cplx
import torch.nn.functional as F
import sigpy as sp
import sigpy.mri as mr

class FeatureExtractor:
    """
    A simple class to store activations from a forward hook.
    """
    def __init__(self):
        self.features = []

    def hook_fn(self, module, input, output):
        # The 'output' is what we want. We detach it from the graph.
        self.features.append(output.detach())

    def clear(self):
        # Clear features for the next run
        self.features = []



#  ---- Model environment setup ----
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
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

checkpoint_file_vitfuser1 = "../checkpoints_" + f"15"+ "dB_paper/model_ViTFuser_100.pt"
checkpoint_vit1 = torch.load(checkpoint_file_vitfuser1,map_location=device)

# 1. Instantiate your model (assuming 'model' is an instance of ViTFuserWrap)
model = ViTFuserWrap(params).to(device)
model.load_state_dict(checkpoint_vit1['model'])
model.eval() 

## INPUT DATA PREPARATION 
# Choose patient, slice number, SNR[dB], mask and factor
patient = 1#3
slice_number = 24
mask_type = '2D'
#mask_type = '1D'
factor = 6.1
minus_factor = 0.1
SNR = 15 #[dB]
SEED = 2025
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
target = target.squeeze(0)
target_torch = cplx.to_tensor(target).float() 
reference_torch = cplx.to_tensor(reference).float() 
reference_kspace_torch = T.fft2(reference_torch)
reference_kspace = cplx.to_numpy(reference_kspace_torch)
kspace_torch = T.fft2(target_torch)
target = cplx.to_numpy(target_torch)
kspace = cplx.to_numpy(kspace_torch)
mask2 = mr.poisson((172,108),factor, calib=(18,14), dtype=float, crop_corner=False, return_density=True, seed=0, max_attempts=6, tol=0.01)
mask2[86-10:86+9,54-8:54+7] = 1
mask_torch_pois = torch.stack([torch.tensor(mask2).float(),torch.tensor(mask2).float()],dim=2)

## Print 1/acceleration  
s = (172)*(108)
print((torch.sum(mask_torch_pois))/s/2)

factor = np.int8(factor - minus_factor)

im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(256,24)),(256,160))))
magnitude_vals = im_lowres.reshape(-1)
k = int(round(0.05 * magnitude_vals.shape[0]))
scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
kspace = kspace/scale
target = target/scale

# Apply kspace crop on target
target_torch = cplx.to_tensor(target)
target_torch = T.ifft2( T.kspace_cut(T.fft2(target_torch),0.67,0.67))
target = cplx.to_numpy(target_torch)
# Convert everything from numpy arrays to tensors
kspace_torch = cplx.to_tensor(kspace).float()
kspace_torch = T.awgn_torch(kspace_torch,SNR,L=1)
kspace_noised = kspace_torch.clone()
kspace_noised = T.kspace_cut(kspace_noised,0.67,0.67)
kspace_torch = T.kspace_cut(kspace_torch,0.67,0.67)
target_torch = cplx.to_tensor(target).float()

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
    kspace_torch = transforms.apply_mask(kspace_torch, mask_func)[0]
else:
    ## 2d poisson
    kspace_torch = kspace_torch*mask_torch_pois
    


# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)







extractor_features = FeatureExtractor() # For the final features
extractor_grid = FeatureExtractor()     # To find the grid size

# 3. Get the target layers and register the hooks
try:
    # Get the model's internal transformer network
    vit_net = model.similaritynets[0].recon_net.net

    # Hook for final features (output of LayerNorm)
    layer_norm = vit_net.norm
    hook_features = layer_norm.register_forward_hook(extractor_features.hook_fn)
    
    # Hook for grid size (output of patch_embed)
    layer_patch_embed = vit_net.patch_embed
    hook_grid = layer_patch_embed.register_forward_hook(extractor_grid.hook_fn)
    
except AttributeError as e:
    print(f"Error: Could not find the target layer. {e}")
    print("Please double-check the path: 'model.similaritynets[0].recon_net.net'")
    # Handle error appropriately
    
# 4. Prepare your sample data
# ... (This is already done in your script) ...

# 5. Run the forward pass
print("Running forward pass to capture features...")
with torch.no_grad():
    output_image = model(kspace_torch.float().unsqueeze(0).to(device),
                       reference_torch.float().unsqueeze(0).to(device))

# 6. IMPORTANT: Remove BOTH hooks
hook_features.remove()
hook_grid.remove()
print("Forward pass complete. Hooks removed.")

# 7. Now, process the extracted data

# Get grid shapes. patch_embed was called twice (input, ref)
if len(extractor_grid.features) == 2:
    # The output of patch_embed is [B, C, H_grid, W_grid]
    grid_shape_input = extractor_grid.features[0].shape[2:] # (H_grid, W_grid)
    grid_shape_ref = extractor_grid.features[1].shape[2:]   # (H_grid, W_grid)
    
    print(f"Detected input grid shape (H_grid, W_grid): {grid_shape_input}")
    print(f"Detected ref grid shape (H_grid, W_grid): {grid_shape_ref}")
else:
    print(f"Error: Expected 2 calls to patch_embed, got {len(extractor_grid.features)}")
    # Handle error

# Get final features. norm was called twice (input, ref)
if len(extractor_features.features) == 2:
    features_input = extractor_features.features[0]  # From mag_image
    features_ref = extractor_features.features[1]    # From mag_ref
    
    print(f"Extracted features shape: {features_input.shape}")
else:
    print(f"Error: Expected 2 calls to norm, got {len(extractor_features.features)}")
    # Handle error


#### Visualization of the extracted features ####

def visualize_features(features_tensor, grid_shape, title, save_filename):
    """
    Reshapes and plots the mean features.
    
    Args:
        features_tensor (torch.Tensor): Shape [B, N, C]
        grid_shape (tuple): (H_grid, W_grid) from the patch_embed output
        title (str): Title for the plot
        save_filename (str): Filename to save the plot (e.g., "plot.png")
    """
    B, N, C = features_tensor.shape
    H_grid, W_grid = grid_shape

    # Assuming batch size is 1 for visualization
    if B > 1:
        print("Visualizing features for batch item 0 only.")
        features_tensor = features_tensor[0:1]

    # THE CRITICAL CHECK
    if N != H_grid * W_grid:
        print(f"Error: Patch count mismatch! N={N}, H_grid*W_grid={H_grid * W_grid}")
        print(f"--> Something is wrong. N={N} but detected grid is {grid_shape}")
        return
    
    print(f"Reshaping features from [1, {N}, C] to [1, {H_grid}, {W_grid}, C]")

    # Reshape: [B, N, C] -> [B, H_grid, W_grid, C]
    features_image = features_tensor.reshape(B, H_grid, W_grid, C)
    
    # Permute to [B, C, H_grid, W_grid]
    features_image = features_image.permute(0, 3, 1, 2)
    
    # Create a single 2D map by taking the mean across the channel dimension
    # Shape becomes [B, H_grid, W_grid]
    mean_features = torch.mean(features_image, dim=1).squeeze(0)
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(mean_features.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    
    # Ensure directory exists and save
    os.makedirs('../XAI_results', exist_ok=True)
    save_path = os.path.join('../XAI_results', save_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    print(f"Successfully saved feature map to {save_path}")


# --- NO LONGER NEEDED ---
# H, W = 320, 320 
# patch_size = (10, 10) 
# ---

# Check if all required variables exist before plotting
if 'features_input' in locals() and 'grid_shape_input' in locals():
    # Visualize the features from the 'current input' (mag_image)
    visualize_features(
        features_input, 
        grid_shape=grid_shape_input,
        title="Features from Current Input (mag_image)",
        save_filename="XAI_fuser_map_CURRENT_INPUT.png"
    )
else:
    print("Could not plot Current Input features. Data not found.")

if 'features_ref' in locals() and 'grid_shape_ref' in locals():
    # Visualize the features from the 'reference input' (mag_ref)
    visualize_features(
        features_ref, 
        grid_shape=grid_shape_ref,
        title="Features from Reference Input (mag_ref)",
        save_filename=f"XAI_fuser_map_REFERENCE_sub{patient}_slice_{slice_number}_SNR_{SNR}.png"
    )
else:
    print("Could not plot Reference Input features. Data not found.")


########### Magnitudes of input images ###########
print("Plotting and saving the input magnitude images...")

try:
    # We calculate the inputs as seen by the ViTfuser
    
    # 1. Reference Magnitude
    # reference_torch is shape [H, W, 2]
    ref_mag = cplx.abs(reference_torch) 
    
    # 2. Zero-Filled (Current) Input Magnitude
    # kspace_torch is the masked k-space, shape [H, W, 2]
    # T.ifft2 is the inverse FFT, equivalent to Sense.adjoint(kspace)
    zf_image = T.ifft2(kspace_torch) 
    input_mag = cplx.abs(zf_image)

    # 3. Target (Ground Truth) Magnitude
    # target_torch is shape [H, W, 2]
    target_mag = cplx.abs(target_torch)

    # Move to CPU & NumPy for plotting
    ref_mag_np = ref_mag.cpu().numpy()
    input_mag_np = input_mag.cpu().numpy()
    target_mag_np = target_mag.cpu().numpy() # <-- Added target

    # Create side-by-side plot (changed to 1x3)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) # <-- Changed to 1, 3 and updated size
    
    # Plot Current Input (Zero-Filled)
    axes[0].imshow(input_mag_np, cmap='gray')
    axes[0].set_title('Current Input (Zero-Filled Magnitude)')
    axes[0].axis('off')
    
    # Plot Reference Input
    axes[1].imshow(ref_mag_np, cmap='gray')
    axes[1].set_title('Reference Input (Magnitude)')
    axes[1].axis('off')
    
    # Plot Target (Ground Truth)
    axes[2].imshow(target_mag_np, cmap='gray') # <-- Added target plot
    axes[2].set_title('Target (Ground Truth)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs('../XAI_results', exist_ok=True)
    save_path_inputs = os.path.join('../XAI_results', f'XAI_INPUT_IMAGES_sub{patient}_slice_{slice_number}_SNR_{SNR}.png')
    plt.savefig(save_path_inputs, bbox_inches='tight', dpi=300)
    print(f"Successfully saved input images to {save_path_inputs}")

except Exception as e:
    print(f"Error plotting input images: {e}")
# --- END OF NEW SECTION ---


# --- NEW SECTION: Plot and Save Overlay Heatmaps ---

def normalize_img(img):
    """Normalize a NumPy image to 0-1 range for visualization"""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        # Avoid division by zero if image is all one value
        return np.zeros_like(img)

def process_and_upsample(features_tensor, grid_shape, target_shape):
    """
    Processes raw features into a heatmap and upsamples it to the target image size.
    
    Args:
        features_tensor (torch.Tensor): Raw features [B, N, C]
        grid_shape (tuple): Low-res grid (H_grid, W_grid)
        target_shape (tuple): High-res target (H_img, W_img)
        
    Returns:
        (np.array): A 2D NumPy array of the upsampled heatmap, normalized 0-1.
    """
    B, N, C = features_tensor.shape
    H_grid, W_grid = grid_shape
    
    # Safety check
    if N != H_grid * W_grid:
        print(f"Overlay Error: Patch count mismatch. N={N}, H_grid*W_grid={H_grid * W_grid}")
        return None

    # 1. Reshape and get mean features (low-res heatmap)
    # [B, N, C] -> [B, H_grid, W_grid, C]
    mean_features = features_tensor.reshape(B, H_grid, W_grid, C)
    # [B, H_grid, W_grid, C] -> [B, C, H_grid, W_grid]
    mean_features = mean_features.permute(0, 3, 1, 2)
    # Take mean over channel dim -> [B, H_grid, W_grid]
    mean_features = torch.mean(mean_features, dim=1) 
    
    # 2. Add batch/channel dims for interpolate
    # [B, H_grid, W_grid] -> [B, 1, H_grid, W_grid]
    mean_features = mean_features.unsqueeze(1)
    
    # 3. Upsample the heatmap
    upsampled_heatmap = F.interpolate(
        mean_features,
        size=target_shape,       # Target H, W
        mode='bilinear',         # Smooth interpolation
        align_corners=False
    )
    
    # 4. Squeeze, normalize, and convert to NumPy
    upsampled_heatmap = upsampled_heatmap.squeeze().cpu().numpy()
    upsampled_heatmap = normalize_img(upsampled_heatmap)
    
    return upsampled_heatmap

print("Plotting and saving cross-analysis overlay heatmaps...")

try:
    # Check if all necessary variables exist
    if ('features_input' in locals() and 'grid_shape_input' in locals() and 
        'features_ref' in locals() and 'grid_shape_ref' in locals() and 
        'input_mag_np' in locals() and 'ref_mag_np' in locals()):
        
        # --- Process all 3 required heatmap combinations ---
        
        # 1. Input features, upsampled to Input image shape
        heatmap_input_on_input = process_and_upsample(
            features_input, 
            grid_shape_input, 
            input_mag_np.shape
        )
        
        # 2. Reference features, upsampled to Input image shape
        heatmap_ref_on_input = process_and_upsample(
            features_ref,       
            grid_shape_ref,     
            input_mag_np.shape  
        )
        
        # 3. Reference features, upsampled to Reference image shape
        heatmap_ref_on_ref = process_and_upsample(
            features_ref,       # <-- CHANGED
            grid_shape_ref,     # <-- CHANGED
            ref_mag_np.shape    
        )

        # Check if processing was successful
        if (heatmap_input_on_input is not None and 
            heatmap_ref_on_input is not None and 
            heatmap_ref_on_ref is not None):  # <-- CHANGED
            
            # Create side-by-side plot with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            
            # --- Plot 1: Input + Input Heatmap ---
            axes[0].imshow(normalize_img(input_mag_np), cmap='gray')
            im0 = axes[0].imshow(heatmap_input_on_input, cmap='jet', alpha=0.5, interpolation='bilinear')
            axes[0].set_title('Input + Input Heatmap')
            axes[0].axis('off')
            
            # --- Plot 2: Input + Reference Heatmap ---
            axes[1].imshow(normalize_img(input_mag_np), cmap='gray')
            im1 = axes[1].imshow(heatmap_ref_on_input, cmap='jet', alpha=0.5, interpolation='bilinear')
            axes[1].set_title('Input + Reference Heatmap')
            axes[1].axis('off')

            # --- Plot 3: Reference + Reference Heatmap ---
            axes[2].imshow(normalize_img(ref_mag_np), cmap='gray')
            im2 = axes[2].imshow(heatmap_ref_on_ref, cmap='jet', alpha=0.5, interpolation='bilinear') # <-- CHANGED
            axes[2].set_title('Reference + Reference Heatmap') # <-- CHANGED
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Add a shared colorbar for the heatmaps
            # We use im2 (the last plot) as the reference for the colorbar
            fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, aspect=40, label='Normalized Feature Activation')
            
            # Ensure directory exists and save
            os.makedirs('../XAI_results', exist_ok=True)
            # Save to a new file name
            save_path_overlay = os.path.join('../XAI_results', f'XAI_OVERLAY_CROSS_ANALYSIS_sub{patient}_slice_{slice_number}_SNR_{SNR}.png')
            plt.savefig(save_path_overlay, bbox_inches='tight', dpi=300)
            print(f"Successfully saved cross-analysis overlay images to {save_path_overlay}")
        else:
            print("Could not create overlays due to processing errors.")
    else:
        print("Could not plot overlays. Missing one or more required variables.")
            
except Exception as e:
    print(f"Error plotting overlay images: {e}")



############### Monkey pruning test ##################
# -------------------------------------------------------------------------
# 1. FUNCTION DEFINITIONS (MUST BE INCLUDED IN THE SCRIPT)
# -------------------------------------------------------------------------

# Function to apply the specialized pruning
def prune_fusion_parameters_reference_only(model, pruning_percentage):
    """
    Prunes the smallest absolute values from the Reference parameter (param2) 
    AND sets the Input parameter (param1) entirely to zero.
    
    Args:
        model (ViTFuserWrap): The wrapper model instance.
        pruning_percentage (float): The percentage of param2 to zero out (e.g., 0.1 for 10%).
        
    Returns:
        tuple: (original_param1, original_param2) copies to restore later.
    """
    # Assuming the ViTfuser is model.similaritynets[0]
    fuser = model.similaritynets[0]
    
    # --- STEP 1: Store Originals ---
    original_p1 = fuser.param1.data.clone()
    original_p2 = fuser.param2.data.clone()
    
    # --- STEP 2: Prune param1 (Input) -> SET TO ZERO ---
    
    # This forces the network to ignore all input features (F_in)
    fuser.param1.data.zero_() 
    
    # --- STEP 3: Prune param2 (Reference) based on percentage ---
    
    p2 = fuser.param2.data.abs()
    k_to_zero = int(pruning_percentage * p2.numel())
    
    if k_to_zero > 0:
        # Find the k-th smallest value (threshold)
        # Using torch.kthvalue on the flattened tensor
        threshold_p2 = torch.kthvalue(p2.view(-1), k_to_zero, keepdim=False).values
        # Create mask: 1 if abs value > threshold, 0 otherwise
        mask_p2 = (fuser.param2.data.abs() > threshold_p2).float()
        # Apply mask to param2
        fuser.param2.data.mul_(mask_p2)
    
    return original_p1, original_p2

# Function to restore parameters
def restore_fusion_parameters(model, original_p1, original_p2):
    fuser = model.similaritynets[0]
    fuser.param1.data.copy_(original_p1)
    fuser.param2.data.copy_(original_p2)

# -------------------------------------------------------------------------
# 2. EXECUTION BLOCK
# -------------------------------------------------------------------------

# Define the pruning percentages you want to test for the REFERENCE weight (P2)
pruning_levels = [0.0, 0.1, 0.2] 

# Store results for plotting
reconstructions = {}

print("Starting REFERENCE-ONLY Fusion Parameter Pruning Test...")

# Store the original parameters before the loop starts
fuser = model.similaritynets[0]
original_p1_full = fuser.param1.data.clone()
original_p2_full = fuser.param2.data.clone()


for p in pruning_levels:
    print(f"Running inference with Input weight (P1) suppressed and Reference weight (P2) pruned by {p*100}%...")
    
    # 1. Apply the specialized pruning: P1=0, P2=P2_pruned
    prune_fusion_parameters_reference_only(model, pruning_percentage=p)
    
    # 2. Run Inference
    with torch.no_grad():
        output_image = model(kspace_torch.float().unsqueeze(0).to(device),
                           reference_torch.float().unsqueeze(0).to(device))
    
    # 3. Process Output (Magnitude)
    output_complex = output_image[0] # Shape [H, W, 2]
    output_mag = torch.sqrt(output_complex[..., 0]**2 + output_complex[..., 1]**2)
    
    reconstructions[p] = output_mag.cpu().numpy()
    
    # 4. Restore parameters to ensure the next iteration starts from the clean state
    restore_fusion_parameters(model, original_p1_full, original_p2_full)


# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
print("Visualizing Reference-Only Pruning Results...")

# Check if target_mag_np exists (reusing previous logic)
if 'target_mag_np' not in locals():
    target_mag = cplx.abs(target_torch)
    target_mag_np = target_mag.cpu().numpy()


num_plots = len(pruning_levels) + 1 
fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

# Plot Target
axes[0].imshow(target_mag_np, cmap='gray')
axes[0].set_title("Target (Ground Truth)")
axes[0].axis('off')

# Plot Reconstructions
for i, p in enumerate(pruning_levels):
    ax = axes[i+1]
    img = reconstructions[p]
    
    # Normalize for display
    img = (img - img.min()) / (img.max() - img.min())
    
    ax.imshow(img, cmap='gray')
    if p == 0.0:
        ax.set_title("Reference-Only (P1=0, P2 No Pruning)")
    else:
        ax.set_title(f"Ref-Only: P2 Pruned {p*100}% Smallest")
    ax.axis('off')

plt.tight_layout()
os.makedirs('../XAI_results', exist_ok=True)
save_path_pruning = os.path.join('../XAI_results', f'XAI_REF_ONLY_PRUNING_sub{patient}_SNR_{SNR}.png')
plt.savefig(save_path_pruning, bbox_inches='tight', dpi=300)
print(f"Saved Reference-Only fusion pruning results to {save_path_pruning}")