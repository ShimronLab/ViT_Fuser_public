
import os, sys
# Add the path to the first folder
folder1_path = '../'
sys.path.append(folder1_path)
from fastmri.data import transforms, subsample
import logging
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
from models_halbach.ViTFuserWrapHalbach import ViTFuserWrap
from models_halbach.ViTWrapHalbach import ViTWrap
import matplotlib.pyplot as plt
# import custom libraries
from utils import transforms as T
from utils import complex_utils as cplx
# import custom classes
import matplotlib.pyplot as plt
from scipy.io import loadmat
import nibabel as nib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

VD = 0
equispaced = 1
diff_contrast = 2
second_scan = 1
factor = 3
phase_mask = 0 
subject = 2
SEED = 0
# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load the .mat file
mat_file = "../ULFData/bSSFP_T1rho_brain_scans_6.5mT.mat"
mat_data = loadmat(mat_file)
print(mat_data.keys())
key_name = 'bSSFP'  # Replace with the actual key name
if key_name in mat_data:
    data = mat_data[key_name]
    print("Read the key ", key_name)
else:
    print(f"Key '{key_name}' not found in the .mat file.")

print(data.shape)
slice_num = 9
slice_data = np.rot90(data[:,:,slice_num],-1)

# Plot the slice
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(slice_data), cmap='gray')  # Use 'gray' colormap for MRI images
plt.title(f"Slice {slice_num}")
plt.colorbar(label='Intensity')
plt.axis('off')  # Optional: Turn off axes for a cleaner plot
plt.show()



# Load the nii files
if subject ==1:
    nifti_file_HF = "../ULFData/HFSub1regULFSub1.nii"
    nifti_file_LF = "../ULFData/ULFSub1reg.nii"
else:
    nifti_file_HF = "../ULFData/HF2reg2ULF2.nii"
    nifti_file_LF = "../ULFData/ULF2reg2.nii"

if diff_contrast ==1:
    if subject ==1:
        nifti_file_HF = "../ULFData/HFreg1ContrastULF.nii"
        nifti_file_LF = "../ULFData/ULFreg1Contrast.nii"
    else:
        nifti_file_HF = "../ULFData/HFregContrastULF.nii"
        nifti_file_LF = "../ULFData/ULFregContrast.nii"

if diff_contrast ==2:
    if subject ==1:
        nifti_file_HF = "../ULFData/HF23regT2ULF10.nii"
        nifti_file_LF = "../ULFData/ULF10regT2.nii"
    elif subject ==2:
        nifti_file_HF = "../ULFData/HF22regT2ULF1022.nii"
        nifti_file_LF = "../ULFData/ULF1022regT2.nii"  
    else:
        nifti_file_HF = "../ULFData/HF24regT2ULF924.nii"
        nifti_file_LF = "../ULFData/ULF924regT2.nii"   

if second_scan ==1:
    if subject ==1:
        nifti_file_HF = "../ULFData/HF23regT2ULF13.nii"
        nifti_file_LF = "../ULFData/ULF13regT2.nii"
    elif subject ==2:
        nifti_file_HF = "../ULFData/HF24regT2ULF12.nii"
        nifti_file_LF = "../ULFData/ULF12regT2.nii"   
    else:
        nifti_file_HF = "../ULFData/HF21regT2ULF14.nii"
        nifti_file_LF = "../ULFData/ULF14regT2.nii"          

nifti_data_HF = nib.load(nifti_file_HF)
nifti_data_LF = nib.load(nifti_file_LF)

# Extract the image data
image_data_HF = nifti_data_HF.get_fdata()
image_data_LF = nifti_data_LF.get_fdata()

concat = np.concatenate((image_data_HF[0,:,:],image_data_LF[0,:,:]),axis=1)
# Plot the image
plt.imshow(concat, cmap="gray")
plt.title("Visit 1-HF                            Visit 2-LF")
plt.axis("off")
plt.show()

print(image_data_HF.shape)

# Padding size
# Desired size
target_height, target_width = 172, 108
target_height, target_width = 155, 155

# Calculate padding for each side
pad_height = target_height - 69
if second_scan ==1:
    pad_height = target_height - 72
pad_width = target_width - 64

top = pad_height // 2
bottom = pad_height - top
left = pad_width // 2
right = pad_width - left

size1 = 69
size2 = 64
if second_scan ==1:
    size1 = 72

image_data_HF = nifti_data_HF.get_fdata()
image_data_LF = nifti_data_LF.get_fdata()

padded_HF = np.pad(image_data_HF[0,:,:], ((top, bottom), (left, right)), mode='constant', constant_values=0)

padded_LF = np.pad(image_data_LF[0,:,:], ((top, bottom), (left, right)), mode='constant', constant_values=0)


target = torch.from_numpy(np.expand_dims(padded_LF,0))
reference = padded_HF
# Padding
print(f'Target shape: {target.shape}')
print(f'Reference shape: {reference.shape}')


random_phase = torch.angle(T.random_map((1,155,155), 'cpu',kspace_radius_range=(0.001, 0.001))) * phase_mask
target = target * (torch.exp(1j * random_phase)).numpy() 
target = target.squeeze(0)
target_torch = cplx.to_tensor(target).float() 
reference_torch = cplx.to_tensor(reference).float()


reference_kspace_torch = T.fft2(reference_torch)
reference_kspace = cplx.to_numpy(reference_kspace_torch)
kspace_torch = T.fft2(target_torch)
target = cplx.to_numpy(target_torch)
kspace = cplx.to_numpy(kspace_torch)

kspace_torch = T.kspace_crop(kspace_torch,0.67)



# %%
im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(155,24)),(155,155))))
magnitude_vals = im_lowres.reshape(-1)
k = int(round(0.05 * magnitude_vals.shape[0]))
scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
kspace = kspace/scale
target = target/scale

# Apply kspace crop on target
target_torch = cplx.to_tensor(target)
target_torch = T.ifft2( T.fft2(target_torch))
# For plot
kspace_HR = np.abs(cplx.to_numpy(T.fft2(cplx.to_tensor(target))))
kspace_LR =cplx.to_numpy( T.fft2( cplx.to_tensor(target)))
target_HR = target
target_LR = cplx.to_numpy(T.ifft2( T.kspace_crop(T.fft2( cplx.to_tensor(target)),0.67)))
target = cplx.to_numpy(target_torch)
# Convert everything from numpy arrays to tensors
kspace_torch = cplx.to_tensor(kspace).float()
kspace_noised = kspace_torch.clone()
target_torch = cplx.to_tensor(target).float()

### Reference addition ###
im_lowres_ref = abs(sp.ifft(sp.resize(sp.resize(reference_kspace,(172,24)),(172,108))))
magnitude_vals_ref = im_lowres_ref.reshape(-1)
k_ref = int(round(0.05 * magnitude_vals_ref.shape[0]))
scale_ref = magnitude_vals_ref[magnitude_vals_ref.argsort()[::-1][k_ref]]
reference = reference / scale_ref
reference_torch = cplx.to_tensor(reference).float()
reference_torch_kspace = T.fft2(reference_torch)
reference_torch = T.ifft2(reference_torch_kspace)


def get_mask_func(factor, seed=1):
    torch.manual_seed(seed)  # Set PyTorch seed
    np.random.seed(seed)  # Set NumPy seed
    
    center_fractions = 0.08 * 4 / factor
    mask_func = subsample.EquiSpacedMaskFunc(
        center_fractions=[center_fractions],
        accelerations=[factor],
        seed=seed  # Ensure deterministic mask generation
    )
    return mask_func

if equispaced == 1 :
    mask_func = get_mask_func(factor)
    kspace_torch = transforms.apply_mask(kspace_torch, mask_func)[0]
elif VD ==1:
        calib = np.array([18,18])
        mask1, pdf, poly_degree = T.gen_2D_var_dens_mask(factor, (155,155), 'strong', calib=calib)
        print(mask1.shape)
        mask_expanded = torch.stack([torch.tensor(mask1).float(),torch.tensor(mask1).float()],dim=2)

        kspace_torch = kspace_torch*mask_expanded
concat = np.concatenate((target,cplx.to_numpy(T.ifft2(kspace_noised)),np.abs(cplx.to_numpy(kspace_torch))!=0,cplx.to_numpy(T.ifft2(kspace_torch))),axis=1)
fig, axs = plt.subplots(1, 1, figsize=(20, 5))  # 1 row, 3 columns
# Plot each image in a subplot
im1 = axs.imshow(np.abs(concat), cmap='gray')
plt.title('     Low-Res scan                      Low-Res Noised scan                 Kspace Sampling mask, R=3          synthetic Low-Field scan')
plt.axis('off')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(10, 5))  # 1 row, 3 columns
print(target_torch.shape)
print(reference_torch.shape)

concat2 = np.concatenate((cplx.to_numpy(target_torch),cplx.to_numpy(reference_torch)),axis=1)
im1 = axs.imshow(np.abs(concat2), cmap='gray')
#im1 = axs.imshow(np.log(np.abs(cplx.to_numpy(kspace_torch))), cmap='gray')
plt.title('Visit 2                 Visit 1')
plt.axis('off')
plt.show()

mask_np = np.abs(cplx.to_numpy(kspace_torch))!=0
print(f'Mask torch size: {mask_np.shape}')
s = (64)*(69)
print(f'Acceleration factor R: {(np.sum(mask_np[bottom:-top,left:-right]))/s}')


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


#Hyper parameters
params = Namespace()
#params.data_path = "./registered_data/patient23b/"
params.data_path = "./registered_data/"
params.batch_size = 2
params.num_grad_steps = 1 # ViTs
params.num_cg_steps = 8 
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.00001
params.weight_decay = 0
params.lr_step_size = 7
params.lr_gamma = 0.1
params.epoch = 61
params.reference_mode = 1
params.reference_lambda = 0.1

# %%

checkpoint_file = "./L2_checkpoints_Sairam/model_40.pt" 
checkpoint_file_fuser = "./checkpoints_15dB/model_ViTFuserMSE_100.pt" 
checkpoint_file_fuser = "../checkpoints_halbach_matt/model_40.pt" # uses 155
checkpoint_file_fuser = "../checkpoints_ViTFuser_FT_new_sairam/model_30.pt" # uses 155

checkpoint_file_fuser = "../checkpoints_Chloe_fuser/model_43.pt" # uses 155
checkpoint_file = "../checkpoints_ViTFuser_FT_new_sairam/model_30.pt" # uses 155

checkpoint_file_fuser = "../checkpoints_halbach_matt/model_40.pt" # 40 - mid overfir, 80 - big overfit
checkpoint_file_vit =  "../checkpoints_halbach_ViT/model_40.pt" # test on MSE ViT R=3


checkpoint_file_fuser = "../checkpoints_halbach_lambda/model_R3_40.pt" # 2 FT no noise added
checkpoint_file_fuser = "../checkpoints_halbach_lambda/model_R3_Matt_40.pt" #  FT , without added noise - 0.1 hybrid
checkpoint_file_fuser = "../checkpoints_halbach_lambda/model_R3_Matt_noise_40.pt" # FT , with added noise - 0.1 hybrid


checkpoint = torch.load(checkpoint_file,map_location=device)
checkpoint_fuser = torch.load(checkpoint_file_fuser,map_location=device)
checkpoint_vit = torch.load(checkpoint_file_vit,map_location=device)


model = ViTFuserWrap(params).to(device)
model_fuser = ViTFuserWrap(params).to(device)
model_vit = ViTWrap(params).to(device)

# load checkpoint
model.load_state_dict(checkpoint['model'])
model_fuser.load_state_dict(checkpoint_fuser['model'])
model_vit.load_state_dict(checkpoint_vit['model'])


# %%


img = cplx.to_tensor(np.abs(cplx.to_numpy(T.ifft2(kspace_torch)))).permute(2,0,1).unsqueeze(0).to(device)
img_chan = img[:,0,:,:].unsqueeze(0)
ref = cplx.to_tensor(np.abs(cplx.to_numpy(reference_torch))).permute(2,0,1).unsqueeze(0).to(device)
ref_chan = ref[:,0,:,:].unsqueeze(0)
ref_np = ref_chan.cpu().numpy()[0,0,:,:]
print(img_chan.shape)
img_padded_np = img_chan.cpu().numpy()[0,0,:,:]

## Regular
print(kspace_torch.shape)
print(reference_torch.shape)
im_out = model(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
print(im_out.shape)
im_out_pad = torch.cat((im_out,torch.zeros_like(im_out)),dim=2)
print(im_out_pad.shape)
im_out = T.ifft2(T.fft2(im_out_pad))
target_torch = T.ifft2(T.fft2(cplx.to_tensor(target)))
target = cplx.to_numpy(target_torch.cpu().detach())
im_out = np.abs(cplx.to_numpy(im_out.cpu().detach()))
print(im_out.shape)

## Fuser
im_out_fuser = model_fuser(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_fuser_pad = torch.cat((im_out_fuser,torch.zeros_like(im_out_fuser)),dim=2)
im_out_fuser = T.ifft2(T.fft2(im_out_fuser_pad))
im_out_fuser = np.abs(cplx.to_numpy(im_out_fuser.cpu().detach()))

## ViT only
im_out_vit = model_vit(kspace_torch.float().unsqueeze(0).to(device),reference_torch.float().unsqueeze(0).to(device)).squeeze(0)
im_out_vit_pad = torch.cat((im_out_vit,torch.zeros_like(im_out_vit)),dim=2)
im_out_vit = T.ifft2(T.fft2(im_out_vit_pad))
im_out_vit = np.abs(cplx.to_numpy(im_out_vit.cpu().detach()))

# Norm
im_out = im_out/np.max(im_out)
im_out_fuser = im_out_fuser/np.max(im_out_fuser)
im_out_vit = im_out_vit/np.max(im_out_vit)
ref_np = ref_np/np.max(ref_np)
img_padded_np = img_padded_np/np.max(img_padded_np)
target = target/np.max(target)

st = 44
ed = 40
st1 = 48
ed1 = 48

# Concatenate images horizontally
concatenated_image = np.concatenate((ref_np[st:-ed,st1:-ed1],img_padded_np[st:-ed,st1:-ed1], im_out_vit[st:-ed,st1:-ed1] ,im_out[st:-ed,st1:-ed1], im_out_fuser[st:-ed,st1:-ed1],np.abs(target[st:-ed,st1:-ed1])),axis=1)
# Plot the concatenated image
plt.figure(figsize=(12, 6))
plt.imshow(concatenated_image, cmap='gray')
plt.title('Prior                          Input                       ViT MSE                    ViT-Fuser no FT                  ViT-Fuser FT                            target')
plt.axis('off')
plt.show()

