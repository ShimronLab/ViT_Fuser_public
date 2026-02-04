import numpy as np
import torch
import os
import argparse
from torch_utils.misc import nrmse_np, psnr, nrmse
import pickle
import dnnlib
from torch_utils.misc import StackedRandomGenerator
from torch_utils import distributed as dist
from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np


# --- Helper to convert Complex (B, 1, H, W) -> Real (B, 2, H, W) ---
def complex_to_real(x):
    """
    Robustly converts a complex tensor to a real tensor with 2 channels.
    Input: Complex tensor of arbitrary shape [..., H, W]
    Output: Real tensor [B, 2, H, W]
    """
    # 1. Strip extra singleton dimensions until we have at most 4 dims [B, C, H, W]
    # This handles cases like [1, 1, 1, H, W] -> [1, 1, H, W]
    while x.ndim > 4:
        if x.shape[1] == 1:
            x = x.squeeze(1)
        else:
            break
            
    # 2. Ensure we have at least 4 dims [B, 1, H, W]
    # This handles cases like [1, H, W] -> [1, 1, H, W]
    if x.ndim == 3:
        x = x.unsqueeze(1)
        
    # Now x is guaranteed to be [B, C, H, W] (usually C=1)
    
    # 3. View as real -> [B, C, H, W, 2]
    x_real = torch.view_as_real(x)
    
    # 4. Move the real/imag dimension (last) to the channel dimension
    # Permute: [B, C, H, W, 2] -> [B, C, 2, H, W]
    x_real = x_real.permute(0, 1, 4, 2, 3)
    
    # 5. Combine C and 2 into a single channel dimension
    # [B, C*2, H, W] -> For C=1, this gives [B, 2, H, W]
    B, C, Two, H, W = x_real.shape
    x_real = x_real.reshape(B, C * Two, H, W)
    
    return x_real.contiguous()
def complex_to_real_channels(x):
    # Ensure input is at least 4D [B, 1, H, W]
    if x.ndim == 3:
        x = x.unsqueeze(1)
    
    # View as real -> [B, 1, H, W, 2]
    x_real = torch.view_as_real(x)
    
    # Permute to move Real/Imag to channel dimension -> [B, 2, H, W]
    # We take the last dim (2) and merge it with dim 1
    x_real = x_real.permute(0, 4, 2, 3).squeeze(1).contiguous()
    
    # Final check to ensure shape is [B, 2, H, W]
    if x_real.ndim == 5 and x_real.shape[1] == 1:
         x_real = x_real.squeeze(1)
         
    return x_real

class MRI_utils:
    def __init__(self, mask, maps):
        self.mask = mask
        self.maps = maps

    def forward(self,x):
        x_cplx = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous())[:,None,...]
        coil_imgs = self.maps*x_cplx
        coil_ksp = fft(coil_imgs)
        sampled_ksp = self.mask*coil_ksp
        return sampled_ksp

    def adjoint(self,y):
        sampled_ksp = self.mask*y
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.maps)*coil_imgs,dim=1) #sum over coil dimension

        return img_out[:,None,...]
    
# Centered, orthogonal fft in torch >= 1.7
def fft(x):
    x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    return x

# Centered, orthogonal ifft in torch >= 1.7
def ifft(x):
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    return x

def fftmod(x):
    x[...,::2,:] *= -1
    x[...,:,::2] *= -1
    return x

def general_forward_SDE_ps(
    y, y_prior_ksp, img_prior, gt_img, mri_inf_utils, mri_prior_utils, task, l_type, l_ss, net, latents, inference_snr_db=None, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7, 
    solver='euler', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, verbose = True
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    ##### My additions #####
    sigma_input_threshold = 0.0
    if inference_snr_db < 100:
        # Calculate Signal Power P using gt_img (clean signal)
        # gt_img shape is likely [B, 1, H, W] complex or similar. 
        # Ensure we compute mean squared magnitude.
        # Assuming gt_img is the complex ground truth.
        # If gt_img is torch tensor:
        #s_mag_sq = torch.abs(img_prior)**2
        #P = torch.mean(s_mag_sq).item() # L=1 assumed
        #P = 1
        zf_estimator = mri_inf_utils.adjoint(y)  # Get image domain from k-space measurements
        s_mag_sq = torch.abs(zf_estimator)**2
        P = torch.mean(s_mag_sq).item() # L=1 assumed
        if inference_snr_db == 15:
            decresed_snr_db = inference_snr_db -3  # Reduce SNR by 3 dB to optimize reconstruction for 15dB
        else:
            decresed_snr_db = inference_snr_db
        gamma = 10**(-float(decresed_snr_db)/10)
        #gamma = 10**(-float(inference_snr_db)/10)
        N0 = P * gamma
        sigma_input_threshold = np.sqrt(N0 / 2) 
    if inference_snr_db >100:
        inference_snr_db = inference_snr_db - 100
    if inference_snr_db == 20:
        i_stop_prior = 320
    if inference_snr_db == 15:
        i_stop_prior = 300
    elif inference_snr_db == 10:
        i_stop_prior = 270  # 250 - poisson, 270 - equispaced
    elif inference_snr_db == 5:
        i_stop_prior = 230 # 200 - poisson, 230 - equispaced
    elif inference_snr_db == 3:
        i_stop_prior = 180 # 150 - poisson, 180 - equispaced
    elif inference_snr_db == 0:
        i_stop_prior = 130 #  100 - poisson, 130 - equispaced
    if verbose:     
        print(f"Calculated Input Noise Sigma: {sigma_input_threshold:.4e} (SNR: {inference_snr_db} dB)")
    last_dc_step = 0
    ########################

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        x_cur = x_cur.requires_grad_() #starting grad tracking with the noised img

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step on Prior.
        h = t_next - t_hat
        
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        
        ###### My Test ######
        current_sigma = sigma(t_hat).item()
        # Apply Data Consistency (Likelihood) only untill the last 250 steps
        if i <= i_stop_prior: 
            last_dc_step += 1
            if l_type == 'DPS':
                E_x_start = (1/s(t_cur))*(x_cur + (s(t_cur)**2)*(denoised-x_cur))
                Ax = mri_prior_utils.forward(x=E_x_start)
            elif l_type == 'ALD':
                Ax = mri_prior_utils.forward(x=x_cur)

            residual = y_prior_ksp - Ax  
            residual = residual.reshape(latents.shape[0],-1)
            sse_ind = torch.norm(residual,dim=-1)**2
            sse = torch.sum(sse_ind)
            likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_cur)[0]
            x_next = x_hat + h * d_cur - (l_ss / torch.sqrt(sse_ind)[:,None,None,None]) * likelihood_score
        
        elif current_sigma > sigma_input_threshold:
        #if i<= 350:
            # Ok: 150 (leaving only 150 steps without DC) for 15dB
            # Noisy: 100 
            last_dc_step += 1
            if l_type == 'DPS':
                E_x_start = (1/s(t_cur))*(x_cur + (s(t_cur)**2)*(denoised-x_cur))
                Ax = mri_inf_utils.forward(x=E_x_start)
            elif l_type == 'ALD':
                Ax = mri_inf_utils.forward(x=x_cur)

            residual = y - Ax  
            residual = residual.reshape(latents.shape[0],-1)
            sse_ind = torch.norm(residual,dim=-1)**2
            sse = torch.sum(sse_ind)
            likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_cur)[0]
            x_next = x_hat + h * d_cur - (l_ss / torch.sqrt(sse_ind)[:,None,None,None]) * likelihood_score
        else:
            x_next = x_hat + h * d_cur

        """
        # Apply Data Consistency (Likelihood) only untill the last 150 steps
        if i <= num_steps - 400 or (i>250 and i<280) or (i>300 and i<330) :  # DC to prior
            if l_type == 'DPS':
                E_x_start = (1/s(t_cur))*(x_cur + (s(t_cur)**2)*(denoised-x_cur))
                Ax = mri_prior_utils.forward(x=E_x_start)
            elif l_type == 'ALD':
                Ax = mri_prior_utils.forward(x=x_cur)

            residual = y_prior_ksp - Ax  
            residual = residual.reshape(latents.shape[0],-1)
            sse_ind = torch.norm(residual,dim=-1)**2
            sse = torch.sum(sse_ind)
            likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_cur)[0]
            x_next = x_hat + h * d_cur - (l_ss / torch.sqrt(sse_ind)[:,None,None,None]) * likelihood_score
        elif i <= num_steps - 150 : # DC to acquired measurements
            if l_type == 'DPS':
                E_x_start = (1/s(t_cur))*(x_cur + (s(t_cur)**2)*(denoised-x_cur))
                Ax = mri_inf_utils.forward(x=E_x_start)
            elif l_type == 'ALD':
                Ax = mri_inf_utils.forward(x=x_cur)

            residual = y - Ax  
            residual = residual.reshape(latents.shape[0],-1)
            sse_ind = torch.norm(residual,dim=-1)**2
            sse = torch.sum(sse_ind)
            likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_cur)[0]
            x_next = x_hat + h * d_cur - (l_ss / torch.sqrt(sse_ind)[:,None,None,None]) * likelihood_score
        else:
            x_next = x_hat + h * d_cur
        """
        """
        # Euler step on liklihood
        if l_type == 'DPS':
            E_x_start = (1/s(t_cur))*(x_cur + (s(t_cur)**2)*(denoised-x_cur))
            Ax = mri_inf_utils.forward(x=E_x_start)
        elif l_type == 'ALD':
            Ax = mri_inf_utils.forward(x=x_cur)

        residual = y - Ax  
        residual = residual.reshape(latents.shape[0],-1)
        sse_ind = torch.norm(residual,dim=-1)**2
        sse = torch.sum(sse_ind)
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_cur)[0]
        x_next = x_hat + h * d_cur - (l_ss / torch.sqrt(sse_ind)[:,None,None,None]) * likelihood_score
        """
        if task=='mri':
            cplx_recon = mri_transform(x_next) #shape: [B,1,H,W]
            with torch.no_grad():
                nrmse_loss = nrmse(abs(gt_img), abs(cplx_recon))
        if verbose:    
            print('Step:%d , Noise LVL: %.3e,  DC Loss: %.3e,  NRMSE: %.3f'%(i, sigma(t_hat), sse.item(), nrmse_loss.item()))

        # Cleanup 
        x_next = x_next.detach()
        x_cur = x_cur.requires_grad_(False)
    return x_cur, last_dc_step

def mri_transform(x):
    return torch.view_as_complex(x.permute(0,-2,-1,1).contiguous())[:,None,...] #shape: [1,1,H,W]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--sample_start', type=int, default=0)
parser.add_argument('--sample_end', type=int, default=100)
parser.add_argument('--l_ss', type=float, default=1)
parser.add_argument('--sigma_max', type=float, default=10)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--inference_R', type=int, default=4)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--latent_seeds', type=int, nargs='+' ,default= [10])
parser.add_argument('--S_churn', type=float, default=40)
parser.add_argument('--net_arch', type=str, default='ddpmpp') 
parser.add_argument('--measurements_path', type=str, default='') 
parser.add_argument('--ksp_path', type=str, default='')
parser.add_argument('--inference_snr', type=str, default='')
parser.add_argument('--discretization', type=str, default='edm') # ['vp', 've', 'iddpm', 'edm']
parser.add_argument('--solver', type=str, default='euler') # ['euler', 'heun']
parser.add_argument('--schedule', type=str, default='linear') # ['vp', 've', 'linear']
parser.add_argument('--scaling', type=str, default='none')
parser.add_argument('--outdir', type=str, default='none')
parser.add_argument('--network', type=str, default='none')
parser.add_argument('--img_channels', type=int, default=2)
parser.add_argument('--method', type=str, default='edm')
parser.add_argument('--snr', type=int, default=None)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device=torch.device('cuda')
batch_size=len(args.latent_seeds)

# load network
net_save = args.network
if dist.get_rank() != 0:
        torch.distributed.barrier()
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

total_params = sum(p.numel() for p in net.parameters())
print(f"\nModel parameters: {total_params}" + "\n")

for args.sample in range(args.sample_start, args.sample_end):
    # load data and preprocess
    print("\nValidation Sample " + str(args.sample+1) + ":\n")
    data_file = args.measurements_path + "/sample_%d.pt"%args.sample
    ksp_file = args.ksp_path + "/sample_%d.pt"%args.sample
    cont = torch.load(data_file)
    cont_ksp = torch.load(ksp_file)
    mask_str = 'mask_%d'%args.inference_R

    gt_img = cont['gt'][None,None,...].cuda() #shape [1,1,384,320]
    s_maps = fftmod(cont_ksp['s_map'])[None,...].cuda() # shape [1,16,384,320]
    fs_ksp = fftmod(cont_ksp['ksp'])[None,...].cuda() #shape [1,16,384,320]
    #prior_img = cont_ksp['prior_img'][None,...].cuda()
    #prior_img = prior_img.squeeze(0)  # shape [1, 2, H, W]
    prior_ksp = fftmod(cont_ksp['prior_ksp'])[None,...].cuda() #shape [1,16,384,320]
    mask = cont[mask_str][None, ...].cuda() # shape [1,1,384,320]
    ksp = mask * fs_ksp

    # setup MRI forward model + utilities for inferance mask
    mri_inf_utils = MRI_utils(maps=s_maps, mask=mask)
    mri_prior_utils = MRI_utils(maps=s_maps, mask=torch.ones_like(mask))
    adj_img = mri_inf_utils.adjoint(ksp)
    # Adjoint Reconstruction: Complex [1, 1, H, W]
    prior_img_complex =  mri_prior_utils.adjoint(prior_ksp)
    # Convert to Real Channels [1, 2, H, W]
    prior_img = complex_to_real(prior_img_complex)
# =========================================================================
    # [ADDED] DEBUG PLOT: Save the Prior Image Condition
    # =========================================================================
    print("Saving debug image for Prior...")
    # 1. Move to CPU
    debug_tensor = prior_img.detach().cpu().numpy() # Shape [1, 2, H, W]
    
    # 2. Compute Magnitude from 2 Channels (Real, Imag)
    # Channel 0 is Real, Channel 1 is Imag
    print("Debug Tensor Shape:", debug_tensor.shape)
    prior_mag = np.sqrt(debug_tensor[0, 0,:,:]**2 + debug_tensor[0, 1,:,:]**2)
    import matplotlib.pyplot as plt
    # 3. Save Plot
    debug_filename = f"debug_prior_sample_{args.sample}.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(prior_mag, cmap='gray')
    plt.colorbar()
    plt.title(f"Prior Input (Condition)\nMin: {prior_mag.min():.2f}, Max: {prior_mag.max():.2f} shape:{prior_img.shape}")
    plt.savefig(debug_filename)
    plt.close()
    print(f"✅ Debug Image Saved: {os.path.abspath(debug_filename)}")
    # =========================================================================

    # Pick latents and labels.
    rnd = StackedRandomGenerator(device, args.latent_seeds)
    latents = torch.load("../logits/latents_%d.pt"%args.sample)['latents'].to(device)
    #latents = rnd.randn([batch_size, args.img_channels, gt_img.shape[-2], gt_img.shape[-1]], device=device)
    """
    logits_dir = "../logits/"
    if not os.path.exists(logits_dir):
        os.makedirs(logits_dir, exist_ok=True)
    torch.save({'latents': latents}, os.path.join("../logits/", "latents_0.pt"))
    """
    class_labels = None
    

    image_recon, stop_dc_iter  = general_forward_SDE_ps(y=ksp, y_prior_ksp=prior_ksp, img_prior = prior_img,  gt_img=gt_img, mri_inf_utils=mri_inf_utils, mri_prior_utils=mri_prior_utils , 
        task='mri', l_type='DPS', l_ss=args.l_ss, net=net, latents=latents, inference_snr_db=args.snr, class_labels=None, 
        randn_like=torch.randn_like, num_steps=args.num_steps, sigma_min=0.004, 
        sigma_max=args.sigma_max, rho=7, solver=args.solver, discretization=args.discretization,
        schedule='linear', scaling=args.scaling, epsilon_s=1e-3, C_1=0.001, C_2=0.008, 
        M=1000, alpha=1, S_churn=args.S_churn, S_min=0, S_max=float('inf'), S_noise=1, 
        verbose = True)

    cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[:,None] #shape: [1,1,H,W]

    cplx_recon=cplx_recon.detach().cpu().numpy()
    mean_recon=np.mean(cplx_recon,axis=0)[None]
    gt_img=gt_img.cpu().numpy()
    img_nrmse = nrmse_np(abs(gt_img[0,0]), abs(mean_recon[0,0]))
    img_SSIM = ssim(abs(gt_img[0,0]), abs(mean_recon[0,0]), data_range=abs(gt_img[0,0]).max() - abs(gt_img[0,0]).min())
    img_PSNR = psnr(gt=abs(gt_img[0,0]), est=abs(mean_recon[0]),max_pixel=np.amax(abs(gt_img)))

    print('Sample %d, seed %d, R: %d, NRMSE: %.3f, SSIM: %.3f, PSNR: %.3f'%(args.sample+1, args.seed, args.inference_R, img_nrmse, img_SSIM, img_PSNR))

    dict = { 
            'gt_img': gt_img,
            'recon': cplx_recon,
            'adj_img': adj_img.cpu().numpy(),
            'nrmse': img_nrmse,
            'ssim': img_SSIM,
            'psnr': img_PSNR,
            'stop_dc_iter': stop_dc_iter
    }

    # designate + create save directory
    results_dir = args.outdir + "/R=%d/snr%s/seed_%d"%(args.inference_R, args.inference_snr, args.seed)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    torch.save(dict, results_dir + "/sample_%d.pt"%args.sample)