"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from utils import complex_utils as cplx
import math
import torch
from torch import nn
import torch.nn.functional as F
class SenseModel_single(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.
    """
    def __init__(self, coord=None, weights=None):
        super().__init__()
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        image = ifft2(self.weights * kspace)
        return image

    def _forward_op(self, image):
        kspace = self.weights * fft2(image)
        return kspace

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output

class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.
    """
    def __init__(self, maps, coord=None, weights=None):
        super().__init__()

        self.maps = maps

        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        image = ifft2(self.weights * kspace)
        image = cplx.mul(image.unsqueeze(-2), cplx.conj(self.maps))
        return image.sum(-3)

    def _forward_op(self, image):
        kspace = cplx.mul(image.unsqueeze(-3), self.maps)
        kspace = self.weights * fft2(kspace.sum(-2))
        return kspace

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output


class ArrayToBlocks(nn.Module):
    def __init__(self, block_size, image_shape, overlapping=False):
        """
        A module that extracts spatial patches from a 6D array with size [1, x, y, t, e, 2].
        Output is also a 6D array with size [N, block_size, block_size, t, e, 2].
        """
        super().__init__()

        # Get image / block dimensions
        self.block_size = block_size
        self.image_shape = image_shape
        _, self.nx, self.ny, self.nt, self.ne, _ = image_shape

        # Overlapping vs. non-overlapping block settings
        if overlapping:
            block_stride = self.block_size // 2
            # Use Hanning window to reduce blocking artifacts
            win1d = torch.hann_window(block_size, dtype=torch.float32) ** 0.5
            self.win = win1d[None,:,None,None,None,None] * win1d[None,None,:,None,None,None]
        else:
            block_stride = self.block_size
            self.win = torch.tensor([1.0], dtype=torch.float32)
            
            
        # Figure out padsize (to avoid black bars)
        num_blocks_x = (self.nx // self.block_size) + 2
        num_blocks_y = (self.ny // self.block_size) + 2
        self.pad_x = (self.block_size*num_blocks_x - self.nx) // 2
        self.pad_y = (self.block_size*num_blocks_y - self.ny) // 2
        nx_pad = self.nx + 2*self.pad_x
        ny_pad = self.ny + 2*self.pad_y

        # Compute total number of blocks
        num_blocks_x = (self.nx-self.block_size+2*self.pad_x) / block_stride + 1
        num_blocks_y = (self.ny-self.block_size+2*self.pad_y) / block_stride + 1
        self.num_blocks = int(num_blocks_x * num_blocks_y)

        # Set fold params
        self.fold_params = dict(kernel_size=2*(block_size,), stride=block_stride)
        self.unfold_op = nn.Unfold(**self.fold_params)
        self.fold_op = nn.Fold(output_size=(ny_pad, nx_pad), **self.fold_params)

    def extract(self, images):
        # Re-shape into a 4D array because nn.Unfold requires it >:(
        images = images.reshape([1, self.nx, self.ny, self.nt*self.ne*2]).permute(0,3,2,1)

        # Pad array
        images = nn.functional.pad(images, 2*(self.pad_x,) + 2*(self.pad_y,), mode='constant')

        # Unfold array into vectorized blocks
        blocks = self.unfold_op(images) # [1, nt*ne*2*bx*by, n]

        # Reshape into 2D blocks
        shape_out = (self.nt, self.ne, 2, self.block_size, self.block_size, self.num_blocks)
        blocks = blocks.reshape(shape_out).permute(5,4,3,0,1,2)

        # Apply window
        blocks *= self.win.to(images.device)

        return blocks

    def combine(self, blocks):
        # Apply window 
        blocks *= self.win.to(blocks.device)

        # Reshape back into nn.Fold format
        blocks = blocks.permute(3,4,5,2,1,0)
        blocks = blocks.reshape((1, self.nt*self.ne*2*self.block_size**2, self.num_blocks))

        # Fold blocks back into array
        images = self.fold_op(blocks) # [1, nt*ne*2, ny, nx]

        # Crop zero-padded images
        images = center_crop(images.permute(0,3,2,1), [1, self.nx, self.ny, self.nt*self.ne*2])
        images = images.reshape(self.image_shape)

        return images

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self.combine(input)
        else:
            output = self.extract(input)
        return output


def decompose_LR(images, num_basis, block_size=16, overlapping=False, block_op=None):
    """
    Decomposes spatio-temporal data into spatial and temporal basis functions (L, R)
    """
    # Get image dimensions
    _, nx, ny, nt, ne, _ = images.shape
    nb = num_basis

    # Initialize ArrayToBlocks op if it hasn't been initialized already
    if block_op is None:
        block_op = ArrayToBlocks(block_size, images.shape, overlapping=overlapping)

    # Extract spatial blocks across images
    blocks = block_op(images)
    nblks = blocks.shape[0] # number of blocks
    blk_size = blocks.shape[1] # block shape [blk_size, blk_size]

    # Reshape into batch of 2D matrices
    blocks = blocks.permute(0,1,2,4,3,5)
    blocks = blocks.reshape((nblks, blk_size*blk_size*ne, nt, 2))

    # Perform SVD to get left and right singular vectors for each patch
    U, S, V = cplx.svd(blocks, compute_uv=True)

    # Truncate singular values and vectors
    U = U[:, :, :nb, :] # [N, Px*Py*E, B, 2]
    S = S[:, :nb]       # [N, B]
    V = V[:, :, :nb, :] # [N, T, B, 2]

    # Combine and reshape matrices
    S_sqrt = S.reshape((nblks, 1, 1, 1, 1, nb, 1)).sqrt()
    L = U.reshape((nblks, blk_size, blk_size,  1, ne, nb, 2)) * S_sqrt
    R = V.reshape((nblks,        1,        1, nt,  1, nb, 2)) * S_sqrt

    return L, R


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension containing
            real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    ndims = len(list(data.size()))

    if ndims == 5:
        data = data.permute(0,3,1,2,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('fft2: ndims > 6 not supported!')

    data = ifftshift(data, dim=(-3, -2))
    data_complex = torch.complex(data[...,0], data[...,1])
    data_fft = torch.fft.fftn(data_complex,dim=(-2, -1), norm="ortho")
    data = torch.stack((data_fft.real, data_fft.imag), dim=-1)
    data = fftshift(data, dim=(-3, -2))

    if ndims == 5:
        data = data.permute(0,2,3,1,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('fft2: ndims > 6 not supported!')

    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension containing
            real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    ndims = len(list(data.size()))

    if ndims == 5:
        data = data.permute(0,3,1,2,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('ifft2: ndims > 6 not supported!')

    data = ifftshift(data, dim=(-3, -2))
    data_complex = torch.complex(data[...,0], data[...,1])
    data_ifft = torch.fft.ifftn(data_complex,dim=(-2, -1), norm="ortho")
    data = torch.stack((data_ifft.real, data_ifft.imag), dim=-1)
    data = fftshift(data, dim=(-3, -2))

    
    if ndims == 5:
        data = data.permute(0,2,3,1,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('ifft2: ndims > 6 not supported!')

    return data


def root_sum_of_squares(x, dim=0):
    """
    Compute the root sum-of-squares (RSS) transform along a given dimension of a complex-valued tensor.
    """
    assert x.size(-1) == 2
    return torch.sqrt((x ** 2).sum(dim=-1).sum(dim))


def time_average(data, dim, eps=1e-6, keepdim=True):
    """
    Computes time average across a specified axis.
    """
    mask = cplx.get_mask(data)
    return data.sum(dim, keepdim=keepdim) / (mask.sum(dim, keepdim=keepdim) + eps)


def sliding_window(data, dim, window_size):
    """
    Computes sliding window with circular boundary conditions across a specified axis.
    """
    assert 0 < window_size <= data.shape[dim]

    windows = [None] * data.shape[dim]
    for i in range(data.shape[dim]):
        data_slide = roll(data, int(window_size/2)-i, dim)
        window = data_slide.narrow(dim, 0, window_size)
        windows[i] = time_average(window, dim)

    return torch.cat(windows, dim=dim)


def center_crop(data, shape):
    """
    Apply a center crop to a batch of images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. 
        shape (list of ints): The output shape. If shape[dim] = -1, then no crop 
            will be applied in that dimension.
    """
    for i in range(len(shape)):
        if (shape[i] == data.shape[i]) or (shape[i] == -1):
            continue
        assert 0 < shape[i] <= data.shape[i]
        idx_start = (data.shape[i] - shape[i]) // 2
        data = data.narrow(i, idx_start, shape[i])

    return data


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def add_rician_noise_torch(complex_data, v, sg):
    """
    Add Rician noise to complex data presented as a two-channel PyTorch tensor.

    Parameters:
    complex_data (torch.Tensor): The complex data tensor with shape (2, H, W),
                                 where the first channel is the real part and
                                 the second channel is the imaginary part.
    v (float): The noise variance parameter.
    sg (float): The standard deviation of the Gaussian noise.

    Returns:
    torch.Tensor: Complex data with added Rician noise.
    """
    # Ensure the data is on the CPU and convert to NumPy for processing
    real_part = complex_data[...,0].numpy()
    imag_part = complex_data[...,1].numpy()
    
    # Generate Gaussian noise
    noise_real = v / np.sqrt(2) + sg * np.random.randn(*real_part.shape)
    noise_imag = v / np.sqrt(2) + sg * np.random.randn(*imag_part.shape)
    
    # Add noise to real and imaginary parts
    noisy_real = real_part + noise_real
    noisy_imag = imag_part + noise_imag
    
    # Combine noisy real and imaginary parts
    noisy_complex_data = np.stack((noisy_real, noisy_imag), axis=-1)

    # Convert back to PyTorch tensor
    noisy_complex_data = torch.from_numpy(noisy_complex_data)
    
    return noisy_complex_data


def awgn_torch(sig,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
        """
    real_part = sig[...,0].numpy()
    imag_part = sig[...,1].numpy()
    s = real_part + 1j*imag_part
    gamma = 10**(-SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/np.size(s)#np.size(s) # if s is a matrix [MxN]
    N0=P*gamma # Find the noise spectral density
    if np.isrealobj(s):# check if input is real/complex object type
        n = np.sqrt(N0)*np.random.randn(s.shape[0],s.shape[1]) # computed noise
    else:
        n = np.sqrt(N0/2)*(np.random.randn(s.shape[0],s.shape[1])+1j*np.random.randn(s.shape[0],s.shape[1]))
    r = s + n # received signal

    noisy_complex_data = np.stack((r.real, r.imag), axis=-1)
    noisy_complex_data = torch.from_numpy(noisy_complex_data)

    return noisy_complex_data
    

def zero_pad_kspace(kspace_tensor, target_shape):
    """
    Pads a k-space tensor with zeros symmetrically to match a target shape.
    Assumes k-space data is centered (DC component in the middle).
    
    Args:
        kspace_tensor: Input tensor (e.g., [Batch, Coil, Height, Width, Complex])
        target_shape: Tuple of (Height, Width) you want to pad to.
    """
    # Get current height and width (assuming last dim is complex, so H, W are -3 and -2)
    # Adjust indices based on your exact shape: [..., H, W, Complex] vs [..., H, W]
    current_h, current_w = kspace_tensor.shape[-3], kspace_tensor.shape[-2] 
    target_h, target_w = target_shape
    
    # Calculate total padding needed
    diff_h = target_h - current_h
    diff_w = target_w - current_w
    
    if diff_h < 0 or diff_w < 0:
        raise ValueError("Target shape must be larger than input shape.")

    # Calculate padding for each side (Left, Right, Top, Bottom)
    # We use floor/ceil to handle odd differences correctly
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left
    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    
    # F.pad expects inputs in reverse order of dimensions: (last_dim_pad, 2nd_last_dim_pad, ...)
    # If shape is (..., H, W, Complex), we pad W and H.
    # We skip the Complex dim (pad=0,0) if it exists as the last dimension.
    
    # Option A: If tensor shape is [..., H, W, 2] (Standard Complex Tensor)
    padding = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
    
    # Option B: If tensor shape is [..., H, W] (Complex numbers already handled or magnitude)
    # padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    padded_kspace = F.pad(kspace_tensor, padding, mode='constant', value=0)
    
    return padded_kspace


def awgn(sig,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
        """
    s = sig
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/np.size(s)#np.size(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if np.isrealobj(s):# check if input is real/complex object type
        n = np.sqrt(N0)*np.random.randn(s.shape[0],s.shape[1]) # computed noise
    else:
        n = np.sqrt(N0/2)*(np.random.randn(s.shape[0],s.shape[1])+1j*np.random.randn(s.shape[0],s.shape[1]))
    r = s + n # received signal

    return r

def reduce_contrast(image, factor=0.5):
    """
    Reduce the contrast of a float image.

    Parameters:
    - image: numpy array, input image (float32 or float64).
    - factor: float, factor by which to reduce the contrast (0 < factor < 1).

    Returns:
    - numpy array, image with reduced contrast.
    """
    # Ensure the image is in float32 format
    image = image.astype(np.float32)

    # Calculate the mean intensity of the image
    mean_intensity = np.mean(image)

    # Apply the contrast reduction formula
    reduced_contrast_image = mean_intensity + factor * (image - mean_intensity)

    return reduced_contrast_image


def PSNR(input, target):
    eps = 1e-8
    return -10*torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3])+eps)


def PSNR_numpy(input, target):
    img1 = target * 255
    img2 = input * 255
    mse = np.mean(np.square(np.subtract(img1.astype(np.int16),
                                        img2.astype(np.int16))))
    if mse == 0:
        return np.Inf
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX) - 10 * math.log10(mse)  


def kspace_crop(tensor, r):
    """
    Zero out the edges of a tensor in both spatial dimensions by a factor of r.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, A, B).
        r (float): Factor to determine the width of the edge to zero out. 
                   r should be between 0 and 1, where 1 means no edges are zeroed, 
                   and 0 means the entire tensor is zeroed out.
    
    Returns:
        torch.Tensor: Tensor with edges zeroed out by factor r, same size as input.
    """
    assert 0 <= r <= 1, "Factor r should be between 0 and 1."

    A, B, C = tensor.shape
    a_margin = int(A * (1 - r) / 2)
    b_margin = int(B * (1 - r) / 2)
    
    # Create a copy of the tensor to avoid modifying the original tensor
    tensor_cropped = tensor.clone()
    
    # Zero out edges
    tensor_cropped[:a_margin, :,:] = 0
    tensor_cropped[-a_margin:, :,:] = 0
    tensor_cropped[:, :b_margin,:] = 0
    tensor_cropped[:, -b_margin:,:] = 0

    return tensor_cropped

def kspace_cut(tensor, r1,r2):
    """
    Zero out the edges of a tensor in both spatial dimensions by a factor of r.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, A, B).
        r (float): Factor to determine the width of the edge to zero out. 
                   r should be between 0 and 1, where 1 means no edges are zeroed, 
                   and 0 means the entire tensor is zeroed out.
    
    Returns:
        torch.Tensor: Tensor with edges zeroed out by factor r, same size as input.
    """
    assert 0 <= r1 <= 1, "Factor r should be between 0 and 1."
    assert 0 <= r2 <= 1, "Factor r should be between 0 and 1."

    A, B, C = tensor.shape
    a_margin = int(A * (1 - r1) / 2)
    b_margin = int(B * (1 - r2) / 2)

    
    # Zero out edges
    tensor_out = tensor[a_margin:-a_margin,b_margin:-b_margin,:]

    return tensor_out

def ifft2c(tensor, dim=(-2, -1)):
    tensor = torch.fft.ifftshift(tensor, dim=dim)
    tensor = torch.fft.ifft2(tensor, dim=dim)
    return torch.fft.fftshift(tensor, dim=dim)

def random_map(shape, device, kspace_radius_range=(0.001, 0.05), crop_middle=True, normalize=True):
        batch_size, height, width = shape

        # kspace weight
        x, y = torch.meshgrid(torch.linspace(-0.5, 0.5, height, device=device),
                              torch.linspace(-0.5, 0.5, width, device=device),indexing='ij')
        r = torch.sqrt(x ** 2 + y ** 2)
        kspace_parameter = 1 / torch.FloatTensor(batch_size).uniform_(*kspace_radius_range).to(device)

        kspace = 2 * torch.rand(shape, dtype=torch.cfloat, device=device) - 1 - 1j  # Channels first
        kspace = kspace * torch.exp(-r.type(torch.cfloat) * kspace_parameter[:, None, None])
        return  ifft2c(kspace)

############### genPDF  ###########################
def genPDF(imSize, p, pctg, distType=2, radius=0, disp=0, pdf_show_flag=0):
    # This function generates a pdf for a 1d or 2d random sampling pattern
    # with polynomial variable density sampling

    # Input:
    # imSize - size of matrix or vector
    # p - power of polynomial
    # pctg - partial sampling factor e.g. 0.5 for half
    # distType - 1 or 2 for L1 or L2 distance measure
    # radius - radius of fully sampled center
    # disp - display output

    # Output:
    # pdf - the pdf
    # val - min sampling density

    # (c) Michael Lustig 2007. Converted from Matlab to Python by Efrat Shimron (2020)

    minval = 0
    maxval = 1
    val = 0.5

    if len(imSize) == 2:  # 2D case

        sx = imSize[0]
        sy = imSize[1]
        PCTG = np.floor(pctg * sx * sy)

        x_co = np.linspace(-1, 1, sy)  # coordinates
        y_co = np.linspace(-1, 1, sx)  # coordinates
        x, y = np.meshgrid(x_co, y_co)

        if distType == 1:
            r = np.max(np.abs(x), np.abs(y))
        else:
            r = np.sqrt(x ** 2 + y ** 2)
            r = r / np.max(np.abs(r.reshape(1, -1)))

    elif len(imSize) == 1:  # 1D case
        sx = imSize[0]
        PCTG = np.floor(pctg * sx)
        r = np.abs(np.linspace(-1, 1, sx))

        # create PDF
    idx = np.where(r < radius)
    pdf = (1 - r) ** p + val
    pdf[pdf > 1] = 1
    pdf[idx] = 1

    if pdf_show_flag == 1:
        # pdf display
        # 2D case
        if len(pdf.shape) == 2:
            plt.imshow(pdf, cmap="gray")
            plt.show()
        # 1D case
        elif len(pdf.shape) == 1:
            plt.plot(pdf)
            plt.show()

    assert np.floor(np.sum(pdf)) > PCTG, "infeasible sampling, fully-sampled DC area is too large, increase poly_degree"

    while (1):
        val = minval / 2 + maxval / 2
        pdf = (1 - r) ** p + val
        pdf[pdf > 1] = 1
        pdf[idx] = 1
        N = np.floor(np.sum(pdf))
        if N > PCTG:  # infeasible
            maxval = val
        if N < PCTG:  # feasible, but not optimal
            minval = val
        if N == PCTG:  # optimal
            break

    return pdf


############### genSampling  ###########################


def genSampling(pdf, iter, tol, calib=[1, 1]):
    #  A monte-carlo algorithm to generate a sampling pattern with
    #  minimum peak interference. The number of samples will be
    #  sum(pdf) +- tol
    #
    # Inputs:
    #   pdf - probability density function to choose samples from
    #   iter - vector of min interferences measured each try
    #   tol  - the deviation from the desired number of samples in samples
    # Outputs:
    #  mask - sampling pattern

    # (c) Michael Lustig 2007.
    # Converted from Matlab to Python by Efrat Shimron (2020)

    # print('calib=', calib)

    pdf[pdf > 1] = 1
    K = np.sum(pdf[::])
    minIntr = np.array([1e99])
    minIntrVec = np.zeros(pdf.shape)

    #np.random.seed(seed=None)  # remove this line to generate the same mask each time

    # 2D case
    if len(pdf.shape) == 2:
        for n in range(iter):
            tmp = np.zeros(pdf.shape)
            while np.abs(np.sum(tmp[::]) - K) > tol:
                tmp = np.random.random(pdf.shape) < pdf

            TMP = np.fft.ifft2(tmp / pdf)
            if np.max(np.abs(TMP[1:-1])) < minIntr:
                minIntr = np.max(np.abs(TMP[1:-1]))
                minIntrVec = tmp

        mask = minIntrVec

        # add calibration area
        nx = mask.shape[-1]
        ny = mask.shape[-2]

        mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
        int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

    # 1D case
    elif len(pdf.shape) == 1:
        for n in range(iter):
            tmp = np.zeros(pdf.shape)
            while np.abs(np.sum(tmp[::]) - K) > tol:
                tmp = np.random.random(pdf.shape) < pdf

            TMP = np.fft.ifft(tmp / pdf)
            if np.max(np.abs(TMP[1:-1])) < minIntr:
                minIntr = np.max(np.abs(TMP[1:-1]))
                minIntrVec = tmp

        mask = minIntrVec

        # add calibration area
        nx = mask.shape[0]

        mask[int(nx / 2 - calib / 2): int(nx / 2 + calib / 2)] = 1

    return mask


#################### 1d VD mask ############################
def gen_1D_var_dens_mask(R,imSize,sampling_flag,calib=[24, 24]):
    NX = imSize[0]
    NY = imSize[1]

    if sampling_flag == 'random':
        # 1D random - sampling columns
        tmp = np.random.randint(1, 1000, NX)
        inds = np.where(tmp <= (1000 / R))
        mask = np.zeros((NX, NY))
        mask[:, inds] = 1

        # fig = plt.figure()
        # plt.imshow(np.abs(mask), cmap="gray")
        # plt.title('mask_1D')
        # plt.show()


    if R >= 8:
        poly_degree = 10
    elif R == 7:
        poly_degree = 8
    elif R == 6:
        if sampling_flag == 'weak':
            poly_degree = 12
        elif sampling_flag == 'strong':
            poly_degree = 6
    elif R == 5:
        if sampling_flag == 'weak':
            poly_degree = 5
        # elif sampling_flag == 'strong':
        # TODO: implement this
    elif R == 4:
        if sampling_flag == 'weak':
            poly_degree = 10
        elif sampling_flag == 'strong':
            poly_degree = 4
            # print('poly_degree=', poly_degree)
    elif R == 3:
        poly_degree = 4
    elif R == 2:
        poly_degree = 4

    # # create a 1D pdf
    imSize_1D = np.array([imSize[1]])
    pdf_1D = genPDF(imSize_1D, poly_degree, 1 / R)
    mask_1D = genSampling(pdf_1D, iter=10, tol=10, calib=calib)

    pdf_1D = np.expand_dims(pdf_1D,0)
    mask_1D = np.expand_dims(mask_1D,0)

    pdf = np.repeat(pdf_1D,NY,0)
    mask = np.repeat(mask_1D,NY,0)

    return mask, pdf, poly_degree


###################################################################################################
#                                             2D var-dens (from 1c_MoDL_OLD_1D_sampling run10)
###################################################################################################
def gen_2D_var_dens_mask(R, imSize, sampling_flag, calib=[24, 24]):
    NX = imSize[0]
    NY = imSize[1]

    if sampling_flag == 'random':
        # 2D random-uniform
        tmp = np.random.randint(1, 1000, (NX, NY))
        inds = np.where(tmp <= (1000 / R))
        mask = np.zeros([NX, NY])
        mask[inds] = 1



        fig = plt.figure()
        plt.imshow(np.abs(mask), cmap="gray")
        plt.title('mask_1D')
        plt.show()

        pdf = (1 / R) * np.ones(imSize)
        poly_degree = 'NaN'

    else:
        if R == 12:
            if sampling_flag == 'weak':
                poly_degree = 15
            elif sampling_flag == 'strong':
                poly_degree = 9
        elif R == 8:
            if sampling_flag == 'weak':
                poly_degree = 12
            elif sampling_flag == 'strong':
                poly_degree = 7
        elif R == 6:
            if sampling_flag == 'weak':
                poly_degree = 12
            elif sampling_flag == 'strong':
                poly_degree = 5
        elif R == 5:
            if sampling_flag == 'weak':
                poly_degree = 12
            elif sampling_flag == 'strong':
                poly_degree = 4.5
        elif R == 4:
            if sampling_flag == 'weak':
                poly_degree = 7
            elif sampling_flag == 'strong':
                poly_degree = 3
        elif R == 3:
            if sampling_flag == 'weak':
                poly_degree = 7
            elif sampling_flag == 'strong':
                poly_degree = 2
        elif R == 2:
            if sampling_flag == 'weak':
                poly_degree = 7
            elif sampling_flag == 'strong':
                poly_degree = 1

    
        # create a 2D pdf
        pdf = genPDF(imSize, poly_degree, 1 / R)
        mask = genSampling(pdf, iter=50, tol=20, calib=calib)

    return mask, pdf, poly_degree

