"""
All rights reserved.
Tal Oved, 2026.
"""

import os, sys
import torch
from torch import nn
import utils.complex_utils as cplx
from utils.transforms import SenseModel,SenseModel_single

from models.SAmodel import MyNetwork
#from image_fusion import fuse
import os
from models.recon_net_wrap import ViTfuser
from models.vision_transformer import VisionTransformer

class Operator(torch.nn.Module):
    def __init__(self, A):
        super(Operator, self).__init__()
        self.operator = A

    def forward(self, x):
        return self.operator(x)

    def adjoint(self, x):
        return self.operator(x, adjoint=True)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        return out

class ViTFuserWrap(nn.Module):
    """
    PyTorch implementation of ViT-Fuser.
    """

    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()

        # Extract network parameters
        self.num_grad_steps = params.num_grad_steps 
        self.num_cg_steps = params.num_cg_steps
        self.share_weights = params.share_weights
        self.modl_lamda = params.modl_lamda
        self.reference_mode = params.reference_mode
        self.reference_lambda = params.reference_lambda
        self.device = 'cuda:3'

        net = VisionTransformer(
        avrg_img_size=320,
        patch_size = (10,10),
        in_chans=1,
        embed_dim=64,
        depth=10,
        num_heads=16
        )

        self.resnets = nn.ModuleList([MyNetwork(2,2)] * self.num_grad_steps)
        self.similaritynets = nn.ModuleList([ViTfuser(net)] * self.num_grad_steps)


    def freezer():
        for net in self.similaritynets:
            net.recon_net.net.forward_features.requires_grad_(False)
            net.recon_net.net.head.requires_grad_(False)
        

    def forward(self, kspace, reference_image,init_image=None, mask=None):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape [batch_size, height, width, time, num_coils, 2]
            maps (torch.Tensor): Input tensor of shape   [batch_size, height, width,    1, num_coils, num_emaps, 2]
            mask (torch.Tensor): Input tensor of shape   [batch_size, height, width, time, 1, 1]

        Returns:
            (torch.Tensor): Output tensor of shape       [batch_size, height, width, time, num_emaps, 2]
        """
        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())
        
        # Declare signal model
        A = SenseModel_single(weights=mask)
        Sense = Operator(A)
        # Compute zero-filled image reconstruction
        zf_image = Sense.adjoint(kspace)

        
        # Reference dealing
        reference_image = reference_image.permute(0,3,1,2) 

        real_part_ref = reference_image[:,0,:,:].unsqueeze(1)
        imag_part_ref = reference_image[:,1,:,:].unsqueeze(1)
        mag_ref = torch.sqrt(real_part_ref**2 + imag_part_ref**2)


        image = zf_image 

        iter = 1
        
        # Run ViT-Fuser 1 time
        for resnet, similaritynet in zip(self.resnets, self.similaritynets):

            image = image.permute(0,3,1,2)
 
            real_part = image[:,0,:,:].unsqueeze(1)
            imag_part = image[:,1,:,:].unsqueeze(1)


            phase = torch.atan2(real_part,imag_part)
            mag_image = torch.sqrt(real_part**2 + imag_part**2)

            #print(f"DEBUG: mag_image shape is {mag_image.shape}") 

            refined_image = similaritynet(mag_image,mag_ref)

            image = torch.cat((refined_image*torch.cos(phase),refined_image*torch.sin(phase)),dim=1)

            if self.reference_mode == 1: # For training loop
                image = refined_image

            image = image.permute(0, 2, 3, 1) # Permute back to original shape


            iter = iter +1

        return image