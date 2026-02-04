"""
Written by Tal Oved for longitudinal data(2026)
"""

import os
import glob
import random
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from utils import transforms as T
from utils import complex_utils as cplx
from scipy.ndimage import rotate, shift
import sigpy as sp

class SliceData(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MR image slices.
    """

    def __init__(self, root, transform, sample_rate=1):
        """
        Args:
            root (pathlib.Path or str): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'reference', 'reference_kspace', and 'slice' as inputs. 'target' may be null
                for test data.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.transform = transform 
        self.examples = self._find_examples(root)

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_files]

    def _find_examples(self, root):
        # Find all files matching the pattern T1_week*reg.nii in the root_dir and subdirectories
        file_pattern = os.path.join(root, '**', 'T1_week*reg.nii')
        files = glob.glob(file_pattern, recursive=True)
        
        # Filter files to include only the relevant ones and their references
        examples = []
        for file in files:
            basename = os.path.basename(file)
            if 'T1_week' in basename:
                week_number = ''.join(filter(str.isdigit, basename.split('T1_week')[1].split('reg')[0]))
                #print(week_number)
                #print(file)
                reference_pattern = f"T1_week*regT1_week{week_number}.nii"
                reference_files = glob.glob(os.path.join(os.path.dirname(file), reference_pattern))
                #print(reference_files[0])
                if reference_files:
                    ref_file = reference_files[0]
                    data = nib.load(file).get_fdata()
                    num_slices = data.shape[2]
                    #examples += [(file, ref_file, slice_idx) for slice_idx in range(num_slices)]
                    ref_file = reference_files[0]
                    data = nib.load(file).get_fdata()
                    num_slices = data.shape[2]
                    middle_slice = num_slices // 2
                    for slice_idx in range(num_slices):
                       #if slice_idx - 16>= 0 and slice_idx + 19 < num_slices:  
                       if slice_idx - 4>= 0 and slice_idx + 4 < num_slices:
                        #print(slice_idx) 
                        #print(reference_pattern) 
                        examples += [(file, ref_file, slice_idx)]
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, ref_fname, slice_idx = self.examples[i]
        
        # Load the .nii files using nibabel
        data = nib.load(fname).get_fdata()
        ref_data = nib.load(ref_fname).get_fdata()
        
        target = data[:, :, slice_idx]
        
        
        ## For hard augmenting
        #print(f'Targert 1st: {target.shape}')
        random_angle = random.uniform(-20, 20) 
        target = rotate(target, angle=random_angle, axes=(0, 1), reshape=False)
        random_shift_x = random.uniform(-10, 10)  # Adjust the shift range as needed # was 20
        random_shift_y = random.uniform(-10, 10)  # Adjust the shift range as needed
        target = shift(target, shift=[random_shift_x, random_shift_y])
        #print(f'Targert 2nd: {target.shape}')
        
        
        random_phase = torch.angle(T.random_map((1,256,160), 'cpu',kspace_radius_range=(0.001, 0.001))) 
        target = target * (torch.exp(1j * random_phase)).numpy() 
        target = target.squeeze(0)
        target_torch = cplx.to_tensor(target).float() 
        
        ## Reduce contrast
        kspace_torch = T.fft2(target_torch)
        #kspace_torch = T.kspace_cut(kspace_torch,0.67,0.67)
        kspace = cplx.to_numpy(kspace_torch)
        #target_torch = T.ifft2(kspace_torch.clone())
        #target = cplx.to_numpy(target_torch)
        
        # Reference slice data
        ref_target = ref_data[:, :, slice_idx] * (torch.exp(1j * random_phase)).numpy() 
        ref_target = ref_target.squeeze(0)
        # Augment Reference
        
        ref_target = rotate(ref_target, angle=random_angle, axes=(0, 1), reshape=False)
        ref_target = shift(ref_target, shift=[random_shift_x, random_shift_y])
        #print(f'ref_target 2nd: {ref_target.shape}')
        

        ref_torch = cplx.to_tensor(ref_target).float() 
        ref_kspace_torch = T.fft2(ref_torch)
        #ref_kspace_torch = T.kspace_cut(ref_kspace_torch,0.67,0.67)
        ref_kspace = cplx.to_numpy(ref_kspace_torch)
        #ref_target_torch = T.ifft2(ref_kspace_torch)
        #ref_target = cplx.to_numpy(ref_target_torch)

        return self.transform( kspace, target, ref_kspace, ref_target, slice_idx)