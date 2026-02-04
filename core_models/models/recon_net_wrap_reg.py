import torch
import torch.nn as nn
from models.recon_net import ReconNet


class ViT(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:2'

        # ViT layer
        self.recon_net = ReconNet(net).to(self.device)
        
        # Load weights
        cp = torch.load('../checkpoints_trained_start/model_100.pt', map_location=self.device) 
        self.recon_net.load_state_dict(cp['model'])
        
        # Fusion layers
        self.epsilon = epsilon


    def printer(self, x):
        print("Current value of param1 during forward:", self.param1)
        return

    def forward(self, img,ref): 
        in_pad, wpad, hpad = self.recon_net.pad(img)

        input_norm,mean,std = self.recon_net.norm(in_pad.float())
  
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)
        
        # Recon Head
        head_out = self.recon_net.net.head(features)
        
        # Low Resolution 
        head_out_img = self.recon_net.net.seq2img(head_out, (180, 110))

        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        
        return im_out