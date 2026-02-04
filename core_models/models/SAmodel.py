import torch
import torch.nn as nn
import torch.nn.functional as F

################ No real use  ##################
################ CNN spattial ##################

class LeakyReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LeakyReLUConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.lrelu(self.conv(x))

class mRCAB(nn.Module):
    def __init__(self, in_channels):
        super(mRCAB, self).__init__()
        self.conv1 = LeakyReLUConv(in_channels, in_channels)
        self.conv2 = LeakyReLUConv(in_channels, in_channels)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        ca = self.ca(res)
        return x + res * ca

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(*[mRCAB(in_channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        return x + self.blocks(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=1):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class MyNetwork(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_residual_groups=3, num_blocks=5):
        super(MyNetwork, self).__init__()
        self.initial_conv = LeakyReLUConv(in_channels, 64)
        self.residual_groups1 = ResidualGroup(64, num_blocks)
        self.residual_groups2 = ResidualGroup(128, num_blocks)
        self.residual_groups3 = ResidualGroup(256, num_blocks)

        self.stride_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.stride_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.Conv2d(256, 128, kernel_size=1)
        self.up2 = nn.Conv2d(256, 64, kernel_size=1)


        self.mrcabs = mRCAB(128) #128
        self.spatial_attention = SpatialAttention()
        self.sub_pixel_conv = SubPixelConv(128, 64) #(128,64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.sub_pixel_conv_end = SubPixelConv(64, 64,upscale_factor=2)
        
    def forward(self, kspace, reference_image,init_image=None, mask=None,iter=0):

        # Get data dimensions
        dims = tuple(kspace.size())

        x = kspace

        ## input part1
        x = self.initial_conv(x)
        res1 = self.residual_groups1(x)
        res1_down = self.stride_conv1(res1)
        res2 = self.residual_groups2(res1_down)
        res2_down = self.stride_conv2(res2)
        res3 = self.residual_groups3(res2_down)

        res3_interp = F.interpolate(res3, scale_factor=2, mode='bilinear', align_corners=False)
        res3_interp = self.up1(res3_interp)

        res2_up1 = torch.cat([res3_interp, res2], dim=1)

        res3_interp2 = F.interpolate(res2_up1, scale_factor=2, mode='bilinear', align_corners=False)
        res3_interp2 = self.up2(res3_interp2)

        x = torch.cat([res3_interp2, res1], dim=1)
        x = self.mrcabs(x)
        x = self.spatial_attention(x) * x
        x = self.sub_pixel_conv(x)
        x = self.final_conv(x)

        return x 