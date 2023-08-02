import torch
import torch.nn as nn
import numpy as np

def basix_3dConv_block_down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, 3, padding=1),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )
def basix_3dConv_block_up(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )
class unet_3d(nn.Module):
    def __init__(self):
        super(unet_3d,self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.layer1_en = basix_3dConv_block_up(1,32)
        self.layer2_en = basix_3dConv_block_down(32,64)
        self.layer3_en = basix_3dConv_block_down(64,128)
        self.layer4_en = basix_3dConv_block_down(128,256)
        
        self.layer3_de = basix_3dConv_block_down(128+128,128)
        self.layer2_de = basix_3dConv_block_down(64+64,64)
        self.layer1_de = basix_3dConv_block_down(32+32,32)
        
        self.upsample3 = nn.ConvTranspose3d(256,128,3,2,1,1)
        self.upsample2 = nn.ConvTranspose3d(128,64,3,2,1,1)
        self.upsample1 = nn.ConvTranspose3d(64,32,3,2,1,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(32,1,1,1,0),
        )
    def forward(self, x):#1,48,200,200
        #encoder
        en1 = self.layer1_en(x)#32,48,200,200
        en2 = self.maxpool(en1)#32,24,100,100
        en2 = self.layer2_en(en2)#64,24,100,100
        en3 = self.maxpool(en2)#64,12,50,50
        en3 = self.layer3_en(en3)#128,12,50,50
        res = self.maxpool(en3)#128,6,25,25
        res = self.layer4_en(res)#256,6,25,25
        #decoder
        res = self.upsample3(res)#128,12,50,50
        res = torch.cat([en3,res],dim=1)#128+128,12,50,50
        res = self.layer3_de(res)#128,12,50,50
        
        res = self.upsample2(res)#64,24,100,100
        res = torch.cat([en2,res],dim=1)#64+64,24,100,100
        res = self.layer2_de(res)#64,24,100,100
        
        res = self.upsample1(res)#32,48,100,100
        res = torch.cat([en1,res],dim=1)#32+32,48,100,100
        res = self.layer1_de(res)#32,48,100,100
        
        res = self.final_conv(res)
        
        return res
    
# from torchsummary import summary

# summary(unet_3d(), (1,48,200,200))