import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import ViT

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
class transUnet_3d(nn.Module):
    def __init__(self):
        super(transUnet_3d,self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.layer1_en = basix_3dConv_block_up(1,16)
        self.layer2_en = basix_3dConv_block_down(16,32)
        self.layer3_en = basix_3dConv_block_down(32,64)
        self.layer4_en = basix_3dConv_block_down(64,128)
        
        self.vit = ViT(in_channels=128, img_size=(6,24,24),patch_size=(2,2,2), pos_embed='conv', classification=False)
        
        self.layer3_de = basix_3dConv_block_down(128+64,64)
        self.layer2_de = basix_3dConv_block_down(64+32,32)
        self.layer1_de = basix_3dConv_block_down(32+16,16)
        
        self.upsample4 = nn.ConvTranspose3d(768,128,3,2,1,1)
        self.upsample3 = nn.ConvTranspose3d(128,128,3,2,1,1)
        self.upsample2 = nn.ConvTranspose3d(64,64,3,2,1,1)
        self.upsample1 = nn.ConvTranspose3d(32,32,3,2,1,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(16,1,1,1,0),
        )
        
        
    def forward(self, x):#1,48,192,192
        #encoder
        en1 = self.layer1_en(x)#16,48,192,192
        en2 = self.maxpool(en1)#16,24,96,96
        en2 = self.layer2_en(en2)#32,24,96,96
        en3 = self.maxpool(en2)#32,12,48,48
        en3 = self.layer3_en(en3)#64,12,48,48
        res = self.maxpool(en3)#64,6,24,24
        res = self.layer4_en(res)#128,6,24,24

        res,_ = self.vit(res)
        res = res.reshape(-1,768,3,12,12)
        
        res=self.upsample4(res)#128,6,24,24
        #decoder
        res = self.upsample3(res)#128,12,48,48
        res = torch.cat([en3,res],dim=1)#128+64,12,48,48
        res = self.layer3_de(res)#64,12,48,48
        
        res = self.upsample2(res)#64,24,96,96
        res = torch.cat([en2,res],dim=1)#64+32,24,96,96
        res = self.layer2_de(res)#32,24,96,96
        
        res = self.upsample1(res)#32,48,192,192
        res = torch.cat([en1,res],dim=1)#32+16,48,192,192
        res = self.layer1_de(res)#16,48,192,192
        
        res = self.final_conv(res)
        
        return res