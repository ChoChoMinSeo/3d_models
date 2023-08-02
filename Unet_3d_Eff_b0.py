from efficientnet_pytorch_3d import EfficientNet3D
import torch
import torch.nn as nn
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
        self.backbone = EfficientNet3D.from_name('efficientnet-b0',override_params={'num_classes':2},in_channels=1)
        self.backbone._blocks[2]= nn.Sequential(
            nn.Conv3d(24,144,(1,1,1),(1,1,1)),
            nn.BatchNorm3d(144),
            nn.Conv3d(144,144,(3,2,2),(1,1,1),(1,0,0)),
            nn.BatchNorm3d(144),
            nn.Conv3d(144,6,(1,1,1),(1,1,1)),
            nn.Conv3d(6,144,(1,1,1),(1,1,1)),
            nn.Conv3d(144,24,(1,1,1),(1,1,1)),
            nn.BatchNorm3d(24),
            nn.SiLU()
        )
        self.increase_size_conv = nn.Sequential(
            nn.Conv3d(32,32,(1,2,2),(1,1,1),(0,1,1))
        )
        self.dconv_down0 = basix_3dConv_block_down(40, 48)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up3 = basix_3dConv_block_up(64 + 32, 64)
        self.dconv_up2 = basix_3dConv_block_up(32 + 16, 64)
        self.dconv_up1 = basix_3dConv_block_up(48 + 24, 32)
        self.conv_last = nn.Sequential(
            basix_3dConv_block_up(64,64),
            nn.Conv3d(64, 1, 1)
        )
    def forward(self, x):#1,48,200,200
        x = self.backbone._conv_stem(x)#32x24x100x100
        x = self.backbone._bn0(x)

        conv0 = self.backbone._blocks[0](x)#16x12x50x50

        conv1 = self.backbone._blocks[1](conv0)#24x6x25x25
        conv1 = self.backbone._blocks[2](conv1)#24x6x24x24

        conv2 = self.backbone._blocks[3](conv1)#40x3x12x12
        conv2 = self.backbone._blocks[4](conv2)#40x3x12x12

        up2 = self.dconv_down0(conv2) #48x3x12x12

        up2 = self.upsample(up2)#48x6x24x24
        up2 = torch.cat([conv1,up2],dim=1)#48+24x6x24x24

        up2 = self.dconv_up1(up2)# 32x6x24x24
        up2 = self.increase_size_conv(up2)# 32x6x25x25
        up2 = self.upsample(up2)# 32x12x50x50

        up2 = torch.cat([conv0,up2],dim=1)#32+16x12x50x50
        up2 = self.dconv_up2(up2)#64x12x50x50
        up2 = self.upsample(up2)#64x24x100x100

        up2 = torch.cat([x,up2],dim=1)#64+32x24x100x100
        up2 = self.dconv_up3(up2)#64x24x100x100
        up2 = self.upsample(up2)#64x48x200x200

        up2 =self.conv_last(up2)
        return up2