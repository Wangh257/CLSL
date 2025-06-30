""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)


class UNet_Encoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        # self.up2 = (Up(512, 256 // factor, bilinear))
        # self.up3 = (Up(256, 128 // factor, bilinear))
        # self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        x1 = self.inc(x) #64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        x5 = self.down4(x4) #1024
        # import pdb;pdb.set_trace()
        # x = self.up1(x5, x4) #512
        # x = self.up2(x, x3) #256
        # x = self.up3(x, x2) #238
        # x = self.up4(x, x1) #64
        # logits = self.outc(x)
        x6 = self.avgpool(x5)
        x6 = x6.view(x6.size(0), -1)
        return x6



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        x1 = self.inc(x) #64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256
        x4 = self.down3(x3) #512
        x5 = self.down4(x4) #1024
        import pdb;pdb.set_trace()
        x = self.up1(x5, x4) #512
        x = self.up2(x, x3) #256
        x = self.up3(x, x2) #238
        x = self.up4(x, x1) #64
        logits = self.outc(x)
        return logits, [x2, x3, x4, x5]

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class MyUNet_Light_Singel_Encoder(nn.Module):
    def __init__(self, name='resnet50v2', mlp=None):
        super(MyUNet_Light_Singel_Encoder, self).__init__()
        self.mlp = mlp
        self.encoder = UNet_Encoder(4, 8)
        if self.mlp:
            dim_mlp = self.encoder.down4.out_channels
            self.fc_f = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, 128),
                Normalize(2)
            ) 
            self.fc_fenzi = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, 128),
                Normalize(2)
            )  
            self.fc_fenmu = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, 128),
                Normalize(2)
            ) 

    def forward(self, x_f, x_fenzi, x_fenmu):

        fea_f = self.encoder(x_f)
        fea_f = self.fc_f(fea_f)

        fea_fenzi = self.encoder(x_fenzi)
        fea_fenzi = self.fc_fenzi(fea_fenzi)

        fea_fenmu = self.encoder(x_fenmu)
        fea_fenmu = self.fc_fenmu(fea_fenmu)
        
        return fea_f, fea_fenzi, fea_fenmu


    



if __name__ == "__main__":
    model = UNet(1, 1)
    x = torch.randn(8, 1, 640, 544)
    logits = model(x)
    print(logits.size())
    print(model)
    # import pdb;pdb.set_trace()
