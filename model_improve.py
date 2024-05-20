import torch
import torch.nn as nn

class ResidualLayer(nn.Module):
    def __init__(self,in_channels):
        super(ResidualLayer,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class improved_model(nn.Module):
    def __init__(self):
        super(improved_model,self).__init__()
        self.ab_norm = 110

        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv6 = nn.Conv2d(256, 313, kernel_size=1, padding=1,dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU()
        self.ResLayer = ResidualLayer(512)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313,2,kernel_size=1,padding=0,dilation=1,stride=1,bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.ResidualBlock = nn.Sequential(
            self.ResLayer,
            self.ResLayer,
            self.ResLayer,
            self.ResLayer,
            self.ResLayer,
            self.ResLayer
        )

    def unnormalize_ab(self,in_ab):
        return in_ab*self.ab_norm

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.bn4(out)
        out = self.ResidualBlock(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out_reg = self.model_out(self.softmax(out))
        color_ab = self.upsample4(out_reg)

        return self.softmax(out),self.unnormalize_ab(color_ab)

def Colormodel():
    return improved_model
