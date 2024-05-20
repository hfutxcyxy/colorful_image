import torch
import torch.nn as nn
from torch.nn import functional as F

class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels // 2,kernel_size=3,stride=2,padding=1)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride, padding=1, groups=out_channels // 2)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.shuffle = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride, padding=1,
                      groups=out_channels // 2),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2)
        )

    def forward(self, x):
        #print(f"x's shape:{x.shape}")

        if self.stride == 2:
            out = self.conv(x)
        else:
            out = x

        x = torch.cat([self.shuffle(x),out], 1)
        #print(f"out/x's shape:{out.shape,x.shape}")
        return x

class improved_model2(nn.Module):
    def __init__(self):
        super(improved_model2, self).__init__()
        self.ab_norm = 110.
        self.conv1 = nn.Conv2d(1,32,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0)
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0)
        self.conv5 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0)
        self.conv8 = nn.ConvTranspose2d(313,313,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.relu = nn.LeakyReLU()
        self.model1 = ShuffleBlock(32, 32,stride=2)
        self.model2 = ShuffleBlock(64, 64,stride=2)
        self.model3 = ShuffleBlock(128, 128,stride=2)
        self.model4 = ShuffleBlock(256, 256,stride=2)
        self.model5 = ShuffleBlock(512, 512,stride=2)
        self.model6 = ShuffleBlock(512, 512,stride=2)
        self.model7 = ShuffleBlock(512, 512,stride=2)
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(313),
            nn.MaxPool2d(kernel_size=8, stride=8)
        )
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(313, 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def unnormalize_ab(self, in_ab):
        return in_ab*self.ab_norm

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        #print(f"model1's shape:{x.shape}")
        x = self.conv2(x)
        x = self.relu(x)
        #print(f"model2's shape:{x.shape}")
        x = self.conv3(x)
        x = self.relu(x)
        #print(f"model3's shape:{x.shape}")
        x = self.conv4(x)
        x = self.relu(x)
        #print(f"model4's shape:{x.shape}")
        x = self.conv5(x)
        x = self.model5(x)
        #print(f"model5's shape:{x.shape}")
        x = self.model8(x)
        x = self.conv8(x)
        #print(f"model8's shape:{x.shape}")
        out_reg = self.out(self.softmax(x))
        color_ab = self.upsample4(out_reg)
        return self.softmax(x), self.unnormalize_ab(color_ab)

def Colormodel():
    return improved_model2
