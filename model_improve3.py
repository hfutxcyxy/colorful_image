import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // self.expansion, out_channels // self.expansion, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels // self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // self.expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class improved_model3(nn.Module):
    def __init__(self):
        super(improved_model3, self).__init__()
        self.ab_norm = 110.
        self.conv1 = nn.Conv2d(1,32,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0)
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0)
        self.conv5 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0)
        self.conv8 = nn.ConvTranspose2d(512,313,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.relu = nn.LeakyReLU()
        self.model1 = Bottleneck(32, 32,stride=1)
        self.model2 = Bottleneck(64, 64,stride=1)
        self.model3 = Bottleneck(128, 128,stride=1)
        self.model4 = Bottleneck(256, 256,stride=1)
        self.model5 = Bottleneck(512, 512,stride=1)
        #self.model6 = Bottleneck(512, 512,stride=1)
        #self.model7 = Bottleneck(512, 512,stride=1)
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
    return improved_model3
