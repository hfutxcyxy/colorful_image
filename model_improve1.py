import torch
import torch.nn as nn

class RevBlock(nn.Module):
    def __init__(self,  out_channels, stride=1):
        super(RevBlock, self).__init__()
        self.conv1 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels // 2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.bn(self.relu(self.conv2(self.relu(self.conv1(x2)))))
        y2 = x2
        return torch.cat([y1, y2], dim=1)

class improved_model1(nn.Module):
    def __init__(self):
        super(improved_model1, self).__init__()
        self.ab_norm = 110.
        self.conv1 = nn.Conv2d(1,32,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(32,64,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(64,128,kernel_size=1,stride=1,padding=0)
        self.conv4 = nn.Conv2d(128,256,kernel_size=1,stride=1,padding=0)
        self.conv5 = nn.Conv2d(256,512,kernel_size=1,stride=1,padding=0)
        self.model1 = RevBlock(32)
        self.model2 = RevBlock(64)
        self.model3 = RevBlock(128)
        self.model4 = RevBlock(256)
        self.model5 = RevBlock(512)
        self.model6 = RevBlock(512)
        self.model7 = RevBlock(512)
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
        x = self.model1(x)
        x = self.conv2(x)
        x = self.model2(x)
        x = self.conv3(x)
        x = self.model3(x)
        x = self.conv4(x)
        x = self.model4(x)
        x = self.conv5(x)
        x = self.model5(x)
        x = self.model8(x)
        out_reg = self.out(self.softmax(x))
        color_ab = self.upsample4(out_reg)
        return self.softmax(x), self.unnormalize_ab(color_ab)

def Colormodel():
    return improved_model1

