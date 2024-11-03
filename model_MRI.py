# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):   # 展平成向量
    def forward(self, input):
        return input.view(input.size(0), -1)

class FCN(nn.Module):   # 全连接层
    def __init__(self, n_input, n_out):
        super(FCN, self).__init__()
        self.network = nn.Sequential(nn.Linear(n_input, n_input // 4),
                                     nn.Dropout(p=0.5),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(n_input // 4, n_out),
                                     nn.LeakyReLU(inplace=True))
    def forward(self,x):
        return self.network(x)


class upBlock(nn.Module):
    def __init__(self, n_input, n_out):
        super(upBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels=n_input, out_channels=n_input*2, kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(n_input*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=n_input*2, out_channels=n_out,kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(n_out),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.network(x)

class downBlock(nn.Module):
    def __init__(self, n_input, n_out):
        super(downBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels=n_input, out_channels=n_input // 2,  kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(n_input // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=n_input // 2, out_channels=n_out, kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(n_out),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.network(x)

class BottomBlock(nn.Module):
    def __init__(self, n_input, n_out):
        super(BottomBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels=n_input, out_channels=n_out, kernel_size=(3, 3, 3), padding=(1,1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(n_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)


class simpleUNet(nn.Module):  # 简易u-net
    def __init__(self, n_classes=2):
        super(simpleUNet, self).__init__()

        self.add_module("conv1", upBlock(1,16)) # n*n
        self.add_module("downS1",nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=(0,0,0)))
        self.add_module("conv2", upBlock(16,32)) # n-1 * n-1
        self.add_module("downS2", nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=(0,0,0)))

        self.add_module("convB", BottomBlock(32, 32))

        self.add_module("upS3", nn.Upsample(scale_factor=2, mode='nearest'))
        self.add_module("conv3", downBlock(64, 16))  # n-1 * n-1
        self.add_module("upS4", nn.Upsample(scale_factor=2, mode='nearest'))
        self.add_module("conv4", downBlock(32, 8))  # n * n
        # n # n

        # self.add_module('flat', Flatten())
        # self.add_module('fc', FCN(8*8*4, 8))
        self.add_module('output', nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=1,  kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1)),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        ))

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)  # torch的CNN函数只接受float32类型

        out_conv1 = self.conv1(x)
        x1 = self.downS1(out_conv1)

        out_conv2 = self.conv2(x1)
        x2 = self.downS2(out_conv2)
#
        out_convB = self.convB(x2)

        x3 = self.upS3(out_convB+x2)
        out_conv3 = self.conv3(torch.cat([x3,out_conv2],dim=1))

        x4 = self.upS4(out_conv3)
        out_conv4 = self.conv4(torch.cat([x4,out_conv1],dim=1))

        # avg = self.avg(out_conv4)
        # flat = self.flat(avg)
        # fc = self.fc(flat)

        logits = self.output(out_conv4)

        return logits
