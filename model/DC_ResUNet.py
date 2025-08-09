import torch
import torch.nn as nn


class DualChannelBlock(nn.Module):
    """双通道CNN块"""

    def __init__(self, in_channels, block_channels, alpha=1.67):
        super(DualChannelBlock, self).__init__()
        W = int(alpha * block_channels)
        in_out_channels = [(in_channels, W // 6), (W // 6, W // 3), (W // 3, W // 2)]

        # 左分支（3个3x3卷积）
        self.left_branch = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_out[0], in_out[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(in_out[1]),
                nn.ReLU(inplace=True)
            ) for in_out in in_out_channels ]
        )

        # 右分支（3个3x3卷积）
        self.right_branch = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_out[0], in_out[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(in_out[1]),
                nn.ReLU(inplace=True)
            ) for in_out in in_out_channels]
        )

    def forward(self, x):
        # 左分支
        x1 = self.left_branch[0](x)
        x2 = self.left_branch[1](x1)
        x3 = self.left_branch[2](x2)
        left_x = torch.cat([x1, x2, x3], dim=1)

        # 右分支
        x1 = self.right_branch[0](x)
        x2 = self.right_branch[1](x1)
        x3 = self.right_branch[2](x2)
        right_x = torch.cat([x1, x2, x3], dim=1)

        return left_x + right_x


class Res(nn.Module):
    """残差块：3*3卷积 + 1*1卷积"""

    def __init__(self, in_channels, out_channels):
        super(Res, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3(x) + self.conv1(x)


class ResPath(nn.Module):
    """带残差的跳跃连接"""

    def __init__(self, in_channels, out_channels, num):
        super(ResPath, self).__init__()
        layers = []
        for i in range(num):
            if i == 0:
                layers.append(Res(in_channels, out_channels))
            else:
                layers.append(Res(out_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DC_ResUNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DC_ResUNet, self).__init__()

        # 编码器（Dual-Channel Block + MaxPool）
        self.encoder1 = DualChannelBlock(in_channels, 32)  # [B, 51, H, W]
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DualChannelBlock(51, 64)  # [B, 105, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DualChannelBlock(105, 128)  # [B, 212, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DualChannelBlock(212, 256)  # [B, 426, H/8, W/8]
        self.pool4 = nn.MaxPool2d(2)

        # 桥接层
        self.bridge = DualChannelBlock(426, 512)  # [B, 854, H/16, W/16]

        # 解码器（UpConv + Res-Path + Dual-Channel Block）
        self.up4 = nn.ConvTranspose2d(854, 256, kernel_size=2, stride=2)
        self.respath4 = ResPath(426, 256, num=1)  # 论文表2：ResPath4有1层
        self.decoder4 = DualChannelBlock(512, 256)  # [B, 426, H/8, W/8]

        self.up3 = nn.ConvTranspose2d(426, 128, kernel_size=2, stride=2)
        self.respath3 = ResPath(212, 128, num=2)  # ResPath3有2层
        self.decoder3 = DualChannelBlock(256, 128)  # [B, 212, H/4, W/4]

        self.up2 = nn.ConvTranspose2d(212, 64, kernel_size=2, stride=2)
        self.respath2 = ResPath(105, 64, num=3)  # ResPath2有3层
        self.decoder2 = DualChannelBlock(128, 64)  # [B, 105, H/2, W/2]

        self.up1 = nn.ConvTranspose2d(105, 32, kernel_size=2, stride=2)
        self.respath1 = ResPath(51 ,32, num=4)  # ResPath1有4层
        self.decoder1 = DualChannelBlock(64, 32)  # [B, 51, H, W]

        # 输出层
        self.out_conv = nn.Conv2d(51, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # 桥接层
        bridge = self.bridge(self.pool4(e4))

        # 解码器
        d = self.up4(bridge)
        d = torch.cat([self.respath4(e4), d], dim=1)
        d = self.decoder4(d)

        d = self.up3(d)
        d = torch.cat([self.respath3(e3), d], dim=1)
        d = self.decoder3(d)

        d = self.up2(d)
        d = torch.cat([self.respath2(e2), d], dim=1)
        d = self.decoder2(d)

        d = self.up1(d)
        d = torch.cat([self.respath1(e1), d], dim=1)
        d = self.decoder1(d)

        # 输出
        return self.out_conv(d)


if __name__ == '__main__':
    net = DC_ResUNet(in_channels=3, out_channels=1)
    print(net)

