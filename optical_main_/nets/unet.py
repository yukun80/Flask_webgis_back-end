import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# 定义标准U-Net中的各个模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout_prob))
        # self.attention = CoTAttention(out_channels)

    def forward(self, x):
        x = self.maxpool_conv(x)
        # x = self.attention(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(in_channels, out_channels, dropout_prob)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = self.attention(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.conv2d(in_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class StandardUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StandardUNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownSample(64, 128, dropout_prob=0.2)
        self.down2 = DownSample(128, 256, dropout_prob=0.2)
        self.down3 = DownSample(256, 512, dropout_prob=0.1)
        self.down4 = DownSample(512, 1024, dropout_prob=0.1)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Up_Block_sk(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
            # nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        up_sample = x
        x = torch.cat((skip, x), dim=1)
        x = self.conv(x)
        return x, up_sample


class R34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        self.out_channels = out_channels
        resnet = models.resnet34(pretrained=False)
        # resnet = models.resnet50(pretrained=False)
        filters_resnet = [64, 64, 128, 256, 512]
        filters_decoder = [32, 64, 128, 256, 512]

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_resnet[0], filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True),
        )
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4

        self.Up5 = Up_Block_sk(filters_resnet[4], filters_resnet[3], filters_decoder[3])
        self.Up4 = Up_Block_sk(filters_decoder[3], filters_resnet[2], filters_decoder[2])
        self.Up3 = Up_Block_sk(filters_decoder[2], filters_resnet[1], filters_decoder[1])
        self.Up2 = Up_Block_sk(filters_decoder[1], filters_resnet[0], filters_decoder[0])

        self.pred = nn.Sequential(
            nn.Conv2d(filters_decoder[0], filters_decoder[0] // 2, kernel_size=1),
            nn.BatchNorm2d(filters_decoder[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_decoder[0] // 2, out_channels, kernel_size=1),
        )
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e1_maxp = self.Maxpool(e1)
        e2 = self.Conv2(e1_maxp)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        d4 = self.Up5(e5, e4)
        d3 = self.Up4(d4, e3)
        d2 = self.Up3(d3, e2)
        d1 = self.Up2(d2, e1)

        if self.n_classes == 1:
            out = self.last_activation(self.pred(d1))
        else:
            out = self.pred(d1)  # if nusing BCEWithLogitsLoss
        return out


class Unet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone="default"):
        super().__init__()
        self.backbone = backbone

        if backbone == "default":
            self.standard_unet = StandardUNet(in_channels=14, out_channels=num_classes)

    def forward(self, inputs):
        if self.backbone == "default":
            return self.standard_unet(inputs)
        elif self.backbone == "+++":
            return self.UNet_3Plus(inputs)
        else:
            if self.backbone == "vgg":
                [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
            elif self.backbone == "resnet50":
                [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

            up4 = self.up_concat4(feat4, feat5)
            up3 = self.up_concat3(feat3, up4)
            up2 = self.up_concat2(feat2, up3)
            up1 = self.up_concat1(feat1, up2)

            if self.up_conv is not None:
                up1 = self.up_conv(up1)

            final = self.final(up1)

            return final

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
