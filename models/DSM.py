import os
import torch
import random
import numpy as np
from torch.nn import init
import torch.nn.functional as F

from models.utils import soft_argmax_tensor
from models.ASM import batch_similarity_transform_torch, pts_transform_np


def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()


class conv_block(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class res_conv_block(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(ch_out),
        )
        self.downsample = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(ch_out),
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        return self.relu(out + residual)


class conv_block_scn(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.LeakyReLU(0.1, True),
            torch.nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
            # torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class up_sample(torch.nn.Module):
    def __init__(self, ch_in, ch_out) -> None:
        super().__init__()
        self.up_conv = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            torch.nn.BatchNorm2d(ch_out),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.up_conv(x)
        x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        return x


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Encoder_5layer(torch.nn.Module):
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], enc='conv') -> None:
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)
        # Unet Encoder
        conv = res_conv_block if enc == 'res_conv' else conv_block
        self.enc1 = conv(img_ch, channels[0])
        self.enc2 = conv(channels[0], channels[1])
        self.enc3 = conv(channels[1], channels[2])
        self.enc4 = conv(channels[2], channels[3])
        self.enc5 = conv(channels[3], channels[4])

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.Maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.enc4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.enc5(x5)
        return [x1, x2, x3, x4, x5]


class Encoder_4layer(torch.nn.Module):
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], enc='conv') -> None:
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)
        # Unet Encoder
        conv = res_conv_block if enc == 'res_conv' else conv_block
        self.enc1 = conv(img_ch, channels[1])
        self.enc2 = conv(channels[1], channels[2])
        self.enc3 = conv(channels[2], channels[3])
        self.enc4 = conv(channels[3], channels[4])

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.Maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.enc3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.enc4(x4)
        return [None, x1, x2, x3, x4]


class Encoder_3layer(torch.nn.Module):
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024]) -> None:
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)
        # Unet Encoder
        self.enc1 = conv_block(img_ch, channels[2])
        self.enc2 = conv_block(channels[2], channels[3])
        self.enc3 = conv_block(channels[3], channels[4])

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.Maxpool(x1)
        x2 = self.enc2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.enc3(x3)
        return [None, None, x1, x2, x3]


class Encoder_3layer_adapt(torch.nn.Module):
    '''
    保持feature map尺寸，最后adaptive conv到7x7
    '''

    def __init__(self, in_ch=512, out_ch=51) -> None:
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)
        # Unet Encoder
        self.enc1 = conv_block(in_ch, in_ch)
        self.enc2 = conv_block(in_ch, in_ch)
        self.enc3 = conv_block(in_ch, out_ch)
        self.adapt = torch.nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        out = self.enc1(x)
        out = self.Maxpool(out)
        out = self.enc2(out)
        out = self.Maxpool(out)
        out = self.enc3(out)
        out = self.adapt(out)
        return out


class Decoder_5layer(torch.nn.Module):
    def __init__(self, channels=[64, 128, 256, 512, 1024], sigmoid=True, output_ch=1, enc='conv') -> None:
        super().__init__()
        conv = res_conv_block if enc == 'res_conv' else conv_block
        self.sigmoid = sigmoid
        self.Up_conv5 = conv(channels[3] * 2, channels[3])
        self.Up_conv4 = conv(channels[2] * 2, channels[2])
        self.Up_conv3 = conv(channels[1] * 2, channels[1])
        self.Up_conv2 = conv(channels[0] * 2, channels[0])
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch, kernel_size=1)
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up2 = up_conv(channels[1], channels[0])

    def forward(self, x_enc):
        d5 = self.Up5(x_enc[4])
        d5 = torch.cat((x_enc[3], d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x_enc[2], d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x_enc[1], d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x_enc[0], d2), dim=1)
        out = self.Up_conv2(d2)
        if self.sigmoid:
            out = self.Conv_1x1(out)
            out = torch.nn.Sigmoid()(out)
        return out


class Decoder_4layer(torch.nn.Module):
    def __init__(self, channels=[64, 128, 256, 512, 1024], sigmoid=True, output_ch=1, enc='conv') -> None:
        super().__init__()
        conv = res_conv_block if enc == 'res_conv' else conv_block
        self.sigmoid = sigmoid
        self.Up_conv5 = conv(channels[3] * 2, channels[3])
        self.Up_conv4 = conv(channels[2] * 2, channels[2])
        self.Up_conv3 = conv(channels[1] * 2, channels[1])
        self.Conv_1x1 = torch.nn.Conv2d(channels[1], output_ch, kernel_size=1)
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up3 = up_conv(channels[2], channels[1])

    def forward(self, x_enc):
        d5 = self.Up5(x_enc[4])
        d5 = torch.cat((x_enc[3], d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x_enc[2], d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x_enc[1], d3), dim=1)
        out = self.Up_conv3(d3)
        if self.sigmoid:
            out = self.Conv_1x1(out)
            out = torch.nn.Sigmoid()(out)
        return out


class Decoder_3layer(torch.nn.Module):
    def __init__(self, channels=[64, 128, 256, 512, 1024], sigmoid=True, output_ch=1) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Conv_1x1 = torch.nn.Conv2d(channels[2], output_ch, kernel_size=1)
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up4 = up_conv(channels[3], channels[2])

    def forward(self, x_enc):
        d5 = self.Up5(x_enc[4])
        d5 = torch.cat((x_enc[3], d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x_enc[2], d4), dim=1)
        out = self.Up_conv4(d4)
        if self.sigmoid:
            out = self.Conv_1x1(out)
            out = torch.nn.Sigmoid()(out)
        return out


class Reduce_Module(torch.nn.Module):
    '''
    用在E2之前，降维
    '''

    def __init__(self, channels) -> None:
        super().__init__()
        self.enc = torch.nn.Sequential(conv_block(channels[0], channels[1]), conv_block(channels[1], channels[2]))

    def forward(self, x):
        return self.enc(x)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class UNet1(torch.nn.Module):
    '''
    输出 label，默认的UNet
    '''

    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=1):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])

        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])

        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])

        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return out


class SCN(torch.nn.Module):
    def __init__(self, channel, scale_factor=16, kernel_size=11) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.Conv1 = conv_block_scn(channel, channel * 2, kernel_size)
        self.Conv2 = conv_block_scn(channel * 2, channel * 2, kernel_size)
        self.Conv3 = conv_block_scn(channel * 2, channel * 2, kernel_size)
        self.Conv4 = conv_block_scn(channel * 2, channel, kernel_size)

    def forward(self, x):
        x = F.interpolate(x, (x.shape[-2] // self.scale_factor, x.shape[-1] // self.scale_factor), mode='bilinear', align_corners=True)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = torch.nn.Sigmoid()(x)
        x = F.interpolate(x, (x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor), mode='bilinear', align_corners=True)
        return x


class SCN_UNet(torch.nn.Module):
    '''
    既有seg，又有pts_heatmap，实际效果不好
    '''

    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1, 4], scale_factor=8, scn_kernel_size=11):
        super().__init__()
        self.local_encoder = UNet2_2(img_ch, channels, output_ch)
        self.scn = SCN(output_ch[1], scale_factor, scn_kernel_size)

    def forward(self, x):
        seg, pts, emb = self.local_encoder(x)
        scn = self.scn(pts)
        pts = pts * scn
        return pts, seg, emb


class UNet2_2(torch.nn.Module):
    '''
    输出 label + pts，但是pts是在第二层输出
    '''

    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1, 3]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

        self.pts_head = torch.nn.Sequential(
            conv_block(channels[1] * 2, channels[0]), torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        pts = self.pts_head(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return out, pts, [x1, x2, x3, x4, x5]


class UNet2_3(torch.nn.Module):
    '''
    输出 label + pts，pts是在第三层输出
    '''

    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1, 3]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

        self.pts_head = torch.nn.Sequential(
            conv_block(channels[2] * 2, channels[0]), torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        pts = self.pts_head(d4)
        att = pts.sum(1, keepdim=True)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = F.interpolate(att, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return out, pts, [x1, x2, x3, x4, x5]


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ResUNet2(torch.nn.Module):
    '''
    输出 label + pts
    '''

    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1, 3]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = res_conv_block(img_ch, channels[0])
        self.Conv2 = res_conv_block(channels[0], channels[1])
        self.Conv3 = res_conv_block(channels[1], channels[2])
        self.Conv4 = res_conv_block(channels[2], channels[3])
        self.Conv5 = res_conv_block(channels[3], channels[4])

        # Decoder path
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = res_conv_block(channels[3] * 2, channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = res_conv_block(channels[2] * 2, channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = res_conv_block(channels[1] * 2, channels[1])
        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = res_conv_block(channels[0] * 2, channels[0])
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

        self.pts_head = torch.nn.Sequential(
            res_conv_block(channels[1] * 2, channels[0]), torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        pts = self.pts_head(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        seg = torch.nn.Sigmoid()(d1)

        return seg, pts, [x1, x2, x3, x4, x5]


class Global_Context(torch.nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()
        self.enc = torch.nn.Sequential(conv_block(ch, ch), conv_block(ch, ch), conv_block(ch, ch), torch.nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.enc(x)


class MLP(torch.nn.Module):
    '''
    接在UNet2_3之后
    '''

    def __init__(self, in_dim, out_dim=102, hidden_dim=512) -> None:
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(hidden_dim * 2, out_dim),
        )

    def forward(self, x):
        B, N, D = x.shape
        x = x.reshape(B, N * D)
        out = self.enc(x)
        out = out.reshape(B, -1, D)
        return out


class DeepShapeModel(torch.nn.Module):
    '''
    串联起整个框架
    '''

    def __init__(self, stem, gc, mlp, asm, key_idxes) -> None:
        super().__init__()
        self.stem = stem
        self.gc = gc
        self.mlp = mlp
        self.asm = asm
        self.mean_shape_np = self.asm.mean_shape.cpu().numpy()
        self.key_idxes = key_idxes

    def forward(self, x):
        key_pts_heatmap, global_context = self.stem(x)
        key_pts_coord = soft_argmax_tensor(key_pts_heatmap)
        shape_init = self.init_shape(key_pts_coord)
        shape_init = torch.flatten(shape_init, 1)
        mlp_input = torch.cat((shape_init, global_context), 0)
        shape = self.mlp(mlp_input)
        shape = shape.reshape((x.shape[0], -1, 2))
        b = self.asm.reflex_shape(shape)
        return key_pts_heatmap, key_pts_coord, shape, b

    def init_shape(self, key_pts_coord):
        '''
        根据关键点，将mean shape映射过来成为initial shape
        处key_pts外，其他点都是叶子结点
        '''
        key_pts_coord_np = key_pts_coord.detach().cpu().numpy()
        shape_init_np = pts_transform_np(key_pts_coord_np, self.key_idxes, self.mean_shape_np)
        shape = shape_init_np.to(key_pts_coord.device)
        # 将shape中key_idxes的位置替换为stem预测值，为了可以梯度回传
        shape[self.key_idxes] = key_pts_coord
        return shape


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# for UNeXt-----
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import math


# class DWConv(torch.nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = torch.nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x


# class shiftmlp(torch.nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, shift_size=5):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.dim = in_features
#         self.fc1 = torch.nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = torch.nn.GELU()
#         self.fc2 = torch.nn.Linear(hidden_features, out_features)
#         self.drop = torch.nn.Dropout(drop)

#         self.shift_size = shift_size
#         self.pad = shift_size // 2

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, torch.nn.Linear) and m.bias is not None:
#                 torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.LayerNorm):
#             torch.nn.init.constant_(m.bias, 0)
#             torch.nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, torch.nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         # pdb.set_trace()
#         B, N, C = x.shape

#         xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
#         xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
#         xs = torch.chunk(xn, self.shift_size, 1)
#         x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
#         x_cat = torch.cat(x_shift, 1)
#         x_cat = torch.narrow(x_cat, 2, self.pad, H)
#         x_s = torch.narrow(x_cat, 3, self.pad, W)

#         x_s = x_s.reshape(B, C, H * W).contiguous()
#         x_shift_r = x_s.transpose(1, 2)

#         x = self.fc1(x_shift_r)

#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)

#         xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
#         xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
#         xs = torch.chunk(xn, self.shift_size, 1)
#         x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
#         x_cat = torch.cat(x_shift, 1)
#         x_cat = torch.narrow(x_cat, 2, self.pad, H)
#         x_s = torch.narrow(x_cat, 3, self.pad, W)
#         x_s = x_s.reshape(B, C, H * W).contiguous()
#         x_shift_c = x_s.transpose(1, 2)

#         x = self.fc2(x_shift_c)
#         x = self.drop(x)
#         return x


# class shiftedBlock(torch.nn.Module):
#     def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
#         super().__init__()
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
#         self.norm2 = torch.nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, torch.nn.Linear) and m.bias is not None:
#                 torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.LayerNorm):
#             torch.nn.init.constant_(m.bias, 0)
#             torch.nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, torch.nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):

#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
#         return x


# class OverlapPatchEmbed(torch.nn.Module):
#     '''
#     Image to Patch Embedding
#     '''

#     def __init__(self, img_size=(544, 736), patch_size=7, stride=4, in_chans=3, embed_dim=768):
#         super().__init__()
#         # img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
#         self.num_patches = self.H * self.W
#         self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
#         self.norm = torch.nn.LayerNorm(embed_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=0.02)
#             if isinstance(m, torch.nn.Linear) and m.bias is not None:
#                 torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.LayerNorm):
#             torch.nn.init.constant_(m.bias, 0)
#             torch.nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, torch.nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x, H, W


# class UNext1(torch.nn.Module):
#     '''
#     Conv 3 + MLP 2 + shifted MLP
#     '''

#     def __init__(
#         self,
#         img_ch=3,
#         channels=[16, 32, 64, 128, 256],
#         output_ch=[1],
#         img_size=(544, 736),
#         drop_rate=0.0,
#         drop_path_rate=0.0,
#         depths=[1, 1, 1],
#         **kwargs
#     ):
#         super().__init__()

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         self.pool = torch.nn.MaxPool2d(2)

#         self.encoder1 = conv_block(img_ch, channels[0])
#         self.encoder2 = conv_block(channels[0], channels[1])
#         self.encoder3 = conv_block(channels[1], channels[2])
#         self.patch_embed3 = OverlapPatchEmbed(
#             img_size=(img_size[0] // 4, img_size[1] // 4), patch_size=3, stride=2, in_chans=channels[2], embed_dim=channels[3]
#         )
#         self.block3 = shiftedBlock(dim=channels[3], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])
#         self.norm3 = torch.nn.LayerNorm(channels[3])
#         self.patch_embed4 = OverlapPatchEmbed(
#             img_size=(img_size[0] // 8, img_size[1] // 8), patch_size=3, stride=2, in_chans=channels[3], embed_dim=channels[4]
#         )
#         self.block4 = shiftedBlock(dim=channels[4], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1])
#         self.norm4 = torch.nn.LayerNorm(channels[4])

#         self.decoder5 = up_sample(channels[4], channels[3])
#         self.dblock4 = shiftedBlock(dim=channels[3], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])
#         self.dnorm4 = torch.nn.LayerNorm(channels[3])
#         self.decoder4 = up_sample(channels[3], channels[2])
#         self.dblock3 = shiftedBlock(dim=channels[2], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1])
#         self.dnorm3 = torch.nn.LayerNorm(channels[2])

#         self.decoder3 = up_sample(channels[2], channels[1])
#         self.decoder2 = up_sample(channels[1], channels[0])
#         self.decoder1 = up_sample(channels[0], channels[0])
#         self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

#     def forward(self, x):

#         B = x.shape[0]
#         ### Encoder
#         ### Conv Stage

#         ### Stage 1
#         out = self.encoder1(x)
#         out = self.pool(out)
#         t1 = out
#         ### Stage 2
#         out = self.encoder2(out)
#         out = self.pool(out)
#         t2 = out
#         ### Stage 3
#         out = self.encoder3(out)
#         out = self.pool(out)
#         t3 = out

#         ### Tokenized MLP Stage
#         ### Stage 4
#         out, H, W = self.patch_embed3(out)
#         # for blk in self.block3:
#         #     out = blk(out, H, W)
#         out = self.block3(out, H, W)
#         out = self.norm3(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         t4 = out  # [8, 128, 23, 17]

#         ### Bottleneck
#         out, H, W = self.patch_embed4(out)
#         out = self.block4(out, H, W)
#         out = self.norm4(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         ### De-Stage 4
#         out = self.decoder5(out)
#         out = out + t4
#         _, _, H, W = out.shape
#         out = out.flatten(2).transpose(1, 2)
#         out = self.dblock4(out, H, W)
#         out = self.dnorm4(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         ### De-Stage 3
#         out = self.decoder4(out)
#         out = out + t3
#         _, _, H, W = out.shape
#         out = out.flatten(2).transpose(1, 2)
#         out = self.dblock3(out, H, W)
#         out = self.dnorm3(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         out = self.decoder3(out)
#         out = out + t2
#         out = self.decoder2(out)
#         out = out + t1
#         out = self.decoder1(out)
#         out = self.Conv_1x1(out)
#         out = torch.nn.Sigmoid()(out)
#         return out
