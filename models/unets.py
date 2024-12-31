import os
import torch
import random
import numpy as np
from torch.nn import init
import torch.nn.functional as F


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                        torch.nn.BatchNorm2d(ch_out), torch.nn.ReLU(inplace=True),
                                        torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                                        torch.nn.BatchNorm2d(ch_out), torch.nn.ReLU(inplace=True))

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


class up_conv(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = torch.nn.Sequential(torch.nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
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


class Attention_block(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = torch.nn.Sequential(torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                       torch.nn.BatchNorm2d(F_int))
        self.W_x = torch.nn.Sequential(torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                                       torch.nn.BatchNorm2d(F_int))
        self.psi = torch.nn.Sequential(torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                       torch.nn.BatchNorm2d(1), torch.nn.Sigmoid())
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class UNet1(torch.nn.Module):
    '''
    输出 label，默认的UNet
    '''
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1]):
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


class UNet2(torch.nn.Module):
    '''
    输出 label + pts
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

        self.pts_head = torch.nn.Sequential(conv_block(channels[2] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())

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
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return [out, pts]


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

        self.sdf_head = torch.nn.Sequential(conv_block(channels[0], channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1), torch.nn.Tanh())

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
        sdf = self.sdf_head(d2)
        return [out, sdf]

class UNet2_(torch.nn.Module):
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

        self.pts_head = torch.nn.Sequential(conv_block(channels[1] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())

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
        return [out, pts]




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

        self.sdf_head = torch.nn.Sequential(conv_block(channels[0], channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1), torch.nn.Tanh())

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
        sdf = self.sdf_head(d2)
        return [out, sdf]

class UNet3(torch.nn.Module):
    '''
    输出 label + edge + pts
    '''
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1, 1, 3]):
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

        self.pts_head = torch.nn.Sequential(conv_block(channels[2] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())
        self.edge_head = torch.nn.Sequential(conv_block(channels[1] * 2, channels[0]),
                                             torch.nn.Conv2d(channels[0], output_ch[1], kernel_size=1),
                                             torch.nn.Sigmoid())

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
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d3 = d3 * (1 + att)
        edge = self.edge_head(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = F.interpolate(edge, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return [out, edge, pts]


class U_Net_6layers(torch.nn.Module):
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])
        self.Conv6 = conv_block(channels[4], channels[4])

        # Decoder path
        self.Up6 = up_conv(channels[4], channels[4])
        self.Up_conv6 = conv_block(channels[4] * 2, channels[4])
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

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

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        # decoding + concat path
        d6 = self.Up6(x6)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)

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


class U_Net_6layers_att(torch.nn.Module):
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[3, 1, 1]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])
        self.Conv6 = conv_block(channels[4], channels[4])

        # Decoder path
        self.Up6 = up_conv(channels[4], channels[4])
        self.Up_conv6 = conv_block(channels[4] * 2, channels[4])
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])

        # output_head
        self.pts_head = torch.nn.Sequential(conv_block(channels[2] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1),
                                            torch.nn.Sigmoid())
        self.edge_head = torch.nn.Sequential(conv_block(channels[1] * 2, channels[0]),
                                             torch.nn.Conv2d(channels[0], output_ch[1], kernel_size=1),
                                             torch.nn.Sigmoid())
        # self.seg_head = torch.nn.Sequential(conv_block(channels[0], channels[0]),
        #                                     torch.nn.Conv2d(channels[0], output_ch[2], kernel_size=1),
        #                                     torch.nn.Sigmoid())

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[2], kernel_size=1)

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

        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        # decoding + concat path
        d6 = self.Up6(x6)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        pts = self.pts_head(d4)
        att = pts.sum(1, keepdim=True)
        d4 = d4 * (1 + att)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        edge = self.edge_head(d3)
        d3 = d3 * (1 + edge)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return [out, edge, pts]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class UNet_pp1(torch.nn.Module):
    '''
    输出 label，默认的UNet
    '''
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1]):
        super().__init__()

        self.Maxpool = torch.nn.MaxPool2d(2)
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])

        self.Up4 = up_conv(channels[3], channels[2])
        self.Up4_1 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up_conv4_2 = conv_block(channels[2] * 3, channels[2])

        self.Up3 = up_conv(channels[2], channels[1])
        self.Up3_1 = up_conv(channels[2], channels[1])
        self.Up3_2 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up_conv3_2 = conv_block(channels[1] * 3, channels[1])
        self.Up_conv3_3 = conv_block(channels[1] * 4, channels[1])

        self.Up2 = up_conv(channels[1], channels[0])
        self.Up2_1 = up_conv(channels[1], channels[0])
        self.Up2_2 = up_conv(channels[1], channels[0])
        self.Up2_3 = up_conv(channels[1], channels[0])

        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])
        self.Up_conv2_2 = conv_block(channels[0] * 3, channels[0])
        self.Up_conv2_3 = conv_block(channels[0] * 4, channels[0])
        self.Up_conv2_4 = conv_block(channels[0] * 5, channels[0])

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_2 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_3 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

    def forward(self, x):
        # L1 level
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        d0_1 = self.Up2(x2)
        d0_1 = torch.cat((x1, d0_1), dim=1)
        d0_1 = self.Up_conv2(d0_1)
        L1 = self.Conv_1x1(d0_1)
        out_1 = torch.nn.Sigmoid()(L1)

        #L2 level
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d1_1 = self.Up3(x3)
        d1_1 = torch.cat((x2, d1_1), dim=1)
        d1_1 = self.Up_conv3(d1_1)

        d0_2 = self.Up2_1(d1_1)
        d0_2 = torch.cat((d0_1, d0_2), dim=1)
        d0_2 = torch.cat((x1, d0_2), dim=1)
        d0_2 = self.Up_conv2_2(d0_2)
        L2 = self.Conv_1x1_1(d0_2)
        out_2 = torch.nn.Sigmoid()(L2)

        #L3 level
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d2_1 = self.Up4(x4)
        d2_1 = torch.cat((x3, d2_1), dim=1)
        d2_1 = self.Up_conv4(d2_1)

        d1_2 = self.Up3_1(d2_1)
        d1_2 = torch.cat((d1_1, d1_2), dim=1)
        d1_2 = torch.cat((x2, d1_2), dim=1)
        d1_2 = self.Up_conv3_2(d1_2)

        d0_3 = self.Up2_2(d1_2)
        d0_3 = torch.cat((d0_2, d0_3), dim=1)
        d0_3 = torch.cat((d0_1, d0_3), dim=1)
        d0_3 = torch.cat((x1, d0_3), dim=1)
        d0_3 = self.Up_conv2_3(d0_3)
        L3 = self.Conv_1x1_2(d0_3)
        out_3 = torch.nn.Sigmoid()(L3)

        #L4 level
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4_1(d5)
        d4 = torch.cat((d2_1, d4), dim=1)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4_2(d4)

        d3 = self.Up3_2(d4)
        d3 = torch.cat((d1_2, d3), dim=1)
        d3 = torch.cat((d1_1, d3), dim=1)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3_3(d3)

        d2 = self.Up2_3(d3)
        d2 = torch.cat((d0_3, d2), dim=1)
        d2 = torch.cat((d0_2, d2), dim=1)
        d2 = torch.cat((d0_1, d2), dim=1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2_4(d2)

        d1 = self.Conv_1x1_3(d2)
        out_4 = torch.nn.Sigmoid()(d1)

        out = (out_1 + out_2 + out_3 + out_4) / 4
        return out


class UNet_pp2(torch.nn.Module):
    '''
    输出 label + pts
    '''
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1, 3]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        # Decoder path
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])

        self.Up4 = up_conv(channels[3], channels[2])
        self.Up4_1 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up_conv4_2 = conv_block(channels[2] * 3, channels[2])

        self.Up3 = up_conv(channels[2], channels[1])
        self.Up3_1 = up_conv(channels[2], channels[1])
        self.Up3_2 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up_conv3_2 = conv_block(channels[1] * 3, channels[1])
        self.Up_conv3_3 = conv_block(channels[1] * 4, channels[1])

        self.Up2 = up_conv(channels[1], channels[0])
        self.Up2_1 = up_conv(channels[1], channels[0])
        self.Up2_2 = up_conv(channels[1], channels[0])
        self.Up2_3 = up_conv(channels[1], channels[0])

        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])
        self.Up_conv2_2 = conv_block(channels[0] * 3, channels[0])
        self.Up_conv2_3 = conv_block(channels[0] * 4, channels[0])
        self.Up_conv2_4 = conv_block(channels[0] * 5, channels[0])

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_2 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_3 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

        self.pts_head = torch.nn.Sequential(conv_block(channels[2] * 3, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())

    def forward(self, x):
        # L1 level
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        d0_1 = self.Up2(x2)
        d0_1 = torch.cat((x1, d0_1), dim=1)
        d0_1 = self.Up_conv2(d0_1)
        L1 = self.Conv_1x1(d0_1)
        out_1 = torch.nn.Sigmoid()(L1)

        #L2 level
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d1_1 = self.Up3(x3)
        d1_1 = torch.cat((x2, d1_1), dim=1)
        d1_1 = self.Up_conv3(d1_1)

        d0_2 = self.Up2_1(d1_1)
        d0_2 = torch.cat((d0_1, d0_2), dim=1)
        d0_2 = torch.cat((x1, d0_2), dim=1)
        d0_2 = self.Up_conv2_2(d0_2)
        L2 = self.Conv_1x1_1(d0_2)
        out_2 = torch.nn.Sigmoid()(L2)

        #L3 level
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d2_1 = self.Up4(x4)
        d2_1 = torch.cat((x3, d2_1), dim=1)
        d2_1 = self.Up_conv4(d2_1)

        d1_2 = self.Up3_1(d2_1)
        d1_2 = torch.cat((d1_1, d1_2), dim=1)
        d1_2 = torch.cat((x2, d1_2), dim=1)
        d1_2 = self.Up_conv3_2(d1_2)

        d0_3 = self.Up2_2(d1_2)
        d0_3 = torch.cat((d0_2, d0_3), dim=1)
        d0_3 = torch.cat((d0_1, d0_3), dim=1)
        d0_3 = torch.cat((x1, d0_3), dim=1)
        d0_3 = self.Up_conv2_3(d0_3)
        L3 = self.Conv_1x1_2(d0_3)
        out_3 = torch.nn.Sigmoid()(L3)

        #L4 level
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4_1(d5)
        d4 = torch.cat((d2_1, d4), dim=1)
        d4 = torch.cat((x3, d4), dim=1)
        pts = self.pts_head(d4)
        d4 = self.Up_conv4_2(d4)

        d3 = self.Up3_2(d4)
        d3 = torch.cat((d1_2, d3), dim=1)
        d3 = torch.cat((d1_1, d3), dim=1)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3_3(d3)

        d2 = self.Up2_3(d3)
        d2 = torch.cat((d0_3, d2), dim=1)
        d2 = torch.cat((d0_2, d2), dim=1)
        d2 = torch.cat((d0_1, d2), dim=1)
        d2 = torch.cat((x1, d2), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2_4(d2)
        d1 = self.Conv_1x1_3(d2)
        out_4 = torch.nn.Sigmoid()(d1)

        seg = (out_1 + out_2 + out_3 + out_4) / 4
        return [seg, pts]


class UNet_pp3(torch.nn.Module):
    '''
    输出 label + edge + pts
    '''
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1, 1, 3]):
        super().__init__()
        self.Maxpool = torch.nn.MaxPool2d(2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        # Decoder path
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[3] * 2, channels[3])

        self.Up4 = up_conv(channels[3], channels[2])
        self.Up4_1 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[2] * 2, channels[2])
        self.Up_conv4_2 = conv_block(channels[2] * 3, channels[2])

        self.Up3 = up_conv(channels[2], channels[1])
        self.Up3_1 = up_conv(channels[2], channels[1])
        self.Up3_2 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[1] * 2, channels[1])
        self.Up_conv3_2 = conv_block(channels[1] * 3, channels[1])
        self.Up_conv3_3 = conv_block(channels[1] * 4, channels[1])

        self.Up2 = up_conv(channels[1], channels[0])
        self.Up2_1 = up_conv(channels[1], channels[0])
        self.Up2_2 = up_conv(channels[1], channels[0])
        self.Up2_3 = up_conv(channels[1], channels[0])

        self.Up_conv2 = conv_block(channels[0] * 2, channels[0])
        self.Up_conv2_2 = conv_block(channels[0] * 3, channels[0])
        self.Up_conv2_3 = conv_block(channels[0] * 4, channels[0])
        self.Up_conv2_4 = conv_block(channels[0] * 5, channels[0])

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_2 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)
        self.Conv_1x1_3 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

        self.pts_head = torch.nn.Sequential(conv_block(channels[2] * 3, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())
        self.edge_head = torch.nn.Sequential(conv_block(channels[1] * 4, channels[0]),
                                             torch.nn.Conv2d(channels[0], output_ch[1], kernel_size=1),
                                             torch.nn.Sigmoid())

    def forward(self, x):
        # L1 level
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        d0_1 = self.Up2(x2)
        d0_1 = torch.cat((x1, d0_1), dim=1)
        d0_1 = self.Up_conv2(d0_1)
        L1 = self.Conv_1x1(d0_1)
        out_1 = torch.nn.Sigmoid()(L1)

        #L2 level
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        d1_1 = self.Up3(x3)
        d1_1 = torch.cat((x2, d1_1), dim=1)
        d1_1 = self.Up_conv3(d1_1)

        d0_2 = self.Up2_1(d1_1)
        d0_2 = torch.cat((d0_1, d0_2), dim=1)
        d0_2 = torch.cat((x1, d0_2), dim=1)
        d0_2 = self.Up_conv2_2(d0_2)
        L2 = self.Conv_1x1_1(d0_2)
        out_2 = torch.nn.Sigmoid()(L2)

        #L3 level
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d2_1 = self.Up4(x4)
        d2_1 = torch.cat((x3, d2_1), dim=1)
        d2_1 = self.Up_conv4(d2_1)

        d1_2 = self.Up3_1(d2_1)
        d1_2 = torch.cat((d1_1, d1_2), dim=1)
        d1_2 = torch.cat((x2, d1_2), dim=1)
        d1_2 = self.Up_conv3_2(d1_2)

        d0_3 = self.Up2_2(d1_2)
        d0_3 = torch.cat((d0_2, d0_3), dim=1)
        d0_3 = torch.cat((d0_1, d0_3), dim=1)
        d0_3 = torch.cat((x1, d0_3), dim=1)
        d0_3 = self.Up_conv2_3(d0_3)
        L3 = self.Conv_1x1_2(d0_3)
        out_3 = torch.nn.Sigmoid()(L3)

        #L4 level
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4_1(d5)
        d4 = torch.cat((d2_1, d4), dim=1)
        d4 = torch.cat((x3, d4), dim=1)
        pts = self.pts_head(d4)
        d4 = self.Up_conv4_2(d4)

        d3 = self.Up3_2(d4)
        d3 = torch.cat((d1_2, d3), dim=1)
        d3 = torch.cat((d1_1, d3), dim=1)
        d3 = torch.cat((x2, d3), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d3 = d3 * (1 + att)
        edge = self.edge_head(d3)
        d3 = self.Up_conv3_3(d3)

        d2 = self.Up2_3(d3)
        d2 = torch.cat((d0_3, d2), dim=1)
        d2 = torch.cat((d0_2, d2), dim=1)
        d2 = torch.cat((d0_1, d2), dim=1)
        d2 = torch.cat((x1, d2), dim=1)
        att = F.interpolate(edge, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2_4(d2)
        d1 = self.Conv_1x1_3(d2)
        out_4 = torch.nn.Sigmoid()(d1)

        seg = (out_1 + out_2 + out_3 + out_4) / 4
        return [seg, edge, pts]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ResUNet1(torch.nn.Module):
    '''
    输出 label，原始 ResUNet
    '''
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1]):
        super().__init__()

        self.Maxpool = torch.nn.MaxPool2d(2)

        self.Conv1 = res_conv_block(img_ch, channels[0])
        self.Conv2 = res_conv_block(channels[0], channels[1])
        self.Conv3 = res_conv_block(channels[1], channels[2])
        self.Conv4 = res_conv_block(channels[2], channels[3])
        self.Conv5 = res_conv_block(channels[3], channels[4])

        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = res_conv_block(channels[3] * 2, channels[3])
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = res_conv_block(channels[2] * 2, channels[2])
        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = res_conv_block(channels[1] * 2, channels[1])
        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = res_conv_block(channels[0] * 2, channels[0])
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

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

        self.pts_head = torch.nn.Sequential(res_conv_block(channels[2] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())

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
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        seg = torch.nn.Sigmoid()(d1)

        return [seg, pts]


class ResUNet3(torch.nn.Module):
    def __init__(self, img_ch=1, channels=[64, 128, 256, 512, 1024], output_ch=[1, 1, 3]):
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

        self.pts_head = torch.nn.Sequential(res_conv_block(channels[2] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())
        self.edge_head = torch.nn.Sequential(conv_block(channels[1] * 2, channels[0]),
                                             torch.nn.Conv2d(channels[0], output_ch[1], kernel_size=1),
                                             torch.nn.Sigmoid())

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
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d3 = d3 * (1 + att)
        edge = self.edge_head(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        att = F.interpolate(edge, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        out = torch.nn.Sigmoid()(d1)
        return [out, edge, pts]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Att_UNet1(torch.nn.Module):
    '''
    输出 label + pts
    '''
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1]):
        super().__init__()

        self.Maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=channels[0])
        self.Conv2 = conv_block(ch_in=channels[0], ch_out=channels[1])
        self.Conv3 = conv_block(ch_in=channels[1], ch_out=channels[2])
        self.Conv4 = conv_block(ch_in=channels[2], ch_out=channels[3])
        self.Conv5 = conv_block(ch_in=channels[3], ch_out=channels[4])

        self.Up5 = up_conv(ch_in=channels[4], ch_out=channels[3])
        self.Up_conv5 = conv_block(ch_in=channels[3] * 2, ch_out=channels[3])
        self.Up4 = up_conv(ch_in=channels[3], ch_out=channels[2])
        self.Up_conv4 = conv_block(ch_in=channels[2] * 2, ch_out=channels[2])
        self.Up3 = up_conv(ch_in=channels[2], ch_out=channels[1])
        self.Up_conv3 = conv_block(ch_in=channels[1] * 2, ch_out=channels[1])
        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[0])
        self.Up_conv2 = conv_block(ch_in=channels[0] * 2, ch_out=channels[0])

        self.Att5 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=channels[2])
        self.Att4 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[1])
        self.Att3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[0])
        self.Att2 = Attention_block(F_g=channels[0], F_l=channels[0], F_int=(channels[0] // 2))

        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1, stride=1, padding=0)

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
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        out = torch.nn.Sigmoid()(d1)
        return out


class Att_UNet2(torch.nn.Module):
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1, 3]):
        super().__init__()

        self.Maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[1])
        self.Conv2 = conv_block(channels[1], channels[2])
        self.Conv3 = conv_block(channels[2], channels[3])
        self.Conv4 = conv_block(channels[3], channels[4])
        self.Conv5 = conv_block(channels[4], channels[4])

        # Decoder path
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
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1, stride=1, padding=0)

        self.Att5 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=channels[2])
        self.Att4 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[1])
        self.Att3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[0])
        self.Att2 = Attention_block(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2)

        self.pts_head = torch.nn.Sequential(conv_block(channels[2] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())

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
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        pts = self.pts_head(d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(4, 4), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        seg = torch.nn.Sigmoid()(d1)

        return [seg, pts]


class Att_UNet3(torch.nn.Module):
    def __init__(self, img_ch=3, channels=[64, 128, 256, 512, 1024], output_ch=[1, 1, 3]):
        super().__init__()

        self.Maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        self.Conv1 = conv_block(img_ch, channels[1])
        self.Conv2 = conv_block(channels[1], channels[2])
        self.Conv3 = conv_block(channels[2], channels[3])
        self.Conv4 = conv_block(channels[3], channels[4])
        self.Conv5 = conv_block(channels[4], channels[4])

        # Decoder path
        self.Up5 = up_conv(channels[4], channels[4])
        self.Up_conv5 = conv_block(channels[4] * 2, channels[4])
        self.Up4 = up_conv(channels[4], channels[3])
        self.Up_conv4 = conv_block(channels[3] * 2, channels[3])
        self.Up3 = up_conv(channels[3], channels[2])
        self.Up_conv3 = conv_block(channels[2] * 2, channels[2])
        self.Up2 = up_conv(channels[2], channels[1])
        self.Up_conv2 = conv_block(channels[1] * 2, channels[0])
        self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1, stride=1, padding=0)

        self.Att5 = Attention_block(F_g=channels[4], F_l=channels[4], F_int=channels[3])
        self.Att4 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=channels[2])
        self.Att3 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[1])
        self.Att2 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[0])

        self.pts_head = torch.nn.Sequential(conv_block(channels[3] * 2, channels[0]),
                                            torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
                                            torch.nn.Sigmoid())
        self.edge_head = torch.nn.Sequential(conv_block(channels[2] * 2, channels[0]),
                                             torch.nn.Conv2d(channels[0], output_ch[1], kernel_size=1),
                                             torch.nn.Sigmoid())

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
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        pts = self.pts_head(d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        att = pts.sum(1, keepdim=True)
        att = F.interpolate(att, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d3 = d3 * (1 + att)
        edge = self.edge_head(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        att = F.interpolate(edge, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        d2 = d2 * (1 + att)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        seg = torch.nn.Sigmoid()(d1)

        return [seg, edge, pts]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., shift_size=5):
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
#             trunc_normal_(m.weight, std=.02)
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
#     def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
#         super().__init__()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
#         self.norm2 = torch.nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=.02)
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
#         self.proj = torch.nn.Conv2d(in_chans,
#                                     embed_dim,
#                                     kernel_size=patch_size,
#                                     stride=stride,
#                                     padding=(patch_size[0] // 2, patch_size[1] // 2))
#         self.norm = torch.nn.LayerNorm(embed_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, torch.nn.Linear):
#             trunc_normal_(m.weight, std=.02)
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
#     def __init__(self,
#                  img_ch=3,
#                  channels=[16, 32, 64, 128, 256],
#                  output_ch=[1],
#                  img_size=(544, 736),
#                  drop_rate=0.,
#                  drop_path_rate=0.,
#                  depths=[1, 1, 1],
#                  **kwargs):
#         super().__init__()

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         self.pool = torch.nn.MaxPool2d(2)

#         self.encoder1 = conv_block(img_ch, channels[0])
#         self.encoder2 = conv_block(channels[0], channels[1])
#         self.encoder3 = conv_block(channels[1], channels[2])
#         self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4),
#                                               patch_size=3,
#                                               stride=2,
#                                               in_chans=channels[2],
#                                               embed_dim=channels[3])
#         self.block3 = shiftedBlock(dim=channels[3], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])
#         self.norm3 = torch.nn.LayerNorm(channels[3])
#         self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8),
#                                               patch_size=3,
#                                               stride=2,
#                                               in_chans=channels[3],
#                                               embed_dim=channels[4])
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


# class UNext2(torch.nn.Module):
#     '''
#     Conv 4 + MLP 2 + shifted MLP
#     '''
#     def __init__(self,
#                  img_ch=3,
#                  channels=[64, 128, 256, 512, 1024],
#                  output_ch=[3, 1, 1],
#                  img_size=(544, 736),
#                  drop_rate=0.,
#                  drop_path_rate=0.,
#                  depths=[1, 1, 1],
#                  **kwargs):
#         super().__init__()

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         self.Maxpool = torch.nn.MaxPool2d(2)

#         # Encoder path
#         self.encoder1 = conv_block(img_ch, channels[0])
#         self.encoder2 = conv_block(channels[0], channels[1])
#         self.encoder3 = conv_block(channels[1], channels[2])
#         self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4),
#                                               patch_size=3,
#                                               stride=2,
#                                               in_chans=channels[2],
#                                               embed_dim=channels[3])
#         self.block3 = shiftedBlock(dim=channels[3], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])
#         self.norm3 = torch.nn.LayerNorm(channels[3])
#         self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8),
#                                               patch_size=3,
#                                               stride=2,
#                                               in_chans=channels[3],
#                                               embed_dim=channels[4])
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
#         self.decoder0 = conv_block(channels[0], channels[0])
#         self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

#         self.pts_head = torch.nn.Sequential(conv_block(channels[0] * 2, channels[0]),
#                                             torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
#                                             torch.nn.Sigmoid())

#     def forward(self, x):
#         B = x.shape[0]
#         ### Encoder
#         ### Conv Stage

#         ### Stage 1
#         out = self.encoder1(x)
#         out = self.Maxpool(out)
#         t1 = out
#         ### Stage 2
#         out = self.encoder2(out)
#         out = self.Maxpool(out)
#         t2 = out
#         ### Stage 3
#         out = self.encoder3(out)
#         out = self.Maxpool(out)
#         t3 = out

#         ### Tokenized MLP Stage
#         ### Stage 4
#         out, H, W = self.patch_embed3(out)
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
#         pts = self.pts_head(out)
#         out = self.decoder2(out)
#         out = out + t1
#         out = self.decoder1(out)
#         att = pts.sum(1, keepdim=True)
#         att = F.interpolate(att, scale_factor=(4, 4), mode='bilinear', align_corners=False)
#         out = out * (1 + att)
#         out = self.decoder0(out)
#         out = self.Conv_1x1(out)
#         seg = torch.nn.Sigmoid()(out)
#         return [seg, pts]


# class UNext3(torch.nn.Module):
#     '''
#     Conv 4 + MLP 2 + shifted MLP
#     '''
#     def __init__(self,
#                  img_ch=3,
#                  channels=[64, 128, 256, 512, 1024],
#                  output_ch=[3, 1, 1],
#                  img_size=(544, 736),
#                  drop_rate=0.,
#                  drop_path_rate=0.,
#                  depths=[1, 1, 1],
#                  **kwargs):
#         super().__init__()

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         self.Maxpool = torch.nn.MaxPool2d(2)

#         # Encoder path
#         self.encoder1 = conv_block(img_ch, channels[0])
#         self.encoder2 = conv_block(channels[0], channels[1])
#         self.encoder3 = conv_block(channels[1], channels[2])
#         self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4),
#                                               patch_size=3,
#                                               stride=2,
#                                               in_chans=channels[2],
#                                               embed_dim=channels[3])
#         self.block3 = shiftedBlock(dim=channels[3], mlp_ratio=1, drop=drop_rate, drop_path=dpr[0])
#         self.norm3 = torch.nn.LayerNorm(channels[3])
#         self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8),
#                                               patch_size=3,
#                                               stride=2,
#                                               in_chans=channels[3],
#                                               embed_dim=channels[4])
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
#         self.decoder0 = conv_block(channels[0], channels[0])
#         self.Conv_1x1 = torch.nn.Conv2d(channels[0], output_ch[0], kernel_size=1)

#         self.pts_head = torch.nn.Sequential(conv_block(channels[0] * 2, channels[0]),
#                                             torch.nn.Conv2d(channels[0], output_ch[-1], kernel_size=1),
#                                             torch.nn.Sigmoid())
#         self.edge_head = torch.nn.Sequential(conv_block(channels[0], channels[0]),
#                                              torch.nn.Conv2d(channels[0], output_ch[1], kernel_size=1),
#                                              torch.nn.Sigmoid())

#     def forward(self, x):
#         B = x.shape[0]
#         ### Encoder
#         ### Conv Stage

#         ### Stage 1
#         out = self.encoder1(x)
#         out = self.Maxpool(out)
#         t1 = out
#         ### Stage 2
#         out = self.encoder2(out)
#         out = self.Maxpool(out)
#         t2 = out
#         ### Stage 3
#         out = self.encoder3(out)
#         out = self.Maxpool(out)
#         t3 = out

#         ### Tokenized MLP Stage
#         ### Stage 4
#         out, H, W = self.patch_embed3(out)
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
#         pts = self.pts_head(out)
#         out = self.decoder2(out)
#         out = out + t1
#         edge = self.edge_head(out)
#         out = self.decoder1(out)
#         att = F.interpolate(edge, scale_factor=(2, 2), mode='bilinear', align_corners=False)
#         out = out * (1 + att)
#         out = self.decoder0(out)
#         out = self.Conv_1x1(out)
#         seg = torch.nn.Sigmoid()(out)
#         return [seg, edge, pts]
