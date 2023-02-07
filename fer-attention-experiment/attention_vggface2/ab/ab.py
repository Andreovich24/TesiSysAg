import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):

    # def __init__(self):
    #     super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()

        reducted_channels = gate_channel // reduction_ratio

        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, reducted_channels,
                                                                kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(reducted_channels))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # for i in range(dilation_conv_num):
        #     self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio,
        #                                                               gate_channel // reduction_ratio, kernel_size=3,
        #                                                               padding=dilation_val, dilation=dilation_val))
        #     self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
        #     self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())

        # ae_in_ch = gate_channel // reduction_ratio

        # self.gate_s.add_module('gate_s_conv_di_0', nn.Conv2d(ae_in_ch, ae_in_ch * 2, kernel_size=3, padding=dilation_val, dilation=dilation_val))
        self.gate_s.add_module('gate_s_conv_di_0', nn.Conv2d(reducted_channels, reducted_channels, kernel_size=3, padding=dilation_val, dilation=dilation_val))
        ## self.gate_s.add_module('gate_s_bn_di_0', nn.BatchNorm2d(ae_in_ch * 2))
        self.gate_s.add_module('gate_s_bn_di_0', nn.BatchNorm2d(reducted_channels))
        self.gate_s.add_module('gate_s_relu_di_0', nn.ReLU())
        self.gate_s.add_module('gate_s_pool_di_0', nn.MaxPool2d(2, 2))

        # self.gate_s.add_module('gate_s_conv_di_1', nn.Conv2d(ae_in_ch * 2, ae_in_ch * 2 // 4, kernel_size=3, padding=dilation_val, dilation=dilation_val))
        self.gate_s.add_module('gate_s_conv_di_1', nn.Conv2d(reducted_channels, reducted_channels, kernel_size=3, padding=dilation_val, dilation=dilation_val))
        ## self.gate_s.add_module('gate_s_bn_di_1', nn.BatchNorm2d(ae_in_ch * 2 / 4))
        self.gate_s.add_module('gate_s_bn_di_1', nn.BatchNorm2d(reducted_channels))
        self.gate_s.add_module('gate_s_relu_di_1', nn.ReLU())
        self.gate_s.add_module('gate_s_pool_di_1', nn.MaxPool2d(2, 2))

        ## self.gate_s.add_module('gate_s_deconv_di_2', nn.ConvTranspose2d(ae_in_ch * 2 / 4, ae_in_ch * 2, 2, stride = 2, dilation = dilation_val))
        # self.gate_s.add_module('gate_s_deconv_di_2', nn.ConvTranspose2d(ae_in_ch * 2 // 4, ae_in_ch * 2, 2, stride = 2))
        if gate_channel == 1024:
            self.gate_s.add_module('gate_s_deconv_di_2', nn.ConvTranspose2d(reducted_channels, reducted_channels, 2, stride = 2, output_padding = 1))
        else:
            self.gate_s.add_module('gate_s_deconv_di_2',
                                   nn.ConvTranspose2d(reducted_channels, reducted_channels, 2, stride = 2))
        self.gate_s.add_module('gate_s_relu_di_2', nn.ReLU())

        # self.gate_s.add_module('gate_s_deconv_di_3', nn.ConvTranspose2d(ae_in_ch * 2, ae_in_ch, 2, stride = 2, dilation = dilation_val))
        # self.gate_s.add_module('gate_s_deconv_di_3', nn.ConvTranspose2d(ae_in_ch * 2, ae_in_ch, 2, stride = 2))
        self.gate_s.add_module('gate_s_deconv_di_3', nn.ConvTranspose2d(reducted_channels, reducted_channels, 2, stride = 2))
        self.gate_s.add_module('gate_s_relu_di_3', nn.ReLU())

        # self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))
        # self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(ae_in_ch, 1, kernel_size=1))
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(reducted_channels, 1, kernel_size=1))

    def forward(self, in_tensor):
        # print("input", in_tensor.size())
        # x = self.gate_s(in_tensor)
        # print("after gate", x.size())
        # x = x.expand_as(in_tensor)
        # print("output", x.size())
        # return x
        return self.gate_s(in_tensor).expand_as(in_tensor)


class AB(nn.Module):

    def __init__(self, gate_channel):
        super(AB, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        # att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        att = 1 + torch.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AB(64 * 4).to(device)
    print("Model archticture: ", model)

    # x = np.random.rand(224, 224, 256)
    x = torch.rand(256, 224, 224)
    # x = Image.fromarray(x.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        # out = model(transforms.ToTensor()(x).unsqueeze_(0))
        out = model(x.unsqueeze_(0))
        print(out)
