import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(channel, channel // reduction)
        self.bn = nn.BatchNorm1d(channel // reduction)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(channel // reduction, channel)

    def forward(self, x):
        y = self.gap(x)
        y = y.view(y.size(0), -1)
        y = self.linear1(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = y.unsqueeze(2).unsqueeze(3)
        y = y.expand_as(x)
        return y


class SpatialAttention(nn.Module):

    def __init__(self, channel):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.conv1 = nn.Conv2d(channel, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_max = torch.max(x, 1)[0].unsqueeze(1)
        x_avg = torch.mean(x, 1).unsqueeze(1)
        x_conv = self.conv1(x)
        x_conv = self.relu(x_conv)
        x_concat = torch.cat((x_max, x_avg, x_conv), dim=1)
        x_conv2 = self.conv2(x_concat)
        y = x_conv2.expand_as(x)
        return y


class AttentionBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(AttentionBlock, self).__init__()
        self.spatial_attention = SpatialAttention(channel)
        self.channel_attention = ChannelAttention(channel, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sa = self.spatial_attention(x)
        ca = self.channel_attention(x)
        att = 1 + self.sigmoid(ca * sa)
        return att * x


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = AttentionBlock(3, 1).to(device)
    model = AttentionBlock(64, 16).to(device)
    print("Model archticture: ", model)

    # x = np.random.rand(224, 224, 3)
    x = np.random.rand(5, 64, 224, 224)
    # x = Image.fromarray(x.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        # out = model(transforms.ToTensor()(x).unsqueeze_(0))
        out = model(torch.from_numpy(x).float().to(device))
        print(out)
