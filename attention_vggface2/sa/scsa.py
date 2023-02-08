import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.channel = ChannelSelfAttention(in_dim)
        self.spatial = SpatialSelfAttention(in_dim)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # c, _ = self.channel(x)
        # s, _ = self.spatial(x)
        c, _ = self.channel(x)
        s, _ = self.spatial(x)



        # TODO: lavorare con attention piuttosto che il valore moltiplicato
        # att = 1 + torch.sigmoid(c * s)
        att = 1 + torch.sigmoid((c * self.gamma) * (s * self.delta))

        # TODO: vedere quale funziona meglio
        return att * x
        # return att
        # return (c * self.gamma) + (s * self.delta) + x


class ChannelSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(ChannelSelfAttention, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        query = x.view(m_batchsize, -1, height * width)
        key = query.permute(0, 2, 1)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)

        value = query

        out = torch.bmm(attention.permute(0, 2, 1), value)
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma * out + x
        # out = self.gamma * out

        return out, attention


class SpatialSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SpatialSelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # out = self.gamma * out + x
        # out = self.gamma * out

        return out, attention


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sa = SpatialSelfAttention(64).to(device)
    # sa = ChannelSelfAttention(64).to(device)
    sa = SelfAttention(64).to(device)
    print("Model archticture: ", sa)

    # x = np.random.rand(224, 224, 3)
    # x = Image.fromarray(x.astype(np.uint8))
    x = torch.rand(4, 64, 32, 32).to(device)

    sa.eval()
    with torch.no_grad():
        # out = sa(transforms.ToTensor()(x).unsqueeze_(0))
        out = sa(x)
        # print(out[0].size())
        print(out.size())
