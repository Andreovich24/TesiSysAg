import torch
import torch.nn as nn
import attention_vggface2.resnet50.vggface2 as model
import attention_vggface2.se.se as se_module


class VGGFace2SEGradCAM(nn.Module):

    def __init__(self):
        super().__init__()
        self.vggface2 = model.VGGFace2(pretrained=True)
        self.se1 = se_module.SE(256)
        self.se2 = se_module.SE(256)
        self.se3 = se_module.SE(256)
        self.se4 = se_module.SE(512)
        self.se5 = se_module.SE(512)
        self.se6 = se_module.SE(512)
        self.se7 = se_module.SE(512)
        self.se8 = se_module.SE(1024)
        self.se9 = se_module.SE(1024)
        self.se10 = se_module.SE(1024)
        self.se11 = se_module.SE(1024)
        self.se12 = se_module.SE(1024)
        self.se13 = se_module.SE(1024)
        self.se14 = se_module.SE(2048)
        self.se15 = se_module.SE(2048)
        self.se16 = se_module.SE(2048)

        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        conv1_7x7_s2 = self.vggface2.vggface2.conv1_7x7_s2(x)
        conv1_7x7_s2_bn = self.vggface2.vggface2.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.vggface2.vggface2.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.vggface2.vggface2.pool1_3x3_s2(conv1_7x7_s2_bnxx)

        conv2_1_1x1_reduce = self.vggface2.vggface2.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.vggface2.vggface2.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.vggface2.vggface2.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.vggface2.vggface2.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.vggface2.vggface2.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.vggface2.vggface2.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.vggface2.vggface2.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        se1x = self.se1(conv2_1_1x1_increase_bn)
        conv2_1_1x1_proj = self.vggface2.vggface2.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.vggface2.vggface2.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        # conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, se1x)
        conv2_1x = self.vggface2.vggface2.conv2_1_relu(conv2_1)

        conv2_2_1x1_reduce = self.vggface2.vggface2.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.vggface2.vggface2.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.vggface2.vggface2.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.vggface2.vggface2.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.vggface2.vggface2.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.vggface2.vggface2.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.vggface2.vggface2.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        se2x = self.se2(conv2_2_1x1_increase_bn)
        # conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2 = torch.add(conv2_1x, 1, se2x)
        conv2_2x = self.vggface2.vggface2.conv2_2_relu(conv2_2)

        conv2_3_1x1_reduce = self.vggface2.vggface2.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.vggface2.vggface2.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.vggface2.vggface2.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.vggface2.vggface2.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.vggface2.vggface2.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.vggface2.vggface2.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.vggface2.vggface2.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        se3x = self.se3(conv2_3_1x1_increase_bn)
        # conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3 = torch.add(conv2_2x, 1, se3x)
        conv2_3x = self.vggface2.vggface2.conv2_3_relu(conv2_3)

        conv3_1_1x1_reduce = self.vggface2.vggface2.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.vggface2.vggface2.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.vggface2.vggface2.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.vggface2.vggface2.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.vggface2.vggface2.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.vggface2.vggface2.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.vggface2.vggface2.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        se4x = self.se4(conv3_1_1x1_increase_bn)
        conv3_1_1x1_proj = self.vggface2.vggface2.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.vggface2.vggface2.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        # conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, se4x)
        conv3_1x = self.vggface2.vggface2.conv3_1_relu(conv3_1)

        conv3_2_1x1_reduce = self.vggface2.vggface2.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.vggface2.vggface2.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.vggface2.vggface2.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.vggface2.vggface2.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.vggface2.vggface2.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.vggface2.vggface2.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.vggface2.vggface2.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        se5x = self.se5(conv3_2_1x1_increase_bn)
        # conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2 = torch.add(conv3_1x, 1, se5x)
        conv3_2x = self.vggface2.vggface2.conv3_2_relu(conv3_2)

        conv3_3_1x1_reduce = self.vggface2.vggface2.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.vggface2.vggface2.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.vggface2.vggface2.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.vggface2.vggface2.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.vggface2.vggface2.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.vggface2.vggface2.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.vggface2.vggface2.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        se6x = self.se6(conv3_3_1x1_increase_bn)
        # conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3 = torch.add(conv3_2x, 1, se6x)
        conv3_3x = self.vggface2.vggface2.conv3_3_relu(conv3_3)

        conv3_4_1x1_reduce = self.vggface2.vggface2.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.vggface2.vggface2.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.vggface2.vggface2.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.vggface2.vggface2.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.vggface2.vggface2.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.vggface2.vggface2.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.vggface2.vggface2.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        se7x = self.se7(conv3_4_1x1_increase_bn)
        # conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4 = torch.add(conv3_3x, 1, se7x)
        conv3_4x = self.vggface2.vggface2.conv3_4_relu(conv3_4)

        conv4_1_1x1_reduce = self.vggface2.vggface2.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.vggface2.vggface2.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.vggface2.vggface2.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.vggface2.vggface2.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.vggface2.vggface2.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.vggface2.vggface2.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.vggface2.vggface2.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        se8x = self.se8(conv4_1_1x1_increase_bn)
        conv4_1_1x1_proj = self.vggface2.vggface2.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.vggface2.vggface2.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        # conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, se8x)
        conv4_1x = self.vggface2.vggface2.conv4_1_relu(conv4_1)

        conv4_2_1x1_reduce = self.vggface2.vggface2.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.vggface2.vggface2.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.vggface2.vggface2.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.vggface2.vggface2.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.vggface2.vggface2.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.vggface2.vggface2.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.vggface2.vggface2.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        se9x = self.se9(conv4_2_1x1_increase_bn)
        # conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2 = torch.add(conv4_1x, 1, se9x)
        conv4_2x = self.vggface2.vggface2.conv4_2_relu(conv4_2)

        conv4_3_1x1_reduce = self.vggface2.vggface2.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.vggface2.vggface2.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.vggface2.vggface2.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.vggface2.vggface2.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.vggface2.vggface2.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.vggface2.vggface2.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.vggface2.vggface2.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        se10x = self.se10(conv4_3_1x1_increase_bn)
        # conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3 = torch.add(conv4_2x, 1, se10x)
        conv4_3x = self.vggface2.vggface2.conv4_3_relu(conv4_3)

        conv4_4_1x1_reduce = self.vggface2.vggface2.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.vggface2.vggface2.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.vggface2.vggface2.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.vggface2.vggface2.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.vggface2.vggface2.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.vggface2.vggface2.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.vggface2.vggface2.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        se11x = self.se11(conv4_4_1x1_increase_bn)
        # conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4 = torch.add(conv4_3x, 1, se11x)
        conv4_4x = self.vggface2.vggface2.conv4_4_relu(conv4_4)

        conv4_5_1x1_reduce = self.vggface2.vggface2.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.vggface2.vggface2.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.vggface2.vggface2.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.vggface2.vggface2.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.vggface2.vggface2.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.vggface2.vggface2.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.vggface2.vggface2.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        se12x = self.se12(conv4_5_1x1_increase_bn)
        # conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5 = torch.add(conv4_4x, 1, se12x)
        conv4_5x = self.vggface2.vggface2.conv4_5_relu(conv4_5)

        conv4_6_1x1_reduce = self.vggface2.vggface2.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.vggface2.vggface2.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.vggface2.vggface2.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.vggface2.vggface2.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.vggface2.vggface2.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.vggface2.vggface2.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.vggface2.vggface2.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        se13x = self.se13(conv4_6_1x1_increase_bn)
        # conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6 = torch.add(conv4_5x, 1, se13x)
        conv4_6x = self.vggface2.vggface2.conv4_6_relu(conv4_6)

        conv5_1_1x1_reduce = self.vggface2.vggface2.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.vggface2.vggface2.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.vggface2.vggface2.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.vggface2.vggface2.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.vggface2.vggface2.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.vggface2.vggface2.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.vggface2.vggface2.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        se14x = self.se14(conv5_1_1x1_increase_bn)
        conv5_1_1x1_proj = self.vggface2.vggface2.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.vggface2.vggface2.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        # conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, se14x)
        conv5_1x = self.vggface2.vggface2.conv5_1_relu(conv5_1)

        conv5_2_1x1_reduce = self.vggface2.vggface2.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.vggface2.vggface2.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.vggface2.vggface2.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.vggface2.vggface2.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.vggface2.vggface2.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.vggface2.vggface2.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.vggface2.vggface2.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        se15x = self.se15(conv5_2_1x1_increase_bn)
        # conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2 = torch.add(conv5_1x, 1, se15x)
        conv5_2x = self.vggface2.vggface2.conv5_2_relu(conv5_2)

        conv5_3_1x1_reduce = self.vggface2.vggface2.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.vggface2.vggface2.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.vggface2.vggface2.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.vggface2.vggface2.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.vggface2.vggface2.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.vggface2.vggface2.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.vggface2.vggface2.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        se16x = self.se16(conv5_3_1x1_increase_bn)
        # conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3 = torch.add(conv5_2x, 1, se16x)
        conv5_3x = self.vggface2.vggface2.conv5_3_relu(conv5_3)

        # register the hook
        h = conv5_3x.register_hook(self.activations_hook)

        pool5_7x7_s1 = self.vggface2.vggface2.pool5_7x7_s1(conv5_3x)
        classifier_flatten = pool5_7x7_s1.view(pool5_7x7_s1.size(0), -1)
        # classifier_flatten = torch.flatten(pool5_7x7_s1)
        classifier = self.vggface2.vggface2.classifier(classifier_flatten)

        return classifier

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        conv1_7x7_s2 = self.vggface2.vggface2.conv1_7x7_s2(x)
        conv1_7x7_s2_bn = self.vggface2.vggface2.conv1_7x7_s2_bn(conv1_7x7_s2)
        conv1_7x7_s2_bnxx = self.vggface2.vggface2.conv1_relu_7x7_s2(conv1_7x7_s2_bn)
        pool1_3x3_s2 = self.vggface2.vggface2.pool1_3x3_s2(conv1_7x7_s2_bnxx)

        conv2_1_1x1_reduce = self.vggface2.vggface2.conv2_1_1x1_reduce(pool1_3x3_s2)
        conv2_1_1x1_reduce_bn = self.vggface2.vggface2.conv2_1_1x1_reduce_bn(conv2_1_1x1_reduce)
        conv2_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv2_1_1x1_reduce_relu(conv2_1_1x1_reduce_bn)
        conv2_1_3x3 = self.vggface2.vggface2.conv2_1_3x3(conv2_1_1x1_reduce_bnxx)
        conv2_1_3x3_bn = self.vggface2.vggface2.conv2_1_3x3_bn(conv2_1_3x3)
        conv2_1_3x3_bnxx = self.vggface2.vggface2.conv2_1_3x3_relu(conv2_1_3x3_bn)
        conv2_1_1x1_increase = self.vggface2.vggface2.conv2_1_1x1_increase(conv2_1_3x3_bnxx)
        conv2_1_1x1_increase_bn = self.vggface2.vggface2.conv2_1_1x1_increase_bn(conv2_1_1x1_increase)
        se1x = self.se1(conv2_1_1x1_increase_bn)
        conv2_1_1x1_proj = self.vggface2.vggface2.conv2_1_1x1_proj(pool1_3x3_s2)
        conv2_1_1x1_proj_bn = self.vggface2.vggface2.conv2_1_1x1_proj_bn(conv2_1_1x1_proj)
        # conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, conv2_1_1x1_increase_bn)
        conv2_1 = torch.add(conv2_1_1x1_proj_bn, 1, se1x)
        conv2_1x = self.vggface2.vggface2.conv2_1_relu(conv2_1)

        conv2_2_1x1_reduce = self.vggface2.vggface2.conv2_2_1x1_reduce(conv2_1x)
        conv2_2_1x1_reduce_bn = self.vggface2.vggface2.conv2_2_1x1_reduce_bn(conv2_2_1x1_reduce)
        conv2_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv2_2_1x1_reduce_relu(conv2_2_1x1_reduce_bn)
        conv2_2_3x3 = self.vggface2.vggface2.conv2_2_3x3(conv2_2_1x1_reduce_bnxx)
        conv2_2_3x3_bn = self.vggface2.vggface2.conv2_2_3x3_bn(conv2_2_3x3)
        conv2_2_3x3_bnxx = self.vggface2.vggface2.conv2_2_3x3_relu(conv2_2_3x3_bn)
        conv2_2_1x1_increase = self.vggface2.vggface2.conv2_2_1x1_increase(conv2_2_3x3_bnxx)
        conv2_2_1x1_increase_bn = self.vggface2.vggface2.conv2_2_1x1_increase_bn(conv2_2_1x1_increase)
        se2x = self.se2(conv2_2_1x1_increase_bn)
        # conv2_2 = torch.add(conv2_1x, 1, conv2_2_1x1_increase_bn)
        conv2_2 = torch.add(conv2_1x, 1, se2x)
        conv2_2x = self.vggface2.vggface2.conv2_2_relu(conv2_2)

        conv2_3_1x1_reduce = self.vggface2.vggface2.conv2_3_1x1_reduce(conv2_2x)
        conv2_3_1x1_reduce_bn = self.vggface2.vggface2.conv2_3_1x1_reduce_bn(conv2_3_1x1_reduce)
        conv2_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv2_3_1x1_reduce_relu(conv2_3_1x1_reduce_bn)
        conv2_3_3x3 = self.vggface2.vggface2.conv2_3_3x3(conv2_3_1x1_reduce_bnxx)
        conv2_3_3x3_bn = self.vggface2.vggface2.conv2_3_3x3_bn(conv2_3_3x3)
        conv2_3_3x3_bnxx = self.vggface2.vggface2.conv2_3_3x3_relu(conv2_3_3x3_bn)
        conv2_3_1x1_increase = self.vggface2.vggface2.conv2_3_1x1_increase(conv2_3_3x3_bnxx)
        conv2_3_1x1_increase_bn = self.vggface2.vggface2.conv2_3_1x1_increase_bn(conv2_3_1x1_increase)
        se3x = self.se3(conv2_3_1x1_increase_bn)
        # conv2_3 = torch.add(conv2_2x, 1, conv2_3_1x1_increase_bn)
        conv2_3 = torch.add(conv2_2x, 1, se3x)
        conv2_3x = self.vggface2.vggface2.conv2_3_relu(conv2_3)

        conv3_1_1x1_reduce = self.vggface2.vggface2.conv3_1_1x1_reduce(conv2_3x)
        conv3_1_1x1_reduce_bn = self.vggface2.vggface2.conv3_1_1x1_reduce_bn(conv3_1_1x1_reduce)
        conv3_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_1_1x1_reduce_relu(conv3_1_1x1_reduce_bn)
        conv3_1_3x3 = self.vggface2.vggface2.conv3_1_3x3(conv3_1_1x1_reduce_bnxx)
        conv3_1_3x3_bn = self.vggface2.vggface2.conv3_1_3x3_bn(conv3_1_3x3)
        conv3_1_3x3_bnxx = self.vggface2.vggface2.conv3_1_3x3_relu(conv3_1_3x3_bn)
        conv3_1_1x1_increase = self.vggface2.vggface2.conv3_1_1x1_increase(conv3_1_3x3_bnxx)
        conv3_1_1x1_increase_bn = self.vggface2.vggface2.conv3_1_1x1_increase_bn(conv3_1_1x1_increase)
        se4x = self.se4(conv3_1_1x1_increase_bn)
        conv3_1_1x1_proj = self.vggface2.vggface2.conv3_1_1x1_proj(conv2_3x)
        conv3_1_1x1_proj_bn = self.vggface2.vggface2.conv3_1_1x1_proj_bn(conv3_1_1x1_proj)
        # conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, conv3_1_1x1_increase_bn)
        conv3_1 = torch.add(conv3_1_1x1_proj_bn, 1, se4x)
        conv3_1x = self.vggface2.vggface2.conv3_1_relu(conv3_1)

        conv3_2_1x1_reduce = self.vggface2.vggface2.conv3_2_1x1_reduce(conv3_1x)
        conv3_2_1x1_reduce_bn = self.vggface2.vggface2.conv3_2_1x1_reduce_bn(conv3_2_1x1_reduce)
        conv3_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_2_1x1_reduce_relu(conv3_2_1x1_reduce_bn)
        conv3_2_3x3 = self.vggface2.vggface2.conv3_2_3x3(conv3_2_1x1_reduce_bnxx)
        conv3_2_3x3_bn = self.vggface2.vggface2.conv3_2_3x3_bn(conv3_2_3x3)
        conv3_2_3x3_bnxx = self.vggface2.vggface2.conv3_2_3x3_relu(conv3_2_3x3_bn)
        conv3_2_1x1_increase = self.vggface2.vggface2.conv3_2_1x1_increase(conv3_2_3x3_bnxx)
        conv3_2_1x1_increase_bn = self.vggface2.vggface2.conv3_2_1x1_increase_bn(conv3_2_1x1_increase)
        se5x = self.se5(conv3_2_1x1_increase_bn)
        # conv3_2 = torch.add(conv3_1x, 1, conv3_2_1x1_increase_bn)
        conv3_2 = torch.add(conv3_1x, 1, se5x)
        conv3_2x = self.vggface2.vggface2.conv3_2_relu(conv3_2)

        conv3_3_1x1_reduce = self.vggface2.vggface2.conv3_3_1x1_reduce(conv3_2x)
        conv3_3_1x1_reduce_bn = self.vggface2.vggface2.conv3_3_1x1_reduce_bn(conv3_3_1x1_reduce)
        conv3_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_3_1x1_reduce_relu(conv3_3_1x1_reduce_bn)
        conv3_3_3x3 = self.vggface2.vggface2.conv3_3_3x3(conv3_3_1x1_reduce_bnxx)
        conv3_3_3x3_bn = self.vggface2.vggface2.conv3_3_3x3_bn(conv3_3_3x3)
        conv3_3_3x3_bnxx = self.vggface2.vggface2.conv3_3_3x3_relu(conv3_3_3x3_bn)
        conv3_3_1x1_increase = self.vggface2.vggface2.conv3_3_1x1_increase(conv3_3_3x3_bnxx)
        conv3_3_1x1_increase_bn = self.vggface2.vggface2.conv3_3_1x1_increase_bn(conv3_3_1x1_increase)
        se6x = self.se6(conv3_3_1x1_increase_bn)
        # conv3_3 = torch.add(conv3_2x, 1, conv3_3_1x1_increase_bn)
        conv3_3 = torch.add(conv3_2x, 1, se6x)
        conv3_3x = self.vggface2.vggface2.conv3_3_relu(conv3_3)

        conv3_4_1x1_reduce = self.vggface2.vggface2.conv3_4_1x1_reduce(conv3_3x)
        conv3_4_1x1_reduce_bn = self.vggface2.vggface2.conv3_4_1x1_reduce_bn(conv3_4_1x1_reduce)
        conv3_4_1x1_reduce_bnxx = self.vggface2.vggface2.conv3_4_1x1_reduce_relu(conv3_4_1x1_reduce_bn)
        conv3_4_3x3 = self.vggface2.vggface2.conv3_4_3x3(conv3_4_1x1_reduce_bnxx)
        conv3_4_3x3_bn = self.vggface2.vggface2.conv3_4_3x3_bn(conv3_4_3x3)
        conv3_4_3x3_bnxx = self.vggface2.vggface2.conv3_4_3x3_relu(conv3_4_3x3_bn)
        conv3_4_1x1_increase = self.vggface2.vggface2.conv3_4_1x1_increase(conv3_4_3x3_bnxx)
        conv3_4_1x1_increase_bn = self.vggface2.vggface2.conv3_4_1x1_increase_bn(conv3_4_1x1_increase)
        se7x = self.se7(conv3_4_1x1_increase_bn)
        # conv3_4 = torch.add(conv3_3x, 1, conv3_4_1x1_increase_bn)
        conv3_4 = torch.add(conv3_3x, 1, se7x)
        conv3_4x = self.vggface2.vggface2.conv3_4_relu(conv3_4)

        conv4_1_1x1_reduce = self.vggface2.vggface2.conv4_1_1x1_reduce(conv3_4x)
        conv4_1_1x1_reduce_bn = self.vggface2.vggface2.conv4_1_1x1_reduce_bn(conv4_1_1x1_reduce)
        conv4_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_1_1x1_reduce_relu(conv4_1_1x1_reduce_bn)
        conv4_1_3x3 = self.vggface2.vggface2.conv4_1_3x3(conv4_1_1x1_reduce_bnxx)
        conv4_1_3x3_bn = self.vggface2.vggface2.conv4_1_3x3_bn(conv4_1_3x3)
        conv4_1_3x3_bnxx = self.vggface2.vggface2.conv4_1_3x3_relu(conv4_1_3x3_bn)
        conv4_1_1x1_increase = self.vggface2.vggface2.conv4_1_1x1_increase(conv4_1_3x3_bnxx)
        conv4_1_1x1_increase_bn = self.vggface2.vggface2.conv4_1_1x1_increase_bn(conv4_1_1x1_increase)
        se8x = self.se8(conv4_1_1x1_increase_bn)
        conv4_1_1x1_proj = self.vggface2.vggface2.conv4_1_1x1_proj(conv3_4x)
        conv4_1_1x1_proj_bn = self.vggface2.vggface2.conv4_1_1x1_proj_bn(conv4_1_1x1_proj)
        # conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, conv4_1_1x1_increase_bn)
        conv4_1 = torch.add(conv4_1_1x1_proj_bn, 1, se8x)
        conv4_1x = self.vggface2.vggface2.conv4_1_relu(conv4_1)

        conv4_2_1x1_reduce = self.vggface2.vggface2.conv4_2_1x1_reduce(conv4_1x)
        conv4_2_1x1_reduce_bn = self.vggface2.vggface2.conv4_2_1x1_reduce_bn(conv4_2_1x1_reduce)
        conv4_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_2_1x1_reduce_relu(conv4_2_1x1_reduce_bn)
        conv4_2_3x3 = self.vggface2.vggface2.conv4_2_3x3(conv4_2_1x1_reduce_bnxx)
        conv4_2_3x3_bn = self.vggface2.vggface2.conv4_2_3x3_bn(conv4_2_3x3)
        conv4_2_3x3_bnxx = self.vggface2.vggface2.conv4_2_3x3_relu(conv4_2_3x3_bn)
        conv4_2_1x1_increase = self.vggface2.vggface2.conv4_2_1x1_increase(conv4_2_3x3_bnxx)
        conv4_2_1x1_increase_bn = self.vggface2.vggface2.conv4_2_1x1_increase_bn(conv4_2_1x1_increase)
        se9x = self.se9(conv4_2_1x1_increase_bn)
        # conv4_2 = torch.add(conv4_1x, 1, conv4_2_1x1_increase_bn)
        conv4_2 = torch.add(conv4_1x, 1, se9x)
        conv4_2x = self.vggface2.vggface2.conv4_2_relu(conv4_2)

        conv4_3_1x1_reduce = self.vggface2.vggface2.conv4_3_1x1_reduce(conv4_2x)
        conv4_3_1x1_reduce_bn = self.vggface2.vggface2.conv4_3_1x1_reduce_bn(conv4_3_1x1_reduce)
        conv4_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_3_1x1_reduce_relu(conv4_3_1x1_reduce_bn)
        conv4_3_3x3 = self.vggface2.vggface2.conv4_3_3x3(conv4_3_1x1_reduce_bnxx)
        conv4_3_3x3_bn = self.vggface2.vggface2.conv4_3_3x3_bn(conv4_3_3x3)
        conv4_3_3x3_bnxx = self.vggface2.vggface2.conv4_3_3x3_relu(conv4_3_3x3_bn)
        conv4_3_1x1_increase = self.vggface2.vggface2.conv4_3_1x1_increase(conv4_3_3x3_bnxx)
        conv4_3_1x1_increase_bn = self.vggface2.vggface2.conv4_3_1x1_increase_bn(conv4_3_1x1_increase)
        se10x = self.se10(conv4_3_1x1_increase_bn)
        # conv4_3 = torch.add(conv4_2x, 1, conv4_3_1x1_increase_bn)
        conv4_3 = torch.add(conv4_2x, 1, se10x)
        conv4_3x = self.vggface2.vggface2.conv4_3_relu(conv4_3)

        conv4_4_1x1_reduce = self.vggface2.vggface2.conv4_4_1x1_reduce(conv4_3x)
        conv4_4_1x1_reduce_bn = self.vggface2.vggface2.conv4_4_1x1_reduce_bn(conv4_4_1x1_reduce)
        conv4_4_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_4_1x1_reduce_relu(conv4_4_1x1_reduce_bn)
        conv4_4_3x3 = self.vggface2.vggface2.conv4_4_3x3(conv4_4_1x1_reduce_bnxx)
        conv4_4_3x3_bn = self.vggface2.vggface2.conv4_4_3x3_bn(conv4_4_3x3)
        conv4_4_3x3_bnxx = self.vggface2.vggface2.conv4_4_3x3_relu(conv4_4_3x3_bn)
        conv4_4_1x1_increase = self.vggface2.vggface2.conv4_4_1x1_increase(conv4_4_3x3_bnxx)
        conv4_4_1x1_increase_bn = self.vggface2.vggface2.conv4_4_1x1_increase_bn(conv4_4_1x1_increase)
        se11x = self.se11(conv4_4_1x1_increase_bn)
        # conv4_4 = torch.add(conv4_3x, 1, conv4_4_1x1_increase_bn)
        conv4_4 = torch.add(conv4_3x, 1, se11x)
        conv4_4x = self.vggface2.vggface2.conv4_4_relu(conv4_4)

        conv4_5_1x1_reduce = self.vggface2.vggface2.conv4_5_1x1_reduce(conv4_4x)
        conv4_5_1x1_reduce_bn = self.vggface2.vggface2.conv4_5_1x1_reduce_bn(conv4_5_1x1_reduce)
        conv4_5_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_5_1x1_reduce_relu(conv4_5_1x1_reduce_bn)
        conv4_5_3x3 = self.vggface2.vggface2.conv4_5_3x3(conv4_5_1x1_reduce_bnxx)
        conv4_5_3x3_bn = self.vggface2.vggface2.conv4_5_3x3_bn(conv4_5_3x3)
        conv4_5_3x3_bnxx = self.vggface2.vggface2.conv4_5_3x3_relu(conv4_5_3x3_bn)
        conv4_5_1x1_increase = self.vggface2.vggface2.conv4_5_1x1_increase(conv4_5_3x3_bnxx)
        conv4_5_1x1_increase_bn = self.vggface2.vggface2.conv4_5_1x1_increase_bn(conv4_5_1x1_increase)
        se12x = self.se12(conv4_5_1x1_increase_bn)
        # conv4_5 = torch.add(conv4_4x, 1, conv4_5_1x1_increase_bn)
        conv4_5 = torch.add(conv4_4x, 1, se12x)
        conv4_5x = self.vggface2.vggface2.conv4_5_relu(conv4_5)

        conv4_6_1x1_reduce = self.vggface2.vggface2.conv4_6_1x1_reduce(conv4_5x)
        conv4_6_1x1_reduce_bn = self.vggface2.vggface2.conv4_6_1x1_reduce_bn(conv4_6_1x1_reduce)
        conv4_6_1x1_reduce_bnxx = self.vggface2.vggface2.conv4_6_1x1_reduce_relu(conv4_6_1x1_reduce_bn)
        conv4_6_3x3 = self.vggface2.vggface2.conv4_6_3x3(conv4_6_1x1_reduce_bnxx)
        conv4_6_3x3_bn = self.vggface2.vggface2.conv4_6_3x3_bn(conv4_6_3x3)
        conv4_6_3x3_bnxx = self.vggface2.vggface2.conv4_6_3x3_relu(conv4_6_3x3_bn)
        conv4_6_1x1_increase = self.vggface2.vggface2.conv4_6_1x1_increase(conv4_6_3x3_bnxx)
        conv4_6_1x1_increase_bn = self.vggface2.vggface2.conv4_6_1x1_increase_bn(conv4_6_1x1_increase)
        se13x = self.se13(conv4_6_1x1_increase_bn)
        # conv4_6 = torch.add(conv4_5x, 1, conv4_6_1x1_increase_bn)
        conv4_6 = torch.add(conv4_5x, 1, se13x)
        conv4_6x = self.vggface2.vggface2.conv4_6_relu(conv4_6)

        conv5_1_1x1_reduce = self.vggface2.vggface2.conv5_1_1x1_reduce(conv4_6x)
        conv5_1_1x1_reduce_bn = self.vggface2.vggface2.conv5_1_1x1_reduce_bn(conv5_1_1x1_reduce)
        conv5_1_1x1_reduce_bnxx = self.vggface2.vggface2.conv5_1_1x1_reduce_relu(conv5_1_1x1_reduce_bn)
        conv5_1_3x3 = self.vggface2.vggface2.conv5_1_3x3(conv5_1_1x1_reduce_bnxx)
        conv5_1_3x3_bn = self.vggface2.vggface2.conv5_1_3x3_bn(conv5_1_3x3)
        conv5_1_3x3_bnxx = self.vggface2.vggface2.conv5_1_3x3_relu(conv5_1_3x3_bn)
        conv5_1_1x1_increase = self.vggface2.vggface2.conv5_1_1x1_increase(conv5_1_3x3_bnxx)
        conv5_1_1x1_increase_bn = self.vggface2.vggface2.conv5_1_1x1_increase_bn(conv5_1_1x1_increase)
        se14x = self.se14(conv5_1_1x1_increase_bn)
        conv5_1_1x1_proj = self.vggface2.vggface2.conv5_1_1x1_proj(conv4_6x)
        conv5_1_1x1_proj_bn = self.vggface2.vggface2.conv5_1_1x1_proj_bn(conv5_1_1x1_proj)
        # conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, conv5_1_1x1_increase_bn)
        conv5_1 = torch.add(conv5_1_1x1_proj_bn, 1, se14x)
        conv5_1x = self.vggface2.vggface2.conv5_1_relu(conv5_1)

        conv5_2_1x1_reduce = self.vggface2.vggface2.conv5_2_1x1_reduce(conv5_1x)
        conv5_2_1x1_reduce_bn = self.vggface2.vggface2.conv5_2_1x1_reduce_bn(conv5_2_1x1_reduce)
        conv5_2_1x1_reduce_bnxx = self.vggface2.vggface2.conv5_2_1x1_reduce_relu(conv5_2_1x1_reduce_bn)
        conv5_2_3x3 = self.vggface2.vggface2.conv5_2_3x3(conv5_2_1x1_reduce_bnxx)
        conv5_2_3x3_bn = self.vggface2.vggface2.conv5_2_3x3_bn(conv5_2_3x3)
        conv5_2_3x3_bnxx = self.vggface2.vggface2.conv5_2_3x3_relu(conv5_2_3x3_bn)
        conv5_2_1x1_increase = self.vggface2.vggface2.conv5_2_1x1_increase(conv5_2_3x3_bnxx)
        conv5_2_1x1_increase_bn = self.vggface2.vggface2.conv5_2_1x1_increase_bn(conv5_2_1x1_increase)
        se15x = self.se15(conv5_2_1x1_increase_bn)
        # conv5_2 = torch.add(conv5_1x, 1, conv5_2_1x1_increase_bn)
        conv5_2 = torch.add(conv5_1x, 1, se15x)
        conv5_2x = self.vggface2.vggface2.conv5_2_relu(conv5_2)

        conv5_3_1x1_reduce = self.vggface2.vggface2.conv5_3_1x1_reduce(conv5_2x)
        conv5_3_1x1_reduce_bn = self.vggface2.vggface2.conv5_3_1x1_reduce_bn(conv5_3_1x1_reduce)
        conv5_3_1x1_reduce_bnxx = self.vggface2.vggface2.conv5_3_1x1_reduce_relu(conv5_3_1x1_reduce_bn)
        conv5_3_3x3 = self.vggface2.vggface2.conv5_3_3x3(conv5_3_1x1_reduce_bnxx)
        conv5_3_3x3_bn = self.vggface2.vggface2.conv5_3_3x3_bn(conv5_3_3x3)
        conv5_3_3x3_bnxx = self.vggface2.vggface2.conv5_3_3x3_relu(conv5_3_3x3_bn)
        conv5_3_1x1_increase = self.vggface2.vggface2.conv5_3_1x1_increase(conv5_3_3x3_bnxx)
        conv5_3_1x1_increase_bn = self.vggface2.vggface2.conv5_3_1x1_increase_bn(conv5_3_1x1_increase)
        se16x = self.se16(conv5_3_1x1_increase_bn)
        # conv5_3 = torch.add(conv5_2x, 1, conv5_3_1x1_increase_bn)
        conv5_3 = torch.add(conv5_2x, 1, se16x)
        conv5_3x = self.vggface2.vggface2.conv5_3_relu(conv5_3)

        return conv5_3x


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGGFace2SEGradCAM().to(device)
    print("Model archticture: ", model)

    x = np.random.rand(224, 224, 3)
    x = Image.fromarray(x.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        out = model(transforms.ToTensor()(x).unsqueeze_(0))
        print(out)
