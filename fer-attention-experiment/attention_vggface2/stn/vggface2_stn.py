import torch
import torch.nn as nn
import attention_vggface2.resnet50.vggface2 as model
import attention_vggface2.stn.stn as stn_module


class VGGFace2STN(nn.Module):

    def __init__(self, classes=7):
        super().__init__()
        self.vggface2 = model.VGGFace2(pretrained=True, classes=classes)
        self.stn = stn_module.STN()

    def forward(self, x):
        x = self.stn(x)
        x = self.vggface2(x)
        return x


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGGFace2STN().to(device)
    print("Model archticture: ", model)

    x = np.random.rand(224, 224, 3)
    x = Image.fromarray(x.astype(np.uint8))

    model.eval()
    with torch.no_grad():
        out = model(transforms.ToTensor()(x).unsqueeze_(0))
        print(out)
