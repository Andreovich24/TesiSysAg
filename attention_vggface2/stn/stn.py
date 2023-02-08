import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):

    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            # nn.Conv2d(1, 8, kernel_size=3),
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            # nn.Linear(1000, 32),
            nn.Linear(100352, 32),  # 32 * 56 * 56
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 100352)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        # grid = F.affine_grid(theta, x.size(), align_corners=True)
        # x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        x = self.stn(x)
        return x
