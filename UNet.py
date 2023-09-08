import torch
from torch import nn

device = "cpu"

print(f"Using {device} device")


class UNetModel(nn.Module):
    def conv_down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def conv_up(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        return conv

    def __init__(self):
        super().__init__()
        self.s1 = self.conv_down(1, 64)
        self.s2 = self.conv_down(64, 128)
        self.s3 = self.conv_down(128, 256)
        self.s4 = self.conv_down(256, 512)

        self.b = self.conv_down(512, 1024)

        self.d1 = self.conv_up(1024, 512)
        self.d2 = self.conv_up(512, 256)
        self.d3 = self.conv_up(256, 128)
        self.d4 = self.conv_up(128, 64)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 22), 'constant', 0)
        x = x[:, :, :-1, :]

        bridge = []

        x = self.s1(x)
        bridge.append(x)
        x = self.s2(x)
        bridge.append(x)
        x = self.s3(x)
        bridge.append(x)
        x = self.s4(x)
        bridge.append(x)

        b = self.b(x)

        x = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)(b)
        z = bridge.pop()
        x = torch.cat([x, z], dim=1)
        x = self.d1(x)

        x = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)(x)
        z = bridge.pop()
        x = torch.cat([x, z], dim=1)
        x = self.d2(x)

        x = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)(x)
        z = bridge.pop()
        x = torch.cat([x, z], dim=1)
        x = self.d3(x)

        x = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)(x)
        z = bridge.pop()
        x = torch.cat([x, z], dim=1)
        x = self.d4(x)

        x = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0)(x)

        x = self.output(x)

        x = x[:, :, :, :-22]

        x = torch.cat((x, x[:, :, -1, :].unsqueeze(2)), dim=2)

        # x = torch.nn.functional.pad(x, (0, 0, 0, 1), 'replicate')

        return x


# model = UNetModel()
#
# a = torch.randn(1, 1, 513, 938)
# # x = torch.nn.functional.pad(a,  (0, 0, 0, 1), 'constant', 0)
# # a = 1
# model(a)