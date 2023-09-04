import torch
from torch import nn

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        #                 padding_mode='zeros', device=None, dtype=None)
        self.s1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
        self.s1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
