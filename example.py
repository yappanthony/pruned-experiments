from ultralytics import YOLO
import torch
from torch import nn
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck

# This class is required to be present in the file where you use the pruned weights
class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

base_model = YOLO('base-v8nano-50ep-16bs/weights/best.pt')
pruned_model = YOLO('10ep-10pr-1iter/weights/best.pt') # PR = prune rate

print(f"Base model no. of params: {sum(p.numel() for p in base_model.parameters())}")
print("-------------------------------------------------------------")
print(f"Pruned model no. of params: {sum(p.numel() for p in pruned_model.parameters())}")