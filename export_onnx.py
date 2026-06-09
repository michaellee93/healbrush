"""
Export the Healbrush UNet to ONNX.

Run with:
  /home/michael/projects/pytorch/healbrush/.venv/bin/python3 export_onnx.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Model definition (must match main.py exactly) ─────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        down = self.pool(features)
        return features, down


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(4, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.bottleneck = ConvBlock(512, 512)
        self.up4 = Up(512, 512, 512)
        self.up3 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up1 = Up(128, 64, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)
        x = self.bottleneck(x)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return torch.sigmoid(self.out(x))


# ── Load checkpoint ────────────────────────────────────────────────────────────

ckpt_path = Path("./outputs/4layer/checkpoint.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

state_dict = ckpt["model"]

# torch.compile prefixes keys with "_orig_mod." — strip it if present
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

model = UNet()
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

# ── Export ─────────────────────────────────────────────────────────────────────

dummy = torch.zeros(1, 4, 256, 256)
out_path = Path("./outputs/4layer/healbrush.onnx")

torch.onnx.export(
    model,
    dummy,
    out_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

print(f"Exported to {out_path}")

# ── Print op summary ───────────────────────────────────────────────────────────

import onnx
from collections import Counter

m = onnx.load(out_path)
ops = Counter(node.op_type for node in m.graph.node)
print("\nOps in graph:")
for op, count in sorted(ops.items()):
    print(f"  {op:30s} x{count}")
