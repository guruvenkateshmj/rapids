# Piece 3: the U-Net.
# Design (decided in our architecture talk):
#   - 3 levels down, bottleneck at 1/8 size  (512 -> 256 -> 128 -> 64)
#   - channels 32 -> 64 -> 128 -> 256
#   - every block: (Conv 3x3 -> BatchNorm -> ReLU) x 2
#   - upsampling: bilinear stretch, then the block's convs refine it
#   - skip connections: decoder gets the matching encoder output
#   - output: 2 channels (dry, water) per pixel, raw scores ("logits")
#
# Test it on its own:  python unet_model.py

import torch
import torch.nn as nn


def double_conv(in_ch, out_ch):
    # bias=False: BatchNorm right after already adds its own shift,
    # so a conv bias would be a wasted, useless parameter.
    # inplace=True: ReLU overwrites its input instead of copying it,
    # which saves GPU memory.
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):

    def __init__(self, in_channels=2, n_classes=2, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8   # 32, 64, 128, 256

        # Encoder: image shrinks, channels grow.
        self.enc1 = double_conv(in_channels, c1)
        self.enc2 = double_conv(c1, c2)
        self.enc3 = double_conv(c2, c3)
        self.bottleneck = double_conv(c3, c4)

        # Decoder: input = upsampled features + skip features, stacked.
        # That's why the in-channels are sums (e.g. 256 + 128).
        self.dec3 = double_conv(c4 + c3, c3)
        self.dec2 = double_conv(c3 + c2, c2)
        self.dec1 = double_conv(c2 + c1, c1)

        # 1x1 conv: turns 32 features per pixel into 2 scores per pixel.
        self.head = nn.Conv2d(c1, n_classes, kernel_size=1)

        # No learnable weights in these two, so one copy is reused everywhere.
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        # 3 halvings must land on whole numbers, so H and W must divide by 8.
        assert x.shape[-2] % 8 == 0 and x.shape[-1] % 8 == 0, \
            "height and width must be divisible by 8"

        # Shapes for a 256 x 256 input, batch size B:
        e1 = self.enc1(x)                       # (B,  32, 256, 256)
        e2 = self.enc2(self.pool(e1))           # (B,  64, 128, 128)
        e3 = self.enc3(self.pool(e2))           # (B, 128,  64,  64)
        b = self.bottleneck(self.pool(e3))      # (B, 256,  32,  32)

        # torch.cat(..., dim=1) stacks along the channel axis = skip connection.
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))    # (B, 128,  64,  64)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))   # (B,  64, 128, 128)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))   # (B,  32, 256, 256)

        # Raw scores, no softmax here: CrossEntropyLoss applies it itself,
        # in a more numerically precise way than doing it twice.
        return self.head(d1)                    # (B,   2, 256, 256)


# --- Self-test: runs only with `python unet_model.py` ------------------
if __name__ == "__main__":
    import time

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    def wait_for_gpu():
        # The GPU works in the background; wait for it so timings are true.
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Device:", device)
    print(f"Parameters: {n_params:,}")

    model.eval()
    with torch.no_grad():
        # Training size and evaluation size: same weights, both must work.
        for batch, size in [(8, 256), (1, 512)]:
            x = torch.randn(batch, 2, size, size, device=device)
            model(x)                            # warm-up (first run is slow)
            wait_for_gpu()
            start = time.time()
            out = model(x)
            wait_for_gpu()
            ms = 1000 * (time.time() - start)
            print(f"  in {tuple(x.shape)} -> out {tuple(out.shape)}   {ms:.0f} ms")
