import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, features=(16, 32, 64)):
        super().__init__()
        self.features = features
        self.num_levels = len(features)

        # === 1) ENCODER ===
        # At each level:
        #   - conv1 (stride=1) → BN → ReLU  (we'll save this as "skip")
        #   - conv2 (stride=2) → BN → ReLU  (this actually halves H,W)
        self.encoders = nn.ModuleList()
        prev_ch = in_ch
        for f in features:
            enc_block = nn.ModuleDict({
                'conv1': nn.Conv2d(prev_ch, f, kernel_size=3, stride=1, padding=1),
                'bn1':   nn.BatchNorm2d(f),
                'conv2': nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
                'bn2':   nn.BatchNorm2d(f),
            })
            self.encoders.append(enc_block)
            prev_ch = f

        # === 2) BOTTLENECK ===
        # One more conv at the "bottom" (no downsampling here).
        bottleneck_ch = features[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )
        prev_ch = bottleneck_ch

        # === 3) DECODER ===
        # Each level:
        #   - upconv (ConvTranspose2d) to double spatial dims
        #   - add (elementwise) the matching skip
        #   - one conv (stride=1) → BN → ReLU
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            dec_block = nn.ModuleDict({
                'upconv': nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2),
                'conv': nn.Sequential(
                    nn.Conv2d(f, f, kernel_size=3, padding=1),
                    nn.BatchNorm2d(f),
                    nn.ReLU(inplace=True),
                )
            })
            self.decoders.append(dec_block)
            prev_ch = f

        # === 4) FINAL 1×1 CONV ===
        self.final = nn.Conv2d(prev_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """
        1) Pad the input so that H and W become multiples of 2**num_levels.
        2) Run through encoder/bottleneck/decoder (with addition‐based skips).
        3) Crop back to the original (H,W) before returning.
        """
        b, c, H, W = x.shape
        orig_h, orig_w = H, W

        with torch.no_grad():
            max_per_sample = x.abs().amax(dim=(1, 2, 3))           # shape: (B,)
            max_per_sample = max_per_sample.clamp(min=1e-6)        # avoid divide‐by‐zero

        # Reshape so we can broadcast: (B,) -> (B, 1, 1, 1)
        scale = max_per_sample.view(b, 1, 1, 1)                   # shape: (B,1,1,1)

        # Now divide each sample by its own scale:
        x = x / scale

        # check for nans or infs in scale
        if torch.isnan(scale).any() or torch.isinf(scale).any():
            print('Scale:', scale)
            print('Input x:', x)
            raise ValueError("NaN or Inf found in scale tensor")

        # --- 1) COMPUTE AND APPLY PADDING ---
        factor = 2 ** self.num_levels
        pad_h = (factor - (H % factor)) % factor
        pad_w = (factor - (W % factor)) % factor

        top    = pad_h // 2
        bottom = pad_h - top
        left   = pad_w // 2
        right  = pad_w - left

        if pad_h > 0 or pad_w > 0:
            # F.pad's order is (pad_left, pad_right, pad_top, pad_bottom)
            x = F.pad(x, (left, right, top, bottom))
            H = H + pad_h
            W = W + pad_w
        else:
            top = bottom = left = right = 0

        # --- 2) ENCODER PATH (grab skips BEFORE downsampling) ---
        skips = []
        for enc in self.encoders:
            # 2a) conv1 (stride=1) → BN → ReLU  → this is the "skip" of size (H_current, W_current)
            x = enc['conv1'](x)
            x = enc['bn1'](x)
            x = F.relu(x, inplace=True)
            skips.append(x)

            # 2b) conv2 (stride=2) → BN → ReLU  → downsample to (H/2, W/2)
            x = enc['conv2'](x)
            x = enc['bn2'](x)
            x = F.relu(x, inplace=True)

        # --- 3) BOTTLENECK (no spatial change) ---
        x = self.bottleneck(x)

        # --- 4) DECODER PATH ---
        # At each decoder level, we do:
        #   upconv → (H',W') doubles
        #   + skip (which has the same (H',W'))
        #   → conv (stride=1)
        for dec in self.decoders:
            x = dec['upconv'](x)
            skip = skips.pop()

            # After padding up‐front, shapes must match exactly.
            # (Optional: you can assert this if you want to double‐check.)
            if x.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    f"Shape mismatch in decoder: upconv is {x.shape[2:]}, skip is {skip.shape[2:]}."
                )

            x = x + skip
            x = dec['conv'](x)

        # --- 5) FINAL 1×1 CONV ---
        x = self.final(x)

        # --- 6) CROP BACK to (orig_h, orig_w) if padded ---
        if pad_h > 0 or pad_w > 0:
            x = x[..., top : top + orig_h, left : left + orig_w]

        # rescale
        x *= scale

        return x

if __name__ == "__main__":
    # figure out and print number of parameters
    model = UNet()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
