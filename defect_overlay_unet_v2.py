# ================= defect_overlay_cpu_v2.py ===========================
"""
Fast CPU-only overlay trainer.

▪ Tiny-U-Net (3 levels, 16 channels) trained at 256 px by default.
▪ Edge-aware loss (Sobel) + L1 + BCE; TV loss off for speed.
▪ Four-row preview per epoch:
      1) real defect
      2) true overlay on white
      3) predicted overlay on white
      4) predicted overlay on 1024-px GAN background
"""

import os, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------ CONFIG -------------------------------------
ROOT_GOOD   = "data/good"
ROOT_DEFECT = "data/defect"
GEN_WEIGHTS = "outputs/fake_good_v6/generator_good_v6.pt"

RES_TRAIN   = 256        # 256 or 512 for training speed
RES_FULL    = 1024       # background GAN native resolution
EPOCHS      = 12
BATCH       = 4
LR          = 1e-3
EDGE_W      = 0.3        # weight of edge (Sobel) loss term
TV_W        = 0.0        # set >0 if you want TV regulariser
THRESH      = 0.08
PREVIEW_N   = 4          # columns in the preview grid

OUT_DIR     = "outputs/defect_overlay_cpu_v2"
CKPT_DIR    = f"{OUT_DIR}/ckpt_v2"

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count() - 1, 1))
DEV = torch.device("cpu")

# --------------------- TRANSFORM -------------------------------------
class PadSquare:
    """Pad to a square, then we’ll resize."""
    def __init__(self, size): self.size = size
    def __call__(self, img):
        w, h = img.size
        side = max(w, h, self.size)
        pad  = [(side - w) // 2, (side - h) // 2,
                side - w - (side - w) // 2,
                side - h - (side - h) // 2]
        return T.functional.pad(img, pad, padding_mode='reflect')

xform = T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN, RES_TRAIN),
             interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))          # → [-1,1]
])

ALLOWED = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
def list_imgs(root):
    return [str(p) for p in Path(root).rglob('*')
            if p.suffix.lower() in ALLOWED]

@torch.no_grad()
def build_overlay(defect, bg):
    delta = defect - bg
    mask  = (delta.abs() > THRESH).float()
    return mask * delta, mask

# -------------------- DATASET ----------------------------------------
class OverlayDS(Dataset):
    def __init__(self):
        self.good   = list_imgs(ROOT_GOOD)
        self.defect = list_imgs(ROOT_DEFECT)
        if not self.good or not self.defect:
            raise RuntimeError("No images found in data/good or data/defect.")
        self.N = max(len(self.good), len(self.defect))
    def __len__(self): return self.N
    def __getitem__(self, idx):
        g = xform(Image.open(random.choice(self.good)).convert('L'))
        d = xform(Image.open(random.choice(self.defect)).convert('L'))
        ov, mask = build_overlay(d, g)
        return g, ov, mask

# ------------------ BACKGROUND GAN -----------------------------------
NZ, NGF = 100, 64
def make_generator():
    layers = [nn.ConvTranspose2d(NZ, NGF*16, 4, 1, 0, bias=False),
              nn.BatchNorm2d(NGF*16), nn.ReLU(True)]
    out_c = NGF*16
    for _ in range(int(math.log2(RES_FULL) - 2)):
        in_c, out_c = out_c, max(NGF, out_c // 2)
        layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(out_c), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(out_c, 1, 3, 1, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

Gbg = make_generator().to(DEV).eval()
Gbg.load_state_dict(torch.load(GEN_WEIGHTS, map_location=DEV))

@torch.no_grad()
def sample_gan(n):
    z  = torch.randn(n, NZ, 1, 1, device=DEV)
    bg = Gbg(z)                                      # full-res (1024×1024)
    bg_small = F.interpolate(bg, size=(RES_TRAIN, RES_TRAIN),
                             mode='bilinear', align_corners=False) \
               if RES_TRAIN != RES_FULL else bg
    return bg, bg_small

# -------------------- TINY U-NET (3-level) ---------------------------
def down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(o),
                         nn.LeakyReLU(0.2, True))
def up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(o),
                         nn.ReLU(True))

class TinyUNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.e1 = down(1,     ch)
        self.e2 = down(ch,    ch*2)
        self.e3 = down(ch*2,  ch*4)

        self.d3 = up(ch*4,  ch*2)
        self.d2 = up(ch*4,  ch)
        self.d1 = nn.ConvTranspose2d(ch*2, 2, 4, 2, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        d3 = self.d3(e3)
        d2 = self.d2(torch.cat([d3, e2], 1))
        d1 = self.d1(torch.cat([d2, e1], 1))

        mask  = torch.sigmoid(d1[:, :1])   # 0–1
        delta = torch.tanh(   d1[:, 1:])   # −1–1
        return mask * delta, mask

# -------------------- EDGE (Sobel) LOSS ------------------------------
def sobel_filters():
    gx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    gy = gx.permute(0, 1, 3, 2).flip(2)
    return gx, gy

Gx, Gy = sobel_filters()
Gx, Gy = Gx.to(DEV), Gy.to(DEV)

def edge_loss(pred, tgt):
    def grad(x):
        gxx = F.conv2d(x, Gx, padding=1)
        gyy = F.conv2d(x, Gy, padding=1)
        return gxx ** 2 + gyy ** 2
    return F.l1_loss(grad(pred), grad(tgt))

def tv_loss(m):
    return F.l1_loss(m[:, :, :, :-1], m[:, :, :, 1:]) + \
           F.l1_loss(m[:, :, :-1, :], m[:, :, 1:, :])

# -------------------- TRAIN ------------------------------------------
def main():
    ds = OverlayDS()
    loader = DataLoader(ds, BATCH, shuffle=True, num_workers=0)

    net = TinyUNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), LR)

    l1_log, edge_log = [], []

    for ep in range(1, EPOCHS + 1):
        l1_sum = edge_sum = 0.0

        for bg, ov_gt, mask_gt in tqdm(loader, desc=f"Epoch {ep}/{EPOCHS}"):
            bg, ov_gt, mask_gt = bg.to(DEV), ov_gt.to(DEV), mask_gt.to(DEV)

            ov_pred, mask_pred = net(bg)

            loss = ( F.l1_loss(ov_pred, ov_gt) +
                     EDGE_W * edge_loss(ov_pred, ov_gt) +
                     F.binary_cross_entropy(mask_pred, mask_gt) )
            if TV_W: loss += TV_W * tv_loss(mask_pred)

            opt.zero_grad()
            loss.backward()
            opt.step()

            l1_sum   += F.l1_loss(ov_pred, ov_gt).item()
            edge_sum += edge_loss(ov_pred, ov_gt).item()

        l1_log.append(l1_sum / len(loader))
        edge_log.append(edge_sum / len(loader))

        # --------------- PREVIEW GRID ---------------------------------
        with torch.no_grad():
            # 1) real defect batch
            real_bg, real_ov, _ = next(iter(loader))
            real_bg, real_ov = real_bg.to(DEV)[:PREVIEW_N], real_ov.to(DEV)[:PREVIEW_N]
            real_defect = torch.clamp(real_bg + real_ov, -1, 1)

            # 2) true overlay on white
            white = torch.ones_like(real_bg)
            true_on_white = torch.clamp(white + real_ov, -1, 1)

            # 3) generated overlay on white
            ov_pred_small, _ = net(real_bg)
            gen_on_white = torch.clamp(white + ov_pred_small, -1, 1)

            # 4) generated overlay on GAN background (full 1024)
            bg_full, _ = sample_gan(PREVIEW_N)
            ov_pred_full = F.interpolate(ov_pred_small,
                                         size=(RES_FULL, RES_FULL),
                                         mode='bilinear', align_corners=False)
            gen_on_gan = torch.clamp(bg_full + ov_pred_full, -1, 1)

            # upsample rows 1–3 to 1024 so widths match row 4
            real_defect_up = F.interpolate(real_defect,
                                           size=(RES_FULL, RES_FULL),
                                           mode='bilinear', align_corners=False)
            true_on_white_up = F.interpolate(true_on_white,
                                             size=(RES_FULL, RES_FULL),
                                             mode='bilinear', align_corners=False)
            gen_on_white_up  = F.interpolate(gen_on_white,
                                             size=(RES_FULL, RES_FULL),
                                             mode='bilinear', align_corners=False)

            # make horizontal strips
            g1 = vutils.make_grid(real_defect_up,   nrow=PREVIEW_N, padding=2)
            g2 = vutils.make_grid(true_on_white_up, nrow=PREVIEW_N, padding=2)
            g3 = vutils.make_grid(gen_on_white_up,  nrow=PREVIEW_N, padding=2)
            g4 = vutils.make_grid(gen_on_gan,       nrow=PREVIEW_N, padding=2)

            preview = torch.cat([g1, g2, g3, g4], dim=1)         # stack vertically
            vutils.save_image(preview * 0.5 + 0.5,
                              f"{OUT_DIR}/side_by_side_{ep:02}.png",
                              normalize=False)

        # --------------- CKPT & LOG -----------------------------------
        torch.save(net.state_dict(), f"{CKPT_DIR}/unet_{ep:02}.pt")
        print(f"Epoch {ep} done – preview & ckpt saved.")

    # ----- CSV & loss plot --------------------------------------------
    with open(f"{OUT_DIR}/losses.csv", "w", newline='') as f:
        csv.writer(f).writerows(
            [("epoch", "L1_overlay", "edge")] +
            [(i + 1, l1_log[i], edge_log[i]) for i in range(len(l1_log))])

    plt.plot(l1_log, label="L1"); plt.plot(edge_log, label="Edge")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/loss_plot.png"); plt.close()
    print("✅ Training complete – check", OUT_DIR)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
# =====================================================================
