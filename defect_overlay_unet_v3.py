# ===================== defect_overlay_cpu_v3.py ======================
"""
CPU-only overlay trainer (fast) with:
  • Tiny-U-Net (3 levels, 16 channels) trained at 256 px   ─ change
      RES_TRAIN to 512 if you want.
  • Stronger threshold + morphological cleanup on masks.
  • Loss =  L1 + BCE  +  EDGE_W * Sobel-edge L1
            + SPARSE_W * mean(mask)  (+ optional TV)
  • Four-row preview grid each epoch:
        1) real defect              (real good + true overlay)
        2) true overlay on white
        3) generated overlay on white
        4) generated overlay on 1024-px GAN background
"""

import os, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------- CONFIG ----------------------------------
ROOT_GOOD   = "data/good"
ROOT_DEFECT = "data/defect"
GEN_WEIGHTS = "outputs/fake_good_v6/generator_good_v6.pt"

RES_TRAIN   = 256          # 256 → fastest; 512 = sharper, slower
RES_FULL    = 1024         # GAN native size
EPOCHS      = 12
BATCH       = 4            # keep small on CPU
LR          = 1e-3

THRESH      = 0.12         # higher threshold → ignore faint edges
SPARSE_W    = 5e-3         # mask L1 sparsity
EDGE_W      = 1.0          # Sobel-edge consistency
TV_W        = 0.0          # TV off by default

PREVIEW_N   = 4            # columns in preview grid

OUT_DIR     = "outputs/defect_overlay_cpu_v3"
CKPT_DIR    = f"{OUT_DIR}/ckpt_v3"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count() - 1, 1))
DEV = torch.device("cpu")

# -------------------- transforms & helpers ---------------------------
class PadSquare:
    """Pad a PIL image to a square, keeping content centred."""
    def __init__(self, size): self.size = size
    def __call__(self, img):
        w, h = img.size
        side = max(w, h, self.size)
        pad  = [(side-w)//2, (side-h)//2,
                side-w-(side-w)//2, side-h-(side-h)//2]
        return T.functional.pad(img, pad, padding_mode='reflect')

xform = T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN, RES_TRAIN),
             interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

ALLOWED = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]

@torch.no_grad()
def make_overlay(defect, bg):
    """Return (mask×delta , mask) after threshold + morph cleanup."""
    delta = defect - bg
    mask  = (delta.abs() > THRESH).float()

    # simple 3×3 closing (dilation ∘ erosion) to clean speckles
    if mask.sum() > 0:
        mask_img = T.ToPILImage()(mask)
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))  # dilate
        mask_img = mask_img.filter(ImageFilter.MinFilter(3))  # erode
        mask     = T.ToTensor()(mask_img)

    return mask * delta, mask

# ------------------------- DATASET -----------------------------------
class OverlayDS(Dataset):
    def __init__(self):
        self.good   = list_imgs(ROOT_GOOD)
        self.defect = list_imgs(ROOT_DEFECT)
        if not self.good or not self.defect:
            raise RuntimeError("Empty data/good or data/defect folder.")
        self.N = max(len(self.good), len(self.defect))
    def __len__(self): return self.N
    def __getitem__(self, _):
        g = xform(Image.open(random.choice(self.good  )).convert("L"))
        d = xform(Image.open(random.choice(self.defect)).convert("L"))
        ov, mask = make_overlay(d, g)
        return g, ov, mask

# --------------------- BACKGROUND GAN --------------------------------
NZ, NGF = 100, 64
def make_generator():
    layers = [nn.ConvTranspose2d(NZ, NGF*16, 4, 1, 0, bias=False),
              nn.BatchNorm2d(NGF*16), nn.ReLU(True)]
    oc = NGF*16
    for _ in range(int(math.log2(RES_FULL) - 2)):
        ic, oc = oc, max(NGF, oc//2)
        layers += [nn.ConvTranspose2d(ic, oc, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(oc), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(oc, 1, 3, 1, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

Gbg = make_generator().to(DEV).eval()
Gbg.load_state_dict(torch.load(GEN_WEIGHTS, map_location=DEV))

@torch.no_grad()
def sample_gan(n):
    z  = torch.randn(n, NZ, 1, 1, device=DEV)
    bg_full = Gbg(z)
    bg_small = F.interpolate(bg_full, (RES_TRAIN, RES_TRAIN),
                             mode="bilinear", align_corners=False) \
               if RES_TRAIN != RES_FULL else bg_full
    return bg_full, bg_small

# --------------------- Tiny-U-Net (3-level) ---------------------------
def down(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.LeakyReLU(0.2,True))
def up(i,o):   return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.ReLU(True))

class TinyUNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.e1 = down(1, ch)
        self.e2 = down(ch, ch*2)
        self.e3 = down(ch*2, ch*4)
        self.d3 = up(ch*4,   ch*2)
        self.d2 = up(ch*4,   ch)
        self.d1 = nn.ConvTranspose2d(ch*2, 2, 4, 2, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2)
        d3 = self.d3(e3)
        d2 = self.d2(torch.cat([d3, e2], 1))
        d1 = self.d1(torch.cat([d2, e1], 1))
        mask  = torch.sigmoid(d1[:, :1])
        delta = torch.tanh(   d1[:, 1:])
        return mask * delta, mask

# --------------------- Loss helpers ----------------------------------
SOBEL_X = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=DEV).view(1,1,3,3)/8
SOBEL_Y = SOBEL_X.transpose(2,3)

def grad_mag(img):
    gx = F.conv2d(img, SOBEL_X, padding=1)
    gy = F.conv2d(img, SOBEL_Y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

def tv_loss(m):
    return F.l1_loss(m[:,:,:,1:], m[:,:,:,:-1]) + F.l1_loss(m[:,:,1:,:], m[:,:,:-1,:])

# -------------------------- TRAIN ------------------------------------
def main():
    ds = OverlayDS()
    loader = DataLoader(ds, BATCH, shuffle=True, num_workers=0)

    net = TinyUNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), LR)
    L1, BCE = nn.L1Loss(), nn.BCELoss()

    for ep in range(1, EPOCHS+1):
        for bg, ov_gt, mask_gt in tqdm(loader, desc=f"Epoch {ep}/{EPOCHS}"):
            bg, ov_gt, mask_gt = [t.to(DEV) for t in (bg, ov_gt, mask_gt)]
            ov_pred, mask_pred = net(bg)

            loss = ( L1(ov_pred, ov_gt) +
                     BCE(mask_pred, mask_gt) +
                     EDGE_W   * L1(grad_mag(ov_pred), grad_mag(ov_gt)) +
                     SPARSE_W * mask_pred.mean() )
            if TV_W: loss += TV_W * tv_loss(mask_pred)

            opt.zero_grad(); loss.backward(); opt.step()

        # ---------------------- preview --------------------------------
        with torch.no_grad():
            bg_full, bg_small = sample_gan(PREVIEW_N)
            ov_pred_small, _ = net(bg_small)
            ov_pred_full = F.interpolate(ov_pred_small, (RES_FULL, RES_FULL),
                                         mode="bilinear", align_corners=False)
            synth_full = torch.clamp(bg_full + ov_pred_full, -1, 1)

            real_bg, real_ov, _ = next(iter(loader))
            real_bg, real_ov = real_bg.to(DEV)[:PREVIEW_N], real_ov.to(DEV)[:PREVIEW_N]
            real_bg_full  = F.interpolate(real_bg, (RES_FULL, RES_FULL),
                                          mode="bilinear", align_corners=False)
            real_ov_full  = F.interpolate(real_ov, (RES_FULL, RES_FULL),
                                          mode="bilinear", align_corners=False)

            white = torch.ones_like(real_bg_full)
            true_on_white = torch.clamp(white + real_ov_full, -1, 1)
            pred_on_white = torch.clamp(white + ov_pred_full,   -1, 1)
            real_defect   = torch.clamp(real_bg_full + real_ov_full, -1, 1)

            rows = [real_defect, true_on_white, pred_on_white, synth_full]
            grids = [vutils.make_grid(r, nrow=PREVIEW_N, padding=2) for r in rows]
            preview = torch.cat(grids, dim=1)          # stack rows

            vutils.save_image(preview*0.5+0.5,
                              f"{OUT_DIR}/side_by_side_{ep:02}.png",
                              normalize=False)

        torch.save(net.state_dict(), f"{CKPT_DIR}/unet_{ep:02}.pt")
        print(f"Epoch {ep} finished – preview + checkpoint saved.")

    print("✅ all epochs done – see", OUT_DIR)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
# =====================================================================
