# ===================== defect_overlay_cpu_v6.py ======================
"""
CPU-only overlay trainer (fast) with:
  • Tiny-U-Net (3 levels, 16-32-64 ch) at 256 px  – change RES_TRAIN if needed
  • Robust background subtraction:
        – best-match good → defect pairing (L2 @ 64 px)
        – high-pass diff  (img − GaussianBlur)
        – adaptive thresh  τ = μ + k·σ  (k = 2)
        – closing 3×3  → opening 5×5  (pure PyTorch)
  • Loss =  L1 + BCE + EDGE_W*edge-L1 + SPARSE_W*mean(mask)
            + DICE_W*(1 − DiceCoef) (+ optional TV)
"""

import os, math, random, itertools
from pathlib import Path
from functools import lru_cache

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import numpy as np

# --------------------------- CONFIG ----------------------------------
ROOT_GOOD   = "data/good"
ROOT_DEFECT = "data/defect"
GEN_WEIGHTS = "outputs/fake_good_v6/generator_good_v6.pt"

RES_TRAIN   = 256
RES_FULL    = 1024
EPOCHS      = 12
BATCH       = 4
LR          = 1e-3

THRESH_K    = 2.0          # k · σ + μ adaptive threshold
SPARSE_W    = 5e-3
EDGE_W      = 1.0
DICE_W      = 0.2
TV_W        = 0.0

PREVIEW_N   = 4

OUT_DIR     = "outputs/defect_overlay_cpu_v6"
CKPT_DIR    = f"{OUT_DIR}/ckpt_v6"
os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count() - 1, 1))
DEV = torch.device("cpu")

# --------------------- transforms & helpers --------------------------
class PadSquare:
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

# ----- tiny utils ----------------------------------------------------
@lru_cache(maxsize=1)
def _gauss_kernel(ch=1, k=5, σ=1.0):
    ax = torch.arange(k) - k//2
    g  = torch.exp(-(ax**2)/(2*σ**2))
    g2d = (g[:,None]@g[None,:]).float()
    g2d /= g2d.sum()
    ker = g2d[None,None]      # 1×1×k×k
    ker = ker.repeat(ch,1,1,1)
    return ker

def gaussian_blur(img, k=5, σ=1):
    """
    2-D Gaussian blur that works for (C,H,W) or (N,C,H,W) tensors.
    """
    added_batch = img.dim() == 3          # (C,H,W) → add batch dim
    if added_batch:
        img = img.unsqueeze(0)            # (1,C,H,W)

    ch = img.size(1)                      # correct #channels (1 for gray)
    ker = _gauss_kernel(ch, k, σ).to(img)

    out = F.conv2d(img, ker, padding=k // 2, groups=ch)

    return out.squeeze(0) if added_batch else out

def high_pass(img, k=5, σ=1):
    return img - gaussian_blur(img, k, σ)

def dice_coeff(pred, target, eps=1e-6):
    num = 2*(pred*target).sum()
    den = pred.sum() + target.sum() + eps
    return num/den

# --------------------- DATASET ---------------------------------------
ALLOWED = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]

# ----- tiny utils ----------------------------------------------------
def quick_vec(path, side=64):
    """
    Return a 64×64 float tensor in [0,1] used for the nearest-neighbour search.
    (No gradients needed, so plain CPU Tensor is fine.)
    """
    img = Image.open(path).convert('L').resize((side, side), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # H × W, float32
    return torch.from_numpy(arr)                         # Tensor 64 × 64

class OverlayDS(Dataset):
    def __init__(self):
        self.good   = list_imgs(ROOT_GOOD)
        self.defect = list_imgs(ROOT_DEFECT)
        if not self.good or not self.defect:
            raise RuntimeError("Empty data/good or data/defect folder.")

        # ---- pair each defect with the closest good (L2 @ 64 px) ----
        g_vecs = torch.stack([quick_vec(p) for p in self.good])
        self.pairs = []
        for d_path in self.defect:
            d_vec = quick_vec(d_path)
            idx = torch.argmin(((g_vecs - d_vec)**2).view(len(self.good), -1).mean(1))
            self.pairs.append((d_path, self.good[idx]))
        # Upsample to roughly balance positives / pure-good samples
        self.good_only = self.good
        self.N = len(self.pairs) + len(self.good_only)

    def __len__(self): return self.N

    def __getitem__(self, i):
        # 50 % chance of a defect pair, 50 % chance of pure good
        if i < len(self.pairs):
            d_path, g_path = self.pairs[i]
            defect = xform(Image.open(d_path).convert("L"))
            good   = xform(Image.open(g_path).convert("L"))
            ov, mask = make_overlay(defect, good)
            return good, ov, mask
        else:                     # pure good → zero overlay
            g = xform(Image.open(self.good_only[i - len(self.pairs)]).convert("L"))
            z = torch.zeros_like(g)
            return g, z, z

# --------------------- overlay creation ------------------------------
def make_overlay(defect, bg):
    """Return (mask×delta_hp , mask) with adaptive thresh & morph."""
    delta_hp = high_pass(defect, k=5, σ=1) - high_pass(bg, k=5, σ=1)
    μ, σ = delta_hp.abs().mean(), delta_hp.abs().std()
    τ = μ + THRESH_K*σ

    mask = (delta_hp.abs() > τ).float()
    # closing 3×3 → opening 5×5  (pool-based)
    mask = F.max_pool2d(mask, 3, 1, 1)             # dilate
    mask = -F.max_pool2d(-mask, 3, 1, 1)           # erode (closing)
    mask = -F.max_pool2d(-mask, 5, 1, 2)           # erode
    mask = F.max_pool2d(mask, 5, 1, 2)             # dilate (opening)

    return mask*delta_hp, mask

# --------------------- BACKGROUND GAN (unchanged) --------------------
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

# --------------------- Tiny-U-Net (unchanged) ------------------------
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
        return mask*delta, mask

# --------------------------- TRAIN -----------------------------------
SOBEL_X = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=DEV).view(1,1,3,3)/8
SOBEL_Y = SOBEL_X.transpose(2,3)

def grad_mag(img):
    gx = F.conv2d(img, SOBEL_X, padding=1)
    gy = F.conv2d(img, SOBEL_Y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

def tv_loss(m):
    return F.l1_loss(m[:,:,:,1:], m[:,:,:,:-1]) + F.l1_loss(m[:,:,1:,:], m[:,:,:-1,:])

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

            dice = dice_coeff(mask_pred, mask_gt)
            loss = ( L1(ov_pred, ov_gt) +
                     BCE(mask_pred, mask_gt) +
                     EDGE_W   * L1(grad_mag(ov_pred), grad_mag(ov_gt)) +
                     SPARSE_W * mask_pred.mean() +
                     DICE_W   * (1 - dice) )
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
            real_bg = real_bg.to(DEV)[:PREVIEW_N]
            real_ov = real_ov.to(DEV)[:PREVIEW_N]
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
            preview = torch.cat(grids, dim=1)

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
